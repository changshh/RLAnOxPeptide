# predictor_with_lora.py
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from peft import PeftModel
    from transformers import T5EncoderModel, T5Tokenizer

    try:
        from antioxidant_predictor_5 import AntioxidantPredictor
        from feature_extract import extract_features
    except ImportError:
        from train.antioxidant_predictor_5 import AntioxidantPredictor
        from train.feature_extract import extract_features
except ImportError as e:
    print(f"Import failed. Check transformers, peft, scikit-learn, joblib, pandas, torch, and biopython: {e}")
    raise


def parse_fasta(fasta_file):
    """Parse FASTA records into (header, sequence) tuples."""
    sequences_data = []
    header = None
    current_sequence_lines = []
    try:
        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None and current_sequence_lines:
                        sequence_str = "".join(current_sequence_lines)
                        if sequence_str:
                            sequences_data.append((header, sequence_str))
                    header = line[1:].strip()
                    current_sequence_lines = []
                elif header is not None:
                    current_sequence_lines.append(line)
            if header is not None and current_sequence_lines:
                sequence_str = "".join(current_sequence_lines)
                if sequence_str:
                    sequences_data.append((header, sequence_str))
    except FileNotFoundError:
        print(f"FASTA file not found: {fasta_file}")
        return []
    if not sequences_data:
        print(f"No sequences were parsed from FASTA file: {fasta_file}")
    return sequences_data


class LoRAProtT5Extractor:
    """Load a ProtT5 encoder with a LoRA adapter and expose an encode method."""

    def __init__(self, base_model_path, lora_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading base ProtT5 model: {base_model_path}")
        base_model = T5EncoderModel.from_pretrained(base_model_path)

        if not os.path.exists(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter directory not found: {lora_adapter_path}")

        print(f"Loading LoRA adapter: {lora_adapter_path}")
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

        print("Merging LoRA weights for inference.")
        self.model = lora_model.merge_and_unload().to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.model.eval()

    def encode(self, sequence):
        sequence_spaced = " ".join(list(sequence))
        encoded_input = self.tokenizer(sequence_spaced, return_tensors="pt", padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            embedding = self.model(**encoded_input).last_hidden_state
        return embedding.squeeze(0).cpu().numpy()


def predict_fasta_with_lora(args):
    seq_records = parse_fasta(args.fasta_file)
    if not seq_records:
        print("No valid sequences found; exiting.")
        return

    headers, sequences_list = zip(*seq_records)
    print(f"Loaded {len(sequences_list)} sequences.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        protT5_extractor = LoRAProtT5Extractor(
            base_model_path=args.protT5_model_base_path,
            lora_adapter_path=args.lora_adapter_path,
        )
    except Exception as e:
        print(f"Failed to load LoRA ProtT5 feature extractor: {e}")
        return

    model = AntioxidantPredictor(
        args.input_dim,
        args.transformer_layers,
        args.transformer_heads,
        args.transformer_dropout,
    )
    try:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        print(f"Loaded predictor checkpoint: {args.model_checkpoint}")
    except Exception as e:
        print(f"Failed to load predictor checkpoint: {e}")
        return
    model.to(device)
    model.eval()

    try:
        scaler = joblib.load(args.scaler_path)
        print(f"Loaded scaler: {args.scaler_path}")
    except Exception as e:
        print(f"Failed to load scaler: {e}")
        return

    all_probs, all_classes = [], []
    for i in tqdm(range(0, len(sequences_list), args.batch_size), desc="Predicting"):
        batch_seqs = sequences_list[i : i + args.batch_size]
        batch_feats = [
            extract_features(seq, protT5_extractor, L_fixed=args.L_fixed_pe_val, d_model_pe=args.d_model_pe_val)
            for seq in batch_seqs
        ]

        scaled_feats = scaler.transform(np.array(batch_feats))

        with torch.no_grad():
            logits = model(torch.tensor(scaled_feats, dtype=torch.float32).to(device))
            probs_t = torch.sigmoid(logits).squeeze()

        probs_batch_np = [probs_t.item()] if probs_t.ndim == 0 else probs_t.cpu().numpy().tolist()
        all_probs.extend(probs_batch_np)
        all_classes.extend([1 if p >= args.threshold else 0 for p in probs_batch_np])

    df = pd.DataFrame(
        {
            "header": headers,
            "sequence": sequences_list,
            "predicted_probability": all_probs,
            "predicted_class": all_classes,
        }
    )
    df.sort_values(by="predicted_probability", ascending=False, inplace=True)
    df.to_csv(args.output_file, index=False)
    print(f"Prediction results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict antioxidant peptide activity from a FASTA file.")
    parser.add_argument("--fasta_file", required=True, help="Input FASTA file.")
    parser.add_argument("--output_file", default="prediction_results_with_lora.csv", help="Output CSV file.")
    parser.add_argument("--protT5_model_base_path", default="./prott5/model/", help="Base ProtT5 model directory.")
    parser.add_argument("--lora_adapter_path", default="./lora_finetuned_prott5", help="LoRA adapter directory.")
    parser.add_argument(
        "--model_checkpoint",
        default="./predictor_sl_checkpoints/final_predictor_sl_only.pth",
        help="Predictor checkpoint file.",
    )
    parser.add_argument("--scaler_path", default="./predictor_sl_checkpoints/scaler_lora.pkl", help="Feature scaler file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    parser.add_argument("--batch_size", type=int, default=32, help="Prediction batch size.")
    parser.add_argument("--input_dim", type=int, default=1914, help="Predictor input dimension.")
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--L_fixed_pe_val", type=int, default=29)
    parser.add_argument("--d_model_pe_val", type=int, default=16)

    predict_fasta_with_lora(parser.parse_args())
