import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from transformers import T5EncoderModel, T5Tokenizer
from peft import PeftModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from train.feature_extract import extract_features
    from training_utils import train_feature_fusion_classifier
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LoRAProtT5Extractor:
    """Load base ProtT5 and merge LoRA adapter for feature encoding."""
    def __init__(self, base_model_path, lora_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - Loading base ProtT5 model: {base_model_path}")
        try:
            base_model = T5EncoderModel.from_pretrained(base_model_path)
        except Exception as e:
            print(f"Failed to load base model: {e}")
            sys.exit(1)

        print(f"  - Loading LoRA adapter: {lora_adapter_path}")
        try:
            lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        except Exception as e:
            print(f"Failed to load LoRA adapter: {e}")
            sys.exit(1)

        print("  - Merging LoRA weights...")
        self.model = lora_model.merge_and_unload().to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.model.eval()
        print("  - LoRA ProtT5 extractor ready.")

    def encode(self, sequence):
        seq_spaced = " ".join(list(sequence))
        encoded_input = self.tokenizer(seq_spaced, return_tensors="pt", padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            embedding = self.model(**encoded_input).last_hidden_state
        return embedding.squeeze(0).cpu().numpy()


def read_fasta_sequences(fasta_file):
    sequences = []
    with open(fasta_file, "r") as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
                continue
            current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences


def read_fasta_with_label_from_header(fasta_file):
    sequences = []
    labels = []
    with open(fasta_file, "r") as f:
        current_seq = []
        current_label = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                label = None
                if "|" in header:
                    last_field = header.split("|")[-1]
                    if last_field in ["0", "1"]:
                        label = int(last_field)
                if label is None and header and header[0] in ["0", "1"]:
                    label = int(header[0])
                if current_seq:
                    sequences.append("".join(current_seq))
                    labels.append(current_label if current_label is not None else 0)
                    current_seq = []
                current_label = label if label is not None else 0
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
            labels.append(current_label if current_label is not None else 0)
    return sequences, labels


def load_anoxpp_test(pos_path, neg_path):
    pos_seqs = read_fasta_sequences(pos_path)
    neg_seqs = read_fasta_sequences(neg_path)
    sequences = pos_seqs + neg_seqs
    labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)
    return sequences, labels


def load_antiox_xls(xls_path, sheet_name="Antiox_train"):
    df = pd.read_excel(xls_path, sheet_name=sheet_name)
    df = df.dropna(subset=["Sequence", "Label"])
    sequences = df["Sequence"].astype(str).tolist()
    labels = df["Label"].astype(int).tolist()
    return sequences, labels


def prepare_features_with_lora_from_sequences(sequences, labels, prott5_extractor_instance, test_seq_set=None, seed=42):
    from sklearn.preprocessing import RobustScaler

    combined = list(zip(sequences, labels))
    rng = np.random.RandomState(seed)
    rng.shuffle(combined)
    sequences, labels = zip(*combined)

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.1, random_state=seed, stratify=labels
    )

    removed_count = 0
    if test_seq_set is not None:
        filtered_train_seqs = []
        filtered_train_labels = []
        for seq, label in zip(train_seqs, train_labels):
            if seq in test_seq_set:
                removed_count += 1
                continue
            filtered_train_seqs.append(seq)
            filtered_train_labels.append(label)
        train_seqs, train_labels = filtered_train_seqs, filtered_train_labels
        print(f"  - Removed duplicates from training set: {removed_count}")

    print(f"  - Train size: {len(train_seqs)}, Val size: {len(val_seqs)}")

    def process_data(seqs_list, desc_text):
        feature_list = []
        for s_item in tqdm(seqs_list, desc=desc_text):
            features = extract_features(s_item, prott5_extractor_instance)
            feature_list.append(features)
        return np.array(feature_list)

    X_train = process_data(train_seqs, "  - Extracting train features")
    X_val = process_data(val_seqs, "  - Extracting val features")

    scaler = RobustScaler()
    print("  - Scaling features with RobustScaler...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, np.array(train_labels), np.array(val_labels), scaler


class TemperatureCalibrator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def calibrate(self, X_val, y_val, initial_temp=1.0, lr=0.01, n_iters=100):
        self.model.set_temperature(initial_temp, self.device)
        print(f"Calibrating temperature (T_initial={initial_temp:.4f})...")
        self.model.to(self.device)
        self.model.eval()
        self.model.temperature.requires_grad = True
        criterion_cal = nn.BCEWithLogitsLoss().to(self.device)
        optimizer_temp = optim.Adam([self.model.temperature], lr=lr)
        val_inputs = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        val_labels_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        for _ in range(n_iters):
            optimizer_temp.zero_grad()
            calibrated_logits = self.model(val_inputs)
            loss = criterion_cal(calibrated_logits, val_labels_tensor)
            loss.backward()
            optimizer_temp.step()
        print(f"Temperature calibration done. T = {self.model.temperature.item():.4f}")
        self.model.temperature.requires_grad = False
        return self.model


def evaluate_model_with_threshold_custom(model, X, y, threshold):
    model.eval()
    device = next(model.parameters()).device
    current_temp = model.get_temperature() if hasattr(model, "get_temperature") else 1.0
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        preds = (probs > threshold).astype(int)
    auc = roc_auc_score(y, probs)
    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    acc = accuracy_score(y, preds)
    print(f"Eval (T={current_temp:.2f}, thr={threshold}): AUC={auc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Acc={acc:.4f}")
    return {"AUC-ROC": auc, "F1-Score": f1, "Precision": prec, "Recall": rec, "Accuracy": acc}


def compute_confusion_metrics(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return mcc, sensitivity, specificity


def supervised_training(X_train, y_train, X_val, y_val, params, calibrate_temp=True):
    device = params["device"]
    print("\n--- Step 3.1: Supervised training (SL) ---")
    model = train_feature_fusion_classifier(X_train, y_train, X_val, y_val, params)
    print("Supervised training finished.")
    if calibrate_temp:
        print("Calibrating temperature...")
        calibrator = TemperatureCalibrator(model, device)
        model = calibrator.calibrate(X_val, y_val)
        print("Temperature calibration finished.")
    _ = evaluate_model_with_threshold_custom(model, X_val, y_val, params["threshold"])
    return model


def run_for_testset(model_name, test_seqs, test_labels, train_seqs, train_labels, prott5_extractor, seed, output_dir):
    test_seq_set = set(test_seqs)

    X_train, X_val, y_train, y_val, scaler = prepare_features_with_lora_from_sequences(
        train_seqs, train_labels, prott5_extractor, test_seq_set=test_seq_set, seed=seed
    )

    scaler_path = os.path.join(output_dir, f"scaler_lora_{model_name}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    sup_params = {
        "input_dim": X_train.shape[1],
        "transformer_layers": 3,
        "transformer_heads": 4,
        "transformer_dropout": 0.1,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "threshold": 0.5,
        "patience": 15,
        "weight_decay": 1e-5,
        "label_smoothing": 0.1,
        "logit_penalty_weight": 0.05,
    }

    print("\n=== Train predictor (SL only) ===")
    sl_model = supervised_training(X_train, y_train, X_val, y_val, sup_params)

    final_model_path = os.path.join(output_dir, f"final_predictor_sl_only_{model_name}.pth")
    torch.save(sl_model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path} (T={sl_model.get_temperature():.4f})")

    print(f"\n=== Evaluate on {model_name} test set ===")
    X_test_list = []
    for s_test in tqdm(test_seqs, desc=f"  - Extracting {model_name} test features"):
        X_test_list.append(extract_features(s_test, prott5_extractor))
    X_test = np.array(X_test_list)
    X_test_scaled = scaler.transform(X_test)

    test_metrics = evaluate_model_with_threshold_custom(
        sl_model, X_test_scaled, np.array(test_labels), threshold=sup_params["threshold"]
    )

    with torch.no_grad():
        preds = (
            torch.sigmoid(sl_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(sup_params["device"])))
            .squeeze()
            .cpu()
            .numpy()
            > sup_params["threshold"]
        ).astype(int)

    mcc, sensitivity, specificity = compute_confusion_metrics(preds, np.array(test_labels))
    return {
        "Models": model_name,
        "Accuracy": test_metrics["Accuracy"],
        "AUROC": test_metrics["AUC-ROC"],
        "MCC": mcc,
        "Precision": test_metrics["Precision"],
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }, int(len(y_train)), int(y_train.sum())


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Step 1: Load LoRA ProtT5 extractor ===")
    prott5_extractor = LoRAProtT5Extractor(args.base_model_path, args.lora_adapter_path)

    print("\n=== Step 2: Load training XLS ===")
    train_seqs, train_labels = load_antiox_xls(args.train_xls, sheet_name=args.train_sheet)

    print("\n=== Step 3: Load test sets ===")
    anoxpp_seqs, anoxpp_labels = load_anoxpp_test(args.anoxpp_pos_path, args.anoxpp_neg_path)
    if args.aopp_test_fasta and os.path.exists(args.aopp_test_fasta):
        aopp_seqs, aopp_labels = read_fasta_with_label_from_header(args.aopp_test_fasta)
    else:
        print(
            "AOPP FASTA was not provided; loading the AOPP-compatible test data "
            f"from sheet '{args.aopp_test_sheet}' in {args.train_xls}."
        )
        aopp_seqs, aopp_labels = load_antiox_xls(args.train_xls, sheet_name=args.aopp_test_sheet)

    table2_rows = []
    table3_rows = []

    metrics, n_train, n_pos = run_for_testset(
        "AnOxPP", anoxpp_seqs, anoxpp_labels, train_seqs, train_labels, prott5_extractor, args.seed, args.output_dir
    )
    table3_rows.append(metrics)
    table2_rows.append({
        "Models": "AnOxPP",
        "Number of sequences": n_train,
        "AOPs": n_pos,
        "Non-AOPs": n_train - n_pos,
    })

    metrics, n_train, n_pos = run_for_testset(
        "AOPP", aopp_seqs, aopp_labels, train_seqs, train_labels, prott5_extractor, args.seed, args.output_dir
    )
    table3_rows.append(metrics)
    table2_rows.append({
        "Models": "AOPP",
        "Number of sequences": n_train,
        "AOPs": n_pos,
        "Non-AOPs": n_train - n_pos,
    })

    table2_path = os.path.join(args.output_dir, "table2.csv")
    table3_path = os.path.join(args.output_dir, "table3.csv")
    pd.DataFrame(table2_rows).to_csv(table2_path, index=False)
    pd.DataFrame(table3_rows).to_csv(table3_path, index=False)
    print(f"\nTable2 saved: {table2_path}")
    print(f"Table3 saved: {table3_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the seed 70877 benchmark comparison.")

    parser.add_argument("--seed", type=int, default=70877)
    parser.add_argument(
        "--train_xls",
        type=str,
        default=os.path.join(CURRENT_DIR, "Antiox_dataset.xls"),
    )
    parser.add_argument("--train_sheet", type=str, default="Antiox_train")
    parser.add_argument("--aopp_test_sheet", type=str, default="Antiox_test")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./prott5/model/",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default="./lora_finetuned_prott5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "checkpoints"),
    )
    parser.add_argument(
        "--anoxpp_pos_path",
        type=str,
        default=os.path.join(CURRENT_DIR, "test_AnOxPs.txt"),
    )
    parser.add_argument(
        "--anoxpp_neg_path",
        type=str,
        default=os.path.join(CURRENT_DIR, "test_non-AnOxPs.txt"),
    )
    parser.add_argument(
        "--aopp_test_fasta",
        type=str,
        default="",
    )

    args = parser.parse_args()
    main(args)
