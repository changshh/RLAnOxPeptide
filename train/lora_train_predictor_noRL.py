# train_predictor_sl_only.py
import argparse
import copy
import os
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

try:
    from feature_extract import extract_features, load_fasta_with_labels
    from antioxidant_predictor_5 import AntioxidantPredictor
except ImportError as e:
    print(f"Import failed. Check local training modules and dependencies: {e}")
    raise

class LoRAProtT5Extractor:
    def __init__(self, base_model_path, lora_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading base ProtT5 model: {base_model_path}")
        try:
            base_model = T5EncoderModel.from_pretrained(base_model_path)
        except Exception as e:
            print(f"Failed to load base model: {e}")
            raise

        print(f"Loading LoRA adapter: {lora_adapter_path}")
        try:
            lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        except Exception as e:
            print(f"Failed to load LoRA adapter: {e}")
            raise

        print("Merging LoRA weights.")
        self.model = lora_model.merge_and_unload().to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.model.eval()
        print("LoRA ProtT5 extractor is ready.")

    def encode(self, sequence):
        sequence_spaced = " ".join(list(sequence))
        encoded_input = self.tokenizer(sequence_spaced, return_tensors="pt", padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            embedding = self.model(**encoded_input).last_hidden_state
        return embedding.squeeze(0).cpu().numpy()

def prepare_features_with_lora(neg_fasta, pos_fasta, prott5_extractor_instance):
    print("Loading FASTA training data.")
    neg_seqs, _ = load_fasta_with_labels(neg_fasta)
    pos_seqs, _ = load_fasta_with_labels(pos_fasta)

    neg_labels = [0] * len(neg_seqs)
    pos_labels = [1] * len(pos_seqs)
    sequences = neg_seqs + pos_seqs
    labels = neg_labels + pos_labels

    combined = list(zip(sequences, labels))
    random.seed(42)
    random.shuffle(combined)
    sequences, labels = zip(*combined)

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )
    print(f"Train size: {len(train_seqs)}, validation size: {len(val_seqs)}")

    def process_data(seqs_list, desc_text):
        feature_list = []
        for s_item in tqdm(seqs_list, desc=desc_text):
            features = extract_features(s_item, prott5_extractor_instance)
            feature_list.append(features)
        return np.array(feature_list)

    X_train = process_data(train_seqs, "Extracting train features")
    X_val = process_data(val_seqs, "Extracting validation features")

    scaler = RobustScaler()
    print("Scaling features with RobustScaler.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, np.array(train_labels), np.array(val_labels), scaler

class TemperatureCalibrator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def calibrate(self, X_val, y_val, initial_temp=1.0, lr=0.01, n_iters=100):
        self.model.set_temperature(initial_temp, self.device)
        print(f"Starting temperature calibration (T_initial={initial_temp:.4f})...")
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
        print(f"Temperature calibration finished. T = {self.model.temperature.item():.4f}")
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
    print(f"Evaluation (T={current_temp:.2f}, thr={threshold}): AUC={auc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Acc={acc:.4f}")
    return {"AUC-ROC": auc, "F1-Score": f1, "Precision": prec, "Recall": rec, "Accuracy": acc}

def train_feature_fusion_classifier(X_train, y_train, X_val, y_val, finetune_params):
    model = AntioxidantPredictor(
        input_dim=X_train.shape[1],
        transformer_layers=finetune_params.get("transformer_layers", 3),
        transformer_heads=finetune_params.get("transformer_heads", 4),
        transformer_dropout=finetune_params.get("transformer_dropout", 0.1),
    )
    device = finetune_params.get("device", "cpu")
    model.to(device)
    model.set_temperature(1.0, device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=finetune_params.get("learning_rate", 1e-4),
        weight_decay=finetune_params.get("weight_decay", 1e-5),
    )
    criterion_bce = nn.BCEWithLogitsLoss()
    label_smoothing = finetune_params.get("label_smoothing", 0.1)
    logit_penalty_weight = finetune_params.get("logit_penalty_weight", 0.01)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
        batch_size=finetune_params.get("batch_size", 64),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)),
        batch_size=finetune_params.get("batch_size", 64),
        shuffle=False,
    )

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    num_epochs = finetune_params.get("num_epochs", 10)
    patience = finetune_params.get("patience", 15)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        for batch_X, batch_y_hard in train_loader:
            num_batches += 1
            batch_X = batch_X.to(device)
            batch_y_hard = batch_y_hard.to(device)
            if label_smoothing > 0:
                batch_y = batch_y_hard * (1.0 - label_smoothing) + (1.0 - batch_y_hard) * label_smoothing
            else:
                batch_y = batch_y_hard

            optimizer.zero_grad()
            logits = model(batch_X)
            bce_loss = criterion_bce(logits, batch_y)
            logit_penalty = torch.mean(logits ** 2) if logit_penalty_weight > 0 else torch.tensor(0.0, device=device)
            loss = bce_loss + logit_penalty_weight * logit_penalty
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                probs = torch.sigmoid(model(batch_X_val.to(device)))
                val_preds.extend(np.atleast_1d((probs.squeeze().cpu().numpy() > 0.5).astype(int)))
                val_labels.extend(np.atleast_1d(batch_y_val.squeeze().cpu().numpy().astype(int)))

        current_f1 = f1_score(val_labels, val_preds, zero_division=0) if val_labels else 0.0
        avg_train_loss = total_train_loss / num_batches if num_batches else 0.0
        print(f"[SL] Epoch {epoch + 1}/{num_epochs}: TrainLoss={avg_train_loss:.4f}, Val F1={current_f1:.4f}")

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def supervised_training(X_train, y_train, X_val, y_val, params, calibrate_temp=True):
    print("Starting supervised predictor training.")
    model = train_feature_fusion_classifier(X_train, y_train, X_val, y_val, params)
    print("Supervised predictor training finished.")
    if calibrate_temp:
        print("Calibrating supervised predictor.")
        calibrator = TemperatureCalibrator(model, params["device"])
        model = calibrator.calibrate(X_val, y_val)
        print("Calibration finished.")
    val_metrics = evaluate_model_with_threshold_custom(model, X_val, y_val, params["threshold"])
    return model, val_metrics

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading LoRA ProtT5 feature extractor.")
    prott5_extractor = LoRAProtT5Extractor(args.base_model_path, args.lora_adapter_path)

    print("Preparing training features.")
    X_train, X_val, y_train, y_val, scaler = prepare_features_with_lora(
        args.neg_fasta_path,
        args.pos_fasta_path,
        prott5_extractor,
    )
    scaler_save_path = os.path.join(args.output_dir, "scaler_lora.pkl")
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to: {scaler_save_path}")

    sup_params = {
        "input_dim": X_train.shape[1],
        "transformer_layers": 3,
        "transformer_heads": 4,
        "transformer_dropout": 0.1,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "device": device,
        "threshold": 0.5,
        "patience": 15,
        "weight_decay": 1e-5,
        "label_smoothing": 0.1,
        "logit_penalty_weight": 0.05,
    }

    sl_model, _ = supervised_training(X_train, y_train, X_val, y_val, sup_params)

    final_model_path = os.path.join(args.output_dir, "final_predictor_sl_only.pth")
    torch.save(sl_model.state_dict(), final_model_path)
    print(f"Final supervised model saved to: {final_model_path} (T={sl_model.get_temperature():.4f})")

    print("Evaluating on the independent test set if available.")
    if os.path.exists(args.independent_test_fasta):
        print(f"Loading independent test set: {args.independent_test_fasta}")
        test_seqs, test_labels_list = load_fasta_with_labels(args.independent_test_fasta)
        if test_seqs:
            test_labels = np.array(test_labels_list)
            X_test_list = []
            for s_test in tqdm(test_seqs, desc="Extracting independent test features"):
                X_test_list.append(extract_features(s_test, prott5_extractor))

            X_test = np.array(X_test_list)
            X_test_scaled = scaler.transform(X_test)

            print("Independent test performance:")
            final_metrics = evaluate_model_with_threshold_custom(sl_model, X_test_scaled, test_labels, threshold=sup_params["threshold"])
            print("Independent test metrics:", final_metrics)
        else:
            print("Independent test set is empty; skipping evaluation.")
    else:
        print(f"Independent test set not found: {args.independent_test_fasta}. Skipping final evaluation.")

    print("Training and evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the supervised RLAnOxPeptide predictor.")
    parser.add_argument("--base_model_path", type=str, default="./prott5/model/", help="Base ProtT5 model directory.")
    parser.add_argument("--lora_adapter_path", type=str, default="./lora_finetuned_prott5", help="LoRA adapter directory.")
    parser.add_argument("--neg_fasta_path", type=str, default="./data/remaining_negative.fasta")
    parser.add_argument("--pos_fasta_path", type=str, default="./data/remaining_positive.fasta")
    parser.add_argument("--output_dir", type=str, default="./predictor_sl_checkpoints", help="Output checkpoint directory.")
    parser.add_argument("--independent_test_fasta", type=str, default="./data/independent_test_cleaned.fasta", help="Independent test FASTA file.")

    main(parser.parse_args())
