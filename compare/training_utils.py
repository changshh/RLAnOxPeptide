import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from train.antioxidant_predictor_5 import AntioxidantPredictor


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
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=finetune_params.get("batch_size", 64),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        ),
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
        total_loss = 0.0
        for batch_X, batch_y_hard in train_loader:
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
            total_loss += loss.item()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                probs = torch.sigmoid(model(batch_X_val.to(device)))
                val_preds.extend(np.atleast_1d((probs.squeeze().cpu().numpy() > 0.5).astype(int)))
                val_labels.extend(np.atleast_1d(batch_y_val.squeeze().cpu().numpy().astype(int)))

        current_f1 = f1_score(val_labels, val_preds, zero_division=0) if val_labels else 0.0
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"[SL] Epoch {epoch + 1}/{num_epochs}: TrainLoss={avg_loss:.4f}, Val F1={current_f1:.4f}")

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
