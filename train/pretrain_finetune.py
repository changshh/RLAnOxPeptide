#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import f1_score

# 确保能导入AntioxidantPredictor
try:
    from antioxidant_predictor_5 import AntioxidantPredictor 
except ImportError:
    print("错误: 无法导入 AntioxidantPredictor。请确保 antioxidant_predictor_5.py 在PYTHONPATH中。")
    # 定义一个虚拟的AntioxidantPredictor以便脚本能运行（用于结构占位）
    class AntioxidantPredictor(nn.Module):
        def __init__(self, input_dim, **kwargs):
            super(AntioxidantPredictor, self).__init__()
            self.fc = nn.Linear(input_dim, 1)
            self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
            print("警告: 使用了虚拟的AntioxidantPredictor定义。")
        def forward(self, x): return self.fc(x) / self.temperature
        def set_temperature(self,val,dev): self.temperature.data = torch.tensor([val],device=dev)
        def get_temperature(self): return self.temperature.item()


# --- 全局变量定义 (ProtT5 微调相关) ---
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYXUBZ" 
VOCAB = {aa: i + 1 for i, aa in enumerate(list(AMINO_ACIDS))} 
VOCAB['[PAD]'] = 0  
VOCAB['[UNK]'] = len(VOCAB) +1 
VOCAB['[MASK]'] = len(VOCAB) +1 
VOCAB_SIZE = len(VOCAB) +1 

# --- ProtT5 微调相关的数据集和模型定义 ---
class AntioxidantProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer_vocab, max_length=512, mask_prob=0.15):
        self.sequences = sequences; self.vocab = tokenizer_vocab; self.max_length = max_length
        self.mask_prob = mask_prob; self.pad_token_id = self.vocab['[PAD]']
        self.mask_token_id = self.vocab['[MASK]']; self.unk_token_id = self.vocab['[UNK]']
        self.aa_to_id = {aa: self.vocab.get(aa, self.unk_token_id) for aa in AMINO_ACIDS}
    def __len__(self): return len(self.sequences)
    def tokenize_sequence(self, seq): return [self.aa_to_id.get(aa.upper(), self.unk_token_id) for aa in seq]
    def mask_tokens(self, token_ids):
        input_ids = list(token_ids); labels = [-100] * len(token_ids)
        for i, token_id in enumerate(token_ids):
            if token_id == self.pad_token_id: continue
            if random.random() < self.mask_prob:
                labels[i] = token_id
                prob = random.random()
                if prob < 0.8: input_ids[i] = self.mask_token_id
                elif prob < 0.9: pass
                else: input_ids[i] = random.choice([self.vocab[aa] for aa in AMINO_ACIDS if aa in self.vocab and aa not in ['[PAD]', '[UNK]', '[MASK]']])
        return input_ids, labels
    def __getitem__(self, idx):
        seq = self.sequences[idx]; token_ids_original = self.tokenize_sequence(seq)
        token_ids_padded = token_ids_original[:self.max_length] if len(token_ids_original) > self.max_length else token_ids_original + [self.pad_token_id] * (self.max_length - len(token_ids_original))
        input_ids, labels = self.mask_tokens(token_ids_padded)
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long), 'labels': torch.tensor(labels, dtype=torch.long), 
                'attention_mask': torch.tensor([1 if id != self.pad_token_id else 0 for id in input_ids], dtype=torch.long)}

class ProtT5ForFineTuning(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, num_heads=8, dropout=0.1, padding_idx=0):
        super(ProtT5ForFineTuning, self).__init__(); self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
    def forward(self, input_ids, attention_mask=None):
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.embedding(input_ids); encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.mlm_head(encoder_output)

def finetune_protT5_on_antioxidant_proteins(protein_csv, model_params, model_path="checkpoints/"):
    print(f"开始 ProtT5 微调，从 {protein_csv} 加载数据...")
    try:
        df = pd.read_csv(protein_csv)
        if df.empty or df.columns[0] not in df : raise ValueError("CSV错误或为空")
        sequences = df.iloc[:, 0].dropna().astype(str).tolist();
        if not sequences: raise ValueError("CSV中无序列")
        print(f"从CSV加载了 {len(sequences)} 条序列用于微调。")
    except Exception as e: print(f"加载或解析 protein_csv 文件失败: {e}"); sequences = ["PEPTIDEEXAMPLEFORFINETUNINGONLY"] 

    global VOCAB, VOCAB_SIZE 
    dataset = AntioxidantProteinDataset(sequences, VOCAB, model_params.get("max_length",256), model_params.get("mask_prob",0.15))
    dataloader = DataLoader(dataset, batch_size=model_params.get("batch_size",8), shuffle=True, num_workers=model_params.get("num_workers",0)) # num_workers for DataLoader
    device = model_params.get("device","cpu") # Default to CPU if not specified
    model = ProtT5ForFineTuning(VOCAB_SIZE, model_params.get("embed_dim",256), model_params.get("num_layers",3),
                                model_params.get("num_heads",4), model_params.get("dropout",0.1), VOCAB['[PAD]']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_params.get("learning_rate",5e-5))
    criterion = nn.CrossEntropyLoss(ignore_index=-100); num_epochs = model_params.get("num_epochs",1) # Default to 1 epoch for quick test
    
    print(f"在设备 {device} 上开始微调 ProtT5ForFineTuning 模型 ({num_epochs} epochs)...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0; num_batches = 0
        for batch_data in dataloader:
            num_batches += 1
            input_ids = batch_data['input_ids'].to(device)
            labels = batch_data['labels'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            optimizer.zero_grad(); logits = model(input_ids, attention_mask); 
            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            loss.backward(); optimizer.step(); total_loss += loss.item()
        avg_epoch_loss = total_loss/num_batches if num_batches > 0 else 0
        print(f"[ProtT5 微调] Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
    
    os.makedirs(model_path, exist_ok=True); save_path = os.path.join(model_path, "finetuned_prott5.bin")
    torch.save(model.state_dict(), save_path); print(f"微调后的 ProtT5 模型已保存至: {save_path}"); return model
# ---------------------------------------------------------------------

def train_feature_fusion_classifier(X_train, y_train, X_val, y_val, finetune_params):
    input_dim = X_train.shape[1]
    model = AntioxidantPredictor(
        input_dim=input_dim,
        transformer_layers=finetune_params.get('transformer_layers', 3),
        transformer_heads=finetune_params.get('transformer_heads', 4),
        transformer_dropout=finetune_params.get('transformer_dropout', 0.1)
    )
    device = finetune_params.get("device", "cpu")
    model.to(device)
    model.set_temperature(1.0, device) # 监督训练时T=1.0
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=finetune_params.get("learning_rate", 1e-4), 
                           weight_decay=finetune_params.get("weight_decay", 1e-5))
    criterion_bce = nn.BCEWithLogitsLoss()
    
    label_smoothing_epsilon = finetune_params.get("label_smoothing", 0.1) 
    logit_penalty_weight = finetune_params.get("logit_penalty_weight", 0.01) 

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                           torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
                              batch_size=finetune_params.get("batch_size", 64), shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                         torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)),
                            batch_size=finetune_params.get("batch_size", 64), shuffle=False)

    best_val_f1 = 0.0; patience_counter = 0; best_state = None
    num_epochs = finetune_params.get("num_epochs", 10)
    patience = finetune_params.get("patience", 15)

    print(f"开始监督训练，标签平滑 eps={label_smoothing_epsilon if label_smoothing_epsilon > 0 else '无'}, Logit惩罚权重={logit_penalty_weight if logit_penalty_weight > 0 else '无'}")

    for epoch in range(num_epochs):
        model.train(); total_train_loss, total_bce_loss, total_logit_reg_loss = 0.0, 0.0, 0.0
        num_batches_train = 0
        for batch_X, batch_y_hard in train_loader: 
            num_batches_train +=1
            batch_X, batch_y_hard = batch_X.to(device), batch_y_hard.to(device)
            batch_y_smooth = batch_y_hard * (1.0-label_smoothing_epsilon) + (1.0-batch_y_hard) * label_smoothing_epsilon if label_smoothing_epsilon > 0 else batch_y_hard
            
            optimizer.zero_grad()
            outputs_logits = model(batch_X) # T=1.0, so these are raw logits
            bce_loss = criterion_bce(outputs_logits, batch_y_smooth)
            
            logit_reg_loss_term = torch.mean(outputs_logits**2) if logit_penalty_weight > 0 else torch.tensor(0.0, device=device)
            loss = bce_loss + logit_penalty_weight * logit_reg_loss_term
            
            loss.backward(); optimizer.step()
            total_train_loss += loss.item(); total_bce_loss += bce_loss.item()
            if logit_penalty_weight > 0: total_logit_reg_loss += logit_reg_loss_term.item()

        avg_train_loss = total_train_loss/num_batches_train if num_batches_train > 0 else 0
        avg_bce_loss = total_bce_loss/num_batches_train if num_batches_train > 0 else 0
        avg_logit_reg_loss = total_logit_reg_loss/num_batches_train if num_batches_train > 0 and logit_penalty_weight > 0 else 0
        
        model.eval(); all_preds_val, all_labels_val = [], []
        with torch.no_grad():
            for batch_X_val, batch_y_val_hard in val_loader:
                outputs_logits_val = model(batch_X_val.to(device))
                probs_val = torch.sigmoid(outputs_logits_val) 
                all_preds_val.extend(np.atleast_1d((probs_val.squeeze().cpu().numpy() > 0.5).astype(int)))
                all_labels_val.extend(np.atleast_1d(batch_y_val_hard.squeeze().cpu().numpy().astype(int)))
        
        current_f1 = 0.0
        if len(all_labels_val) > 0 and len(np.unique(all_labels_val)) > 1:
            current_f1 = f1_score(all_labels_val, all_preds_val, zero_division=0)
        elif len(all_labels_val) > 0 and np.array_equal(all_labels_val, all_preds_val):
             # Simplified F1 for single class case (can be 1 if all correct for that class, or 0)
            unique_label_val = np.unique(all_labels_val)[0]
            current_f1 = f1_score(all_labels_val, all_preds_val, pos_label=unique_label_val, zero_division=0)


        print(f"[监督训练] Epoch {epoch+1}/{num_epochs}: TrainLoss={avg_train_loss:.4f} (BCE={avg_bce_loss:.4f}, LogitReg={avg_logit_reg_loss:.4f}), Val F1={current_f1:.4f}")
        
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1; best_state = copy.deepcopy(model.state_dict()); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: print(f"早停在监督训练 epoch {epoch+1}。"); break
    
    if best_state: model.load_state_dict(best_state)
    return model

if __name__ == "__main__":
    print("--- ProtT5 微调演示 (如果CSV文件存在) ---")
    dummy_csv_path = "dummy_data/dummy_protein_finetune.csv" # 确保dummy_data目录存在
    os.makedirs("dummy_data", exist_ok=True)
    if not os.path.exists(dummy_csv_path): 
        pd.DataFrame({'sequence':['PEPTIDEEXAMPLE']*20}).to_csv(dummy_csv_path, index=False)
    
    finetune_prot_params_main = {
        "device":"cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs":1, "max_length":30, "batch_size":2, 
        "embed_dim":32,"num_layers":1,"num_heads":1, "learning_rate":1e-4
    }
    demo_checkpoints_path = "demo_checkpoints/" # 微调模型保存路径
    # finetune_protT5_on_antioxidant_proteins(dummy_csv_path, finetune_prot_params_main, model_path=demo_checkpoints_path)


    print("\n--- 特征融合分类器训练演示 (增强Logit惩罚) ---")
    # 准备虚拟数据
    num_train_samples, num_val_samples, feature_dimension = 90, 10, 1914
    X_train_demo = np.random.rand(num_train_samples, feature_dimension).astype(np.float32)
    y_train_demo = np.random.randint(0, 2, num_train_samples).astype(np.float32)
    X_val_demo = np.random.rand(num_val_samples, feature_dimension).astype(np.float32)
    y_val_demo = np.random.randint(0, 2, num_val_samples).astype(np.float32)
    
    params_for_logit_penalty_demo = {
        'input_dim': feature_dimension, 
        'transformer_layers': 1, 'transformer_heads': 1, 'transformer_dropout': 0.1, 
        'batch_size': 16, 'learning_rate': 1e-3, 'num_epochs': 2, 
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'patience': 1, 'weight_decay': 1e-5, 
        'label_smoothing': 0.05, 
        'logit_penalty_weight': 0.05 # **演示使用的惩罚权重**
    }
    train_feature_fusion_classifier(X_train_demo, y_train_demo, X_val_demo, y_val_demo, params_for_logit_penalty_demo)
    print("演示完成。")
