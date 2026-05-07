# train_predictor_sl_only.py
import os
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import copy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from transformers import T5EncoderModel, T5Tokenizer
from peft import PeftModel

# 确保所有自定义模块都可以被导入
try:
    from feature_extract import extract_features, load_fasta_with_labels
    from antioxidant_predictor_5 import AntioxidantPredictor
    # 移除了 RL 相关的导入
    from pretrain_finetune import train_feature_fusion_classifier
except ImportError as e:
    print(f"导入模块失败，请确保所有依赖的.py文件都在当前目录: {e}")
    exit()

# =====================================================================================

class LoRAProtT5Extractor:
    """一个包装类，用于加载带LoRA的ProtT5模型并提供encode接口"""
    def __init__(self, base_model_path, lora_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - 正在加载基础ProtT5模型: {base_model_path}")
        try:
            base_model = T5EncoderModel.from_pretrained(base_model_path)
        except Exception as e:
            print(f"无法加载基础模型，请检查路径: {e}")
            exit()
        
        print(f"  - 正在加载并应用LoRA适配器: {lora_adapter_path}")
        try:
            lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        except Exception as e:
            print(f"无法加载LoRA适配器，请检查路径: {e}")
            exit()
        
        print("  - 正在合并LoRA权重...")
        self.model = lora_model.merge_and_unload().to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.model.eval()
        print("  - LoRA增强的特征提取器准备就绪。")

    def encode(self, sequence):
        """使用合并后的模型进行编码"""
        # ProtT5 需要在氨基酸之间加空格
        sequence_spaced = " ".join(list(sequence))
        encoded_input = self.tokenizer(sequence_spaced, return_tensors='pt', padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            embedding = self.model(**encoded_input).last_hidden_state
        return embedding.squeeze(0).cpu().numpy()

def prepare_features_with_lora(neg_fasta, pos_fasta, prott5_extractor_instance):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    import random

    print("  - 正在加载FASTA文件...")
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

    # 划分训练集和验证集
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    print(f"  - 训练集样本数: {len(train_seqs)}, 验证集样本数: {len(val_seqs)}")
    
    def process_data(seqs_list, desc_text):
        feature_list = []
        for s_item in tqdm(seqs_list, desc=desc_text):
            features = extract_features(s_item, prott5_extractor_instance)
            feature_list.append(features)
        return np.array(feature_list)

    X_train = process_data(train_seqs, "  - 提取训练集特征中")
    X_val = process_data(val_seqs, "  - 提取验证集特征中")
    
    scaler = RobustScaler()
    print("  - 正在使用RobustScaler进行特征归一化...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled, np.array(train_labels), np.array(val_labels), scaler


# =====================================================================================
class TemperatureCalibrator:
    def __init__(self, model, device): self.model = model; self.device = device
    def calibrate(self, X_val, y_val, initial_temp=1.0, lr=0.01, n_iters=100):
        self.model.set_temperature(initial_temp, self.device)
        print(f"开始温度校准 (T_initial={initial_temp:.4f})...")
        self.model.to(self.device); self.model.eval(); self.model.temperature.requires_grad = True
        criterion_cal = nn.BCEWithLogitsLoss().to(self.device)
        optimizer_temp = optim.Adam([self.model.temperature], lr=lr)
        val_inputs = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        val_labels_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        for _ in range(n_iters):
            optimizer_temp.zero_grad(); calibrated_logits = self.model(val_inputs)
            loss = criterion_cal(calibrated_logits, val_labels_tensor); loss.backward(); optimizer_temp.step()
        print(f"温度校准完成。优化后 T = {self.model.temperature.item():.4f}")
        self.model.temperature.requires_grad = False; return self.model

def evaluate_model_with_threshold_custom(model, X, y, threshold, model_outputs_logits=True):
    model.eval(); device = next(model.parameters()).device
    current_temp = model.get_temperature() if hasattr(model,'get_temperature') and callable(model.get_temperature) else 1.0
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        preds = (probs > threshold).astype(int)
    auc = roc_auc_score(y, probs)
    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    acc = accuracy_score(y, preds)
    print(f"评估 (T={current_temp:.2f}, thr={threshold}): AUC={auc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Acc={acc:.4f}")
    return {"AUC-ROC":auc, "F1-Score":f1, "Precision":prec, "Recall":rec, "Accuracy":acc}

def supervised_training(X_train, y_train, X_val, y_val, params, calibrate_temp=True):
    device = params["device"]; print("\n--- 步骤 3.1: 开始纯监督训练 (SL) ---")
    model = train_feature_fusion_classifier(X_train, y_train, X_val, y_val, params)
    print("纯监督训练结束。")
    if calibrate_temp:
        print("对纯监督模型进行温度校准...");
        calibrator = TemperatureCalibrator(model, device)
        model = calibrator.calibrate(X_val, y_val)
        print("纯监督模型温度校准完成。")
    val_metrics = evaluate_model_with_threshold_custom(model, X_val, y_val, params["threshold"])
    return model, val_metrics

# =====================================================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== 步骤 1: 加载带LoRA的特征提取器 ===")
    prott5_extractor = LoRAProtT5Extractor(args.base_model_path, args.lora_adapter_path)

    print("\n=== 步骤 2: 准备特征 ===")
    X_train, X_val, y_train, y_val, scaler = prepare_features_with_lora(
        args.neg_fasta_path, args.pos_fasta_path, prott5_extractor
    )
    scaler_save_path = os.path.join(args.output_dir, "scaler_lora.pkl")
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler 已保存至: {scaler_save_path}")

    print("\n=== 步骤 3: 训练预测器 (仅监督学习) ===")
    sup_params = {
        'input_dim': X_train.shape[1], 'transformer_layers': 3, 'transformer_heads': 4, 
        'transformer_dropout': 0.1, 'batch_size': 64, 'learning_rate': 1e-4, 
        'num_epochs': 50, 'device': device, 'threshold': 0.5, 'patience': 15, 
        'weight_decay': 1e-5, 'label_smoothing': 0.1, 'logit_penalty_weight': 0.05
    }
    
    # 训练监督模型
    sl_model, _ = supervised_training(X_train, y_train, X_val, y_val, sup_params)
    
    # 保存最终监督模型
    final_model_path = os.path.join(args.output_dir, "final_predictor_sl_only.pth")
    torch.save(sl_model.state_dict(), final_model_path)
    print(f"最终监督模型已保存至: {final_model_path} (T={sl_model.get_temperature():.4f})")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("\n=== 步骤 4: 在独立测试集上评估最终模型 ===")
    if os.path.exists(args.independent_test_fasta):
        print(f"正在加载独立测试集: {args.independent_test_fasta}")
        test_seqs, test_labels_list = load_fasta_with_labels(args.independent_test_fasta)
        if test_seqs:
            test_labels = np.array(test_labels_list)
            
            # 使用与训练时相同的LoRA特征提取器
            X_test_list = []
            for s_test in tqdm(test_seqs, desc="  - 提取独立测试集特征中"):
                 X_test_list.append(extract_features(s_test, prott5_extractor))
            
            X_test = np.array(X_test_list)
            X_test_scaled = scaler.transform(X_test)

            print("\n--- 最终模型在独立测试集上的表现 ---")
            # 使用 sl_model 进行评估
            final_metrics = evaluate_model_with_threshold_custom(sl_model, X_test_scaled, test_labels, threshold=sup_params["threshold"])
            print("最终模型独立测试集指标:", final_metrics)
        else:
            print("独立测试集为空，跳过评估。")
    else:
        print(f"警告: 独立测试集文件未找到: {args.independent_test_fasta}。跳过最终评估。")
        
    print("\n=== 训练和评估流程全部完成！ ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LoRA微调后的ProtT5训练监督学习预测器。")
    
    # --- 路径参数 ---
    parser.add_argument("--base_model_path", type=str, default="./prott5/model/", help="原始ProtT5模型目录。")
    parser.add_argument("--lora_adapter_path", type=str, default="./lora_finetuned_prott5", help="已保存的LoRA适配器目录。")
    parser.add_argument("--neg_fasta_path", type=str, default="./data/remaining_negative.fasta")
    parser.add_argument("--pos_fasta_path", type=str, default="./data/remaining_positive.fasta")
    parser.add_argument("--output_dir", type=str, default="./predictor_sl_checkpoints", help="保存最终预测器和相关文件的目录。")
    
    # 新增的独立测试集路径参数
    parser.add_argument("--independent_test_fasta", type=str, default="./data/independent_test_cleaned.fasta", help="用于最终评估的独立测试集FASTA文件。")
    
    args = parser.parse_args()
    main(args)
