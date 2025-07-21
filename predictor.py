# predictor_with_lora.py
import os
import argparse
import pandas as pd
import torch
import joblib
import numpy as np
from tqdm import tqdm

# 确保这些库已安装
try:
    from transformers import T5EncoderModel, T5Tokenizer
    from peft import PeftModel
    from antioxidant_predictor_5 import AntioxidantPredictor
    from feature_extract import extract_features
except ImportError as e:
    print(f"导入模块失败，请确保已安装 transformers, peft, scikit-learn, joblib, pandas, torch, biopython。错误: {e}")
    print("同时，请确保 antioxidant_predictor_5.py 和 feature_extract.py 文件在当前目录。")
    exit()

# =====================================================================================
#  !! 关键修正点 1 !!
#  将 parse_fasta 函数的定义直接包含在本文件中
# =====================================================================================
def parse_fasta(fasta_file):
    """从FASTA文件中解析序列头和序列本身"""
    sequences_data = [] 
    header = None
    current_sequence_lines = [] 
    try:
        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if header is not None and current_sequence_lines: 
                        sequence_str = "".join(current_sequence_lines)
                        if sequence_str: sequences_data.append((header, sequence_str))
                    header = line[1:].strip() 
                    current_sequence_lines = [] 
                elif header is not None: 
                    current_sequence_lines.append(line)
            if header is not None and current_sequence_lines: 
                sequence_str = "".join(current_sequence_lines)
                if sequence_str: sequences_data.append((header, sequence_str))
    except FileNotFoundError:
        print(f"错误：FASTA 文件 {fasta_file} 未找到。"); return []
    if not sequences_data: print(f"警告：从FASTA文件 {fasta_file} 未解析到任何序列。")
    return sequences_data

class LoRAProtT5Extractor:
    """一个包装类，用于加载带LoRA的ProtT5模型并提供encode接口"""
    def __init__(self, base_model_path, lora_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - 正在加载基础ProtT5模型: {base_model_path}")
        base_model = T5EncoderModel.from_pretrained(base_model_path)
        
        if not os.path.exists(lora_adapter_path):
            raise FileNotFoundError(f"错误: 未找到LoRA适配器目录: {lora_adapter_path}")
            
        print(f"  - 正在加载并应用LoRA适配器: {lora_adapter_path}")
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        print("  - 正在合并LoRA权重以提高推理速度...")
        self.model = lora_model.merge_and_unload().to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.model.eval()
        print("  - LoRA增强的特征提取器准备就绪。")

    def encode(self, sequence):
        """使用合并后的模型进行编码(特征提取)"""
        sequence_spaced = " ".join(list(sequence))
        encoded_input = self.tokenizer(sequence_spaced, return_tensors='pt', padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            embedding = self.model(**encoded_input).last_hidden_state
        return embedding.squeeze(0).cpu().numpy()

def predict_fasta_with_lora(args):
    """使用LoRA增强的模型进行序列预测"""
    seq_records = parse_fasta(args.fasta_file)
    if not seq_records:
        print("未解析到任何有效序列，程序退出。")
        return
    
    headers, sequences_list = zip(*seq_records)
    print(f"共读取 {len(sequences_list)} 条序列。")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 1. 初始化LoRA增强的ProtT5特征提取器 ---
    try:
        protT5_extractor = LoRAProtT5Extractor(
            base_model_path=args.protT5_model_base_path,
            lora_adapter_path=args.lora_adapter_path
        )
    except Exception as e:
        print(f"加载LoRA增强的ProtT5特征提取器失败: {e}"); return

    # --- 2. 加载下游的预测器头部模型和Scaler ---
    model = AntioxidantPredictor(args.input_dim, args.transformer_layers, args.transformer_heads, args.transformer_dropout)
    try:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        print(f"加载预测器头部模型 {args.model_checkpoint} 成功。")
    except Exception as e:
        print(f"加载模型权重失败: {e}"); return
    model.to(device); model.eval()
    
    try: 
        scaler = joblib.load(args.scaler_path)
        print(f"加载Scaler成功。")
    except Exception as e:
        print(f"加载Scaler失败: {e}"); return
        
    # --- 3. 批量进行特征提取和预测 ---
    all_probs, all_classes = [], []
    for i in tqdm(range(0, len(sequences_list), args.batch_size), desc="正在预测"):
        batch_seqs = sequences_list[i : i + args.batch_size]
        batch_feats = [extract_features(seq, protT5_extractor, L_fixed=args.L_fixed_pe_val, d_model_pe=args.d_model_pe_val) for seq in batch_seqs]
        
        scaled_feats = scaler.transform(np.array(batch_feats))
        
        with torch.no_grad():
            logits = model(torch.tensor(scaled_feats, dtype=torch.float32).to(device))
            probs_t = torch.sigmoid(logits).squeeze()
        
        probs_batch_np = [probs_t.item()] if probs_t.ndim == 0 else probs_t.cpu().numpy().tolist()
        
        all_probs.extend(probs_batch_np)
        all_classes.extend([1 if p >= args.threshold else 0 for p in probs_batch_np])

    # --- 4. 保存结果 ---
    df = pd.DataFrame({
        "header": headers, "sequence": sequences_list, 
        "predicted_probability": all_probs, "predicted_class": all_classes
    })
    df.sort_values(by="predicted_probability", ascending=False, inplace=True)
    df.to_csv(args.output_file, index=False)
    print(f"\n预测结果已成功保存至: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LoRA微调后的ProtT5模型，对FASTA文件中的序列进行抗氧化活性预测。")
    
    parser.add_argument("--fasta_file", required=True, help="待预测的FASTA格式序列文件。")
    parser.add_argument("--output_file", default="prediction_results_with_lora.csv", help="保存预测结果的CSV文件名。")
    parser.add_argument("--protT5_model_base_path", default="/root/shared-nvme/chshhan/diffusion/prott5/model/", help="原始ProtT5模型文件所在的【目录】。")
    parser.add_argument("--lora_adapter_path", default="./lora_finetuned_prott5", help="【必需】已训练好的LoRA适配器所在的【目录】。")
    parser.add_argument("--model_checkpoint", default="/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/no_over_confident/predictor_with_lora_checkpoints/final_predictor_with_lora.pth", help="已训练的【预测器头部】模型检查点文件(.pth)。")
    parser.add_argument("--scaler_path", default="/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/no_over_confident/predictor_with_lora_checkpoints/scaler_lora.pkl", help="与模型匹配的特征缩放器文件(.pkl)。")
    parser.add_argument("--threshold", type=float, default=0.5, help="判定为正类的概率阈值。")
    parser.add_argument("--batch_size", type=int, default=32, help="预测时的批处理大小。")
    parser.add_argument("--input_dim", type=int, default=1914, help="模型输入的特征维度。")
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--L_fixed_pe_val", type=int, default=29)
    parser.add_argument("--d_model_pe_val", type=int, default=16)

    args = parser.parse_args()
    predict_fasta_with_lora(args)
