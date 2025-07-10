#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import torch
import joblib
import numpy as np

# 确保这些导入与您的项目结构匹配
try:
    # 假设 antioxidant_predictor_5 包含了 AntioxidantPredictor 类
    from antioxidant_predictor_5 import AntioxidantPredictor
    # 假设 feature_extract 包含了 ProtT5Model 和 extract_features 函数
    from feature_extract import ProtT5Model, extract_features
except ImportError as e:
    print(f"导入模块失败，请确保 antioxidant_predictor_5.py 和 feature_extract.py 在正确的路径下: {e}")
    exit()

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

def predict_fasta(args):
    """
    使用与训练脚本(predictor_train.py)对齐的逻辑进行序列预测。
    """
    seq_records = parse_fasta(args.fasta_file)
    if not seq_records: 
        print("未解析到任何有效序列，程序退出。")
        return
    
    headers, sequences_list = zip(*seq_records) 
    print(f"共读取 {len(sequences_list)} 条序列。")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 1. 初始化与训练时相同的 ProtT5 特征提取器 ---
    # 这是关键的修改，确保特征提取方法与训练时完全一致
    try:
        if args.finetuned_prott5_checkpoint and os.path.exists(args.finetuned_prott5_checkpoint):
            print(f"特征提取将使用【微调后】的 ProtT5 模型: {args.finetuned_prott5_checkpoint}")
            protT5_extractor = ProtT5Model(
                model_path=args.protT5_model_base_path,
                finetuned_model_file=args.finetuned_prott5_checkpoint
            )
        else:
            print("特征提取将使用【原始】的 ProtT5 模型。")
            protT5_extractor = ProtT5Model(model_path=args.protT5_model_base_path)
        print("ProtT5 特征提取器加载成功。")
    except Exception as e:
        print(f"加载 ProtT5 特征提取器失败: {e}")
        return

    # --- 2. 加载预测模型和对应的Scaler ---
    # 加载下游的抗氧化肽预测模型
    model = AntioxidantPredictor(args.input_dim, args.transformer_layers, args.transformer_heads, args.transformer_dropout)
    try:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        temp_info = f"T={model.get_temperature():.4f}" if hasattr(model, 'get_temperature') else "无温度参数"
        print(f"加载预测模型 {args.model_checkpoint} 成功 ({temp_info})。")
    except Exception as e:
        print(f"加载模型权重失败: {e}。请确保模型架构参数与检查点匹配。")
        return
    model.to(device)
    model.eval()
    
    # 加载与模型匹配的Scaler
    try: 
        scaler = joblib.load(args.scaler_path)
        print(f"加载Scaler ({type(scaler).__name__}) 成功。")
    except Exception as e:
        print(f"加载Scaler失败: {e}")
        return
        
    # --- 3. 批量进行特征提取和预测 ---
    all_probs, all_classes = [], []
    num_sequences = len(sequences_list)

    for i in range(0, num_sequences, args.batch_size):
        batch_seqs = sequences_list[i : i + args.batch_size] 
        batch_headers = headers[i : i + args.batch_size]
        batch_feats = []
        
        print(f"处理批次 {i//args.batch_size + 1}/{ (num_sequences + args.batch_size - 1)//args.batch_size }...")
        for seq, header in zip(batch_seqs, batch_headers):
            if not seq or not isinstance(seq, str) or not seq.strip():
                print(f"警告：序列头 '{header}' 对应的序列为空，已跳过。")
                continue
            try:
                # 使用与您第一个脚本中相同的调用方式，包含L_fixed和d_model_pe
                feat = extract_features(seq, protT5_extractor, 
                                        L_fixed=args.L_fixed_pe_val,
                                        d_model_pe=args.d_model_pe_val)
                
                if feat.shape[0] != args.input_dim: 
                    print(f"警告：序列 '{header}' 特征维度 ({feat.shape[0]}) 与模型输入维度 ({args.input_dim}) 不符。已跳过。")
                    continue
                batch_feats.append(feat)
            except Exception as e:
                print(f"警告：序列 '{header}' 特征提取失败: {e}。已跳过。")
        
        if not batch_feats:
            # 如果整个批次都失败了，用NaN填充结果以保持行数对应
            all_probs.extend([float('nan')] * len(batch_seqs))
            all_classes.extend([-1] * len(batch_seqs))
            continue
            
        # 对成功提取的特征进行归一化和预测
        feats_np = np.array(batch_feats)
        try:
            scaled_feats = scaler.transform(feats_np)
        except Exception as e:
            print(f"批次 {i//args.batch_size+1} 归一化失败: {e}")
            all_probs.extend([float('nan')] * len(batch_seqs))
            all_classes.extend([-1] * len(batch_seqs))
            continue
        
        with torch.no_grad():
            logits = model(torch.tensor(scaled_feats, dtype=torch.float32).to(device)) 
            probs_t = torch.sigmoid(logits).squeeze()
        
        probs_batch_np = [probs_t.item()] if probs_t.ndim == 0 else probs_t.cpu().numpy().tolist()
        
        all_probs.extend(probs_batch_np)
        all_classes.extend([1 if p >= args.threshold else 0 for p in probs_batch_np])

    # --- 4. 保存结果 ---
    df = pd.DataFrame({
        "header": headers, 
        "sequence": sequences_list, 
        "predicted_probability": all_probs, 
        "predicted_class": all_classes
    })
    df_sorted = df.sort_values(by="predicted_probability", ascending=False, na_position='last')
    try:
        df_sorted.to_csv(args.output_file, index=False)
        print(f"\n预测结果已成功保存至: {args.output_file}")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用与训练脚本对齐的方法，对FASTA文件中的序列进行抗氧化活性预测。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 核心输入/输出参数 ---
    parser.add_argument("--fasta_file", required=True, help="待预测的FASTA格式序列文件。")
    parser.add_argument("--output_file", default="prediction_results.csv", help="保存预测结果的CSV文件名。")

    # --- 模型和Scaler路径参数 (需要根据训练时是否微调来选择) ---
    parser.add_argument("--model_checkpoint", default="checkpoints/final_rl_model_logitp0.1_calibrated_FINETUNED_PROTT5.pth", help="已训练的抗氧化肽预测器模型检查点文件(.pth)。")
    parser.add_argument("--scaler_path", default="checkpoints/scaler_FINETUNED_PROTT5.pkl", help="与模型匹配的特征缩放器文件(.pkl)。")

    # --- ProtT5 特征提取相关参数 ---
    parser.add_argument("--protT5_model_base_path", default="/root/shared-nvme/chshhan/diffusion/prott5/model/", help="原始ProtT5模型文件所在的【目录】。")
    parser.add_argument("--finetuned_prott5_checkpoint", default="/root/shared-nvme/chshhan/diffusion/prott5/model/finetuned_prott5.bin", help="【可选】微调后的ProtT5模型检查点文件(.bin)。如果提供此路径且文件存在，则使用微调模型提取特征。")

    # --- 预测和模型架构参数 (应与训练时保持一致) ---
    parser.add_argument("--threshold", type=float, default=0.5, help="判定为正类的概率阈值。")
    parser.add_argument("--batch_size", type=int, default=32, help="预测时的批处理大小。")
    parser.add_argument("--input_dim", type=int, default=1914, help="模型输入的特征维度。")
    parser.add_argument("--transformer_layers", type=int, default=3, help="预测器中Transformer的层数。")
    parser.add_argument("--transformer_heads", type=int, default=4, help="预测器中Transformer的头数。")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="预测器中Transformer的dropout率。")
    parser.add_argument("--L_fixed_pe_val", type=int, default=29, help="特征提取中位置编码的L_fixed值。")
    parser.add_argument("--d_model_pe_val", type=int, default=16, help="特征提取中位置编码的d_model_pe值。")
    
    args = parser.parse_args()

    # --- 路径检查 ---
    print("--- 检查路径和文件 ---")
    paths_to_check = {
        "FASTA文件": args.fasta_file,
        "预测模型检查点": args.model_checkpoint,
        "Scaler文件": args.scaler_path,
        "ProtT5基础模型目录": args.protT5_model_base_path,
    }
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"错误: {name} 路径不存在 -> {path}")
            exit(1)
            
    if args.finetuned_prott5_checkpoint and not os.path.exists(args.finetuned_prott5_checkpoint):
        print(f"警告: 您提供了微调ProtT5模型的路径，但该文件不存在 -> {args.finetuned_prott5_checkpoint}")
        print("程序将继续，但会使用【原始】ProtT5模型进行特征提取。")
        args.finetuned_prott5_checkpoint = None # 将其设为None以确保后续逻辑正确

    print("--- 参数配置检查完成 --- \n")
    
    predict_fasta(args)
