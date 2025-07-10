#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import joblib
from sklearn.cluster import KMeans
import torch.nn as nn
from tqdm import tqdm

# 假设这些py文件与当前脚本在同一目录或PYTHONPATH中
try:
    from antioxidant_predictor_5 import AntioxidantPredictor
    from feature_extract import ProtT5Model as FeatureProtT5Model, extract_features
except ImportError as e:
    print(f"导入验证所需模块失败: {e}")
    print("请确保 antioxidant_predictor_5.py 和 feature_extract.py 文件在工作目录中。")
    exit()

# 定义统一词表（共22个 token）
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
token2id = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
token2id["<PAD>"] = 0
token2id["<EOS>"] = 1
id2token = {i: t for t, i in token2id.items()}
VOCAB_SIZE = len(token2id)


class ProtT5Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super(ProtT5Generator, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size
        self.eos_token_id = token2id["<EOS>"]
    def forward(self, input_ids):
        embeddings = self.embed_tokens(input_ids)
        encoder_output = self.encoder(embeddings)
        logits = self.lm_head(encoder_output)
        return logits
    def sample(self, batch_size, max_length=20, device="cpu", temperature=2.5, min_decoded_length=3):
        start_token = torch.randint(2, self.vocab_size, (batch_size, 1), device=device)
        generated = start_token
        for i in range(max_length - 1):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / temperature
            if generated.size(1) < min_decoded_length:
                next_logits[:, self.eos_token_id] = -float("inf")
            probs = torch.softmax(next_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample().unsqueeze(1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
    def decode(self, token_ids_batch):
        seqs = []
        for ids_tensor in token_ids_batch:
            seq = ""
            for token_id in ids_tensor.tolist()[1:]:
                if token_id == self.eos_token_id: break
                if token_id == token2id["<PAD>"]: continue
                seq += id2token.get(token_id, "?")
            seqs.append(seq)
        return seqs

def generate_unique_sequences(generator, device, num_seq, min_length=2, max_length=20, temperature=2.5, batch_size=32):
    generator.eval()
    unique_seqs = set()
    max_attempts = 10 
    attempts = 0
    while len(unique_seqs) < num_seq and attempts < max_attempts:
        generated_tokens = generator.sample(
            batch_size=batch_size, max_length=max_length, device=device,
            temperature=temperature, min_decoded_length=min_length
        )
        decoded = generator.decode(generated_tokens.cpu())
        for seq in decoded:
            if not seq: continue
            if not (min_length <= len(seq) <= max_length): continue
            unique_seqs.add(seq)
        attempts += 1
    return list(unique_seqs)


def cluster_sequences(generator, sequences, num_clusters, device):
    if not sequences or len(sequences) < num_clusters:
        print("警告：用于聚类的序列不足，将直接截取或返回。")
        return sequences[:num_clusters]
    with torch.no_grad():
        token_ids_list = []
        max_len = max(len(seq) for seq in sequences) + 1
        for seq in sequences:
            ids = [token2id.get(aa, 0) for aa in seq] + [generator.eos_token_id]
            ids += [token2id["<PAD>"]] * (max_len - len(ids))
            token_ids_list.append(ids)
        input_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)
        embeddings = generator.embed_tokens(input_ids)
        mask = (input_ids != token2id["<PAD>"]).unsqueeze(-1).float()
        embeddings = embeddings * mask
        lengths = mask.sum(dim=1)
        seq_embeds = embeddings.sum(dim=1) / (lengths + 1e-9)
        seq_embeds_np = seq_embeds.cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(seq_embeds_np)
    reps = []
    for i in range(num_clusters):
        idxs = np.where(kmeans.labels_ == i)[0]
        if len(idxs) == 0: continue
        cluster_embeds = seq_embeds_np[idxs]
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_embeds - center, axis=1)
        rep_idx = idxs[np.argmin(distances)]
        reps.append(sequences[rep_idx])
    return reps

def save_to_fasta_with_prob(results, output_path):
    with open(output_path, "w") as f:
        for i, (seq, prob) in enumerate(results):
            f.write(f">seq_{i+1}|probability={prob:.4f}\n")
            f.write(seq + "\n")

def load_predictor(model_path, device, input_dim, transformer_layers, transformer_heads, transformer_dropout):
    classifier = AntioxidantPredictor(input_dim, transformer_layers, transformer_heads, transformer_dropout)
    try:
        state_dict = torch.load(model_path, map_location=device)
        classifier.load_state_dict(state_dict)
        print(f"验证用分类器 {os.path.basename(model_path)} 加载成功。")
    except Exception as e:
        print(f"加载验证用分类器 {model_path} 失败: {e}"); raise
    classifier.to(device); classifier.eval(); return classifier

def run_validation_batch(classifier_model, seq_list, feature_prott5_instance, scaler_instance, device):
    if not seq_list: return []
    all_features, valid_indices = [], []
    for idx, seq_str in enumerate(seq_list):
        if not seq_str: continue
        try:
            feat = extract_features(seq_str, feature_prott5_instance)
            all_features.append(feat)
            valid_indices.append(idx)
        except Exception: pass
    if not all_features: return [None] * len(seq_list)
    features_np = np.array(all_features, dtype=np.float32)
    features_scaled = scaler_instance.transform(features_np)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        scaled_logits = classifier_model(features_tensor)
        probs_tensor = torch.sigmoid(scaled_logits).squeeze()
        if probs_tensor.ndim == 0: raw_probs_valid = [probs_tensor.item()]
        else: raw_probs_valid = probs_tensor.cpu().numpy().tolist()
    final_probs = [None] * len(seq_list)
    for i, original_idx in enumerate(valid_indices):
        final_probs[original_idx] = raw_probs_valid[i]
    return final_probs


def main():
    parser = argparse.ArgumentParser(description="生成指定数量的、经过验证的高质量抗氧化蛋白序列")
    parser.add_argument("--target_count", type=int, default=1000, help="最终需要生成的、验证后概率 > 90%% 的序列数量。")
    parser.add_argument("--batch_size", type=int, default=200, help="在寻找达标序列时，每一轮验证的候选序列数量。")
    parser.add_argument("--diversity_factor", type=float, default=1.2, help="为保证多样性，实际收集的序列数量会是目标数量乘以该因子，然后再聚类。")
    parser.add_argument("--min_length", type=int, default=2, help="生成序列最小长度。")
    parser.add_argument("--max_length", type=int, default=20, help="生成序列最大长度。")
    parser.add_argument("--temperature", type=float, default=2.5, help="采样温度。")
    parser.add_argument("--output_file", type=str, default="guaranteed_validated_sequences.fasta", help="最终序列保存的文件路径。")
    parser.add_argument("--generator_model", type=str, default=os.path.join("generator_checkpoints_v3.6", "final_generator_model.pth"), help="预训练的生成器模型路径。")
    parser.add_argument("--no_validation", action='store_true', help="不执行验证，仅生成并保存指定数量的序列（不推荐）。")

    args = parser.parse_args()
    
    # ### 修正部分 ###：修改了字典的键名，使其与load_predictor函数的参数名完全对应
    validator_params = {
        "model_path": "/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/no_over_confident/checkpoints/final_rl_model_logitp0.01_calibrated_FINETUNED_PROTT5.pth",
        "scaler_path": "/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/no_over_confident/checkpoints/scaler_FINETUNED_PROTT5.pkl",
        "feature_prott5_model_path": "/root/shared-nvme/chshhan/diffusion/prott5/model/",
        "feature_finetuned_prott5_file": None,
        # 以下为分类器架构参数，键名与函数定义一致
        "input_dim": 1914,
        "transformer_layers": 3,
        "transformer_heads": 4,
        "transformer_dropout": 0.1
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("--- 步骤一：加载所有模型 ---")
    generator = ProtT5Generator(vocab_size=VOCAB_SIZE, embed_dim=512, num_layers=6, num_heads=8, dropout=0.1)
    if not os.path.exists(args.generator_model):
        print(f"未找到生成器模型文件：{args.generator_model}"); return
    generator.load_state_dict(torch.load(args.generator_model, map_location=device))
    generator.to(device)
    print(f"生成器 '{os.path.basename(args.generator_model)}' 加载成功。")

    if not args.no_validation:
        try:
            scaler = joblib.load(validator_params["scaler_path"])
            feature_extractor = FeatureProtT5Model(validator_params["feature_prott5_model_path"], validator_params["feature_finetuned_prott5_file"])
            
            # ### 修正部分 ###：以更清晰、直接的方式调用load_predictor函数
            # 准备predictor架构参数
            predictor_arch_params = {
                'input_dim': validator_params['input_dim'],
                'transformer_layers': validator_params['transformer_layers'],
                'transformer_heads': validator_params['transformer_heads'],
                'transformer_dropout': validator_params['transformer_dropout']
            }
            # 调用函数
            predictor = load_predictor(
                model_path=validator_params["model_path"], 
                device=device,
                **predictor_arch_params
            )

        except Exception as e:
            print(f"加载验证模型失败，无法继续执行。错误: {e}"); return
    
    print("\n--- 步骤二：循环生成、验证并收集高质量序列 ---")
    
    validated_pool = {} 
    target_pool_size = int(args.target_count * args.diversity_factor)

    with tqdm(total=target_pool_size, desc="寻找高质量序列") as pbar:
        while len(validated_pool) < target_pool_size:
            candidate_seqs = generate_unique_sequences(
                generator, device, num_seq=args.batch_size, 
                min_length=args.min_length, max_length=args.max_length,
                temperature=args.temperature, batch_size=args.batch_size
            )
            candidate_seqs = [s for s in candidate_seqs if s not in validated_pool]
            if not candidate_seqs: continue

            probabilities = run_validation_batch(predictor, candidate_seqs, feature_extractor, scaler, device)
            
            for seq, prob in zip(candidate_seqs, probabilities):
                if prob is not None and prob > 0.90:
                    if seq not in validated_pool:
                        validated_pool[seq] = prob
                        pbar.update(1)
                        if len(validated_pool) >= target_pool_size:
                            break
    
    print(f"\n成功收集了 {len(validated_pool)} 条概率 > 90% 的序列。")

    print("\n--- 步骤三：对高质量序列进行最终聚类以保证多样性 ---")
    high_quality_sequences = list(validated_pool.keys())
    final_diverse_seqs = cluster_sequences(generator, high_quality_sequences, args.target_count, device)
    final_results_to_save = [(seq, validated_pool[seq]) for seq in final_diverse_seqs]
    final_results_to_save.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- 步骤四：保存最终结果 ---")
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    save_to_fasta_with_prob(final_results_to_save, args.output_file)
    
    print("\n--- ✨ 完成 ✨ ---")
    print(f"成功生成并验证了 {len(final_results_to_save)} 条序列（目标: {args.target_count}）。")
    print(f"所有序列的预测概率均 > 90%。")
    print(f"最终结果已保存至：{args.output_file}")

if __name__ == "__main__":
    main()
