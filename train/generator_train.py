#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成器模型训练代码（改进版 v3.6 - 强力鼓励更长序列及高预测值）
- 目标: 生成更长的序列，并促使较长序列也能获得高预测值
- 修改内容 (基于v3.5):
    - argparse 默认值调整:
        - --target_len_gaussian: 从 15.0 提高到 17.0 (更强力鼓励长序列)
        - --std_dev_len_gaussian: 从 5.0 提高到 5.5 (围绕新目标长度允许更多样性)
        - --reward_w_len: 从 0.50 提高到 0.60 (增加长度奖励的影响)
        - --min_gen_len: 从 3 提高到 4 (进一步避免过短序列)
        - --sampling_temperature: 从 1.5 降低到 1.3 (尝试更稳定的生成)
    - 更新checkpoint目录和输出文件名至 v3.6
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import Dataset, DataLoader
import argparse

import transformers
transformers.logging.set_verbosity_error()

try:
    from pretrain_finetune import finetune_protT5_on_antioxidant_proteins
    from antioxidant_predictor_5 import AntioxidantPredictor
    from feature_extract import ProtT5Model as FeatureProtT5Model, extract_features
except ImportError as e:
    print(f"导入自定义模块失败: {e}")
    print("请确保 pretrain_finetune.py, antioxidant_predictor_5.py, feature_extract.py 文件在PYTHONPATH中或当前目录。")
    exit()

AMINO_ACIDS_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
_token2id = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS_VOCAB)}
_token2id["<PAD>"] = 0
_token2id["<EOS>"] = 1
_id2token = {i: t for t, i in _token2id.items()}
GENERATOR_VOCAB_SIZE = len(_token2id)

class PeptideDataset(Dataset):
    def __init__(self, fasta_file=None, sequences=None, max_len=21, token_to_id_map=None):
        if sequences is not None:
            self.sequences = sequences
        elif fasta_file is not None:
            try:
                with open(fasta_file, "r") as f:
                    self.sequences = [line.strip() for line in f if line.strip() and not line.startswith(">")]
            except FileNotFoundError:
                print(f"错误: 监督数据FASTA文件 {fasta_file} 未找到。")
                self.sequences = []
        else:
            self.sequences = []
        self.max_len = max_len
        self.token2id = token_to_id_map if token_to_id_map else _token2id
        self.pad_token_id = self.token2id["<PAD>"]
        self.eos_token_id = self.token2id["<EOS>"]

    def encode(self, seq_str):
        ids = [self.token2id.get(aa.upper(), self.pad_token_id) for aa in seq_str]
        ids.append(self.eos_token_id)
        if len(ids) < self.max_len:
            ids.extend([self.pad_token_id] * (self.max_len - len(ids)))
        else:
            ids = ids[:self.max_len]
            if ids[-1] != self.eos_token_id :
                ids[-1] = self.eos_token_id
        return ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        token_ids = self.encode(seq_str)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, labels

def top_k_filtering(logits, top_k):
    if top_k <= 0: return logits
    values, _ = torch.topk(logits, top_k, dim=-1)
    kth_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < kth_values, torch.full_like(logits, -float('inf')), logits)

def top_p_filtering(logits, top_p, filter_value=-float('inf')):
    if top_p >= 1.0: return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    sorted_logits[sorted_indices_to_remove] = filter_value
    filtered_logits = torch.empty_like(logits)
    filtered_logits.scatter_(1, sorted_indices, sorted_logits)
    return filtered_logits

class ProtT5Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, num_heads=8, dropout=0.1, pad_token_id=0, eos_token_id=1):
        super(ProtT5Generator, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def forward(self, input_ids): 
        embeddings = self.embed_tokens(input_ids) 
        embeddings_transposed = embeddings.transpose(0, 1) 
        encoder_output = self.encoder(embeddings_transposed) 
        encoder_output_transposed = encoder_output.transpose(0, 1) 
        logits = self.lm_head(encoder_output_transposed) 
        return logits

    def sample(self, batch_size, max_length=20, device="cpu", temperature=1.0,
               min_peptide_len=4, # MODIFIED for v3.6
               repetition_penalty=1.2, top_k=0, top_p=0.9):
        start_tokens_pool = [i for i in range(2, self.vocab_size)] 
        if not start_tokens_pool: 
            start_tokens_pool = [self.pad_token_id if self.pad_token_id != self.eos_token_id else 2]

        start_token_indices = torch.randint(0, len(start_tokens_pool), (batch_size, 1), device=device)
        start_token = torch.tensor([start_tokens_pool[i.item()] for i in start_token_indices], device=device).unsqueeze(1)

        generated_ids = start_token 
        log_probs_list, entropy_list = [], []
        eos_generated = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_length -1): 
            logits = self.forward(generated_ids) 
            next_token_logits = logits[:, -1, :] / (temperature + 1e-8) 

            current_peptide_length = generated_ids.size(1) - 1 
            if current_peptide_length < min_peptide_len:
                next_token_logits[:, self.eos_token_id] = -float("inf") 

            if i > 0 : 
                next_token_logits[eos_generated, :] = -float("inf")
                next_token_logits[eos_generated, self.pad_token_id] = 0 

            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    if eos_generated[b]: continue 
                    for token_id_in_seq in generated_ids[b].tolist():
                        if token_id_in_seq == self.pad_token_id or token_id_in_seq == self.eos_token_id:
                            continue
                        if next_token_logits[b, token_id_in_seq] < 0: 
                            next_token_logits[b, token_id_in_seq] *= repetition_penalty
                        else: 
                            next_token_logits[b, token_id_in_seq] /= repetition_penalty
            
            next_token_logits = top_k_filtering(next_token_logits, top_k)
            next_token_logits = top_p_filtering(next_token_logits, top_p)

            probs = torch.softmax(next_token_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample() 
            next_token[eos_generated] = self.pad_token_id 

            log_prob = dist.log_prob(next_token) 
            entropy = dist.entropy() 
            log_probs_list.append(log_prob * (~eos_generated)) 
            entropy_list.append(entropy * (~eos_generated))   

            generated_ids = torch.cat((generated_ids, next_token.unsqueeze(1)), dim=1)
            
            eos_generated = eos_generated | (next_token == self.eos_token_id)
            if eos_generated.all(): break 
        
        total_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1) if log_probs_list else torch.zeros(batch_size, device=device)
        sum_entropy = torch.stack(entropy_list, dim=1).sum(dim=1) if entropy_list else torch.zeros(batch_size, device=device)
        
        actual_peptide_lengths = torch.zeros(batch_size, device=device, dtype=torch.float)
        for b in range(batch_size):
            seq_tokens = generated_ids[b, 1:].tolist() 
            try:
                eos_pos = seq_tokens.index(self.eos_token_id)
                actual_peptide_lengths[b] = eos_pos 
            except ValueError: 
                actual_peptide_lengths[b] = len(seq_tokens) 
        
        avg_entropy = sum_entropy / (actual_peptide_lengths + 1e-9) 
        return generated_ids, total_log_prob, avg_entropy

    def decode(self, token_ids_batch, id_to_token_map): 
        decoded_seqs = []
        for ids_tensor in token_ids_batch:
            seq_str = ""
            for token_id in ids_tensor.tolist()[1:]:
                if token_id == self.eos_token_id: break
                if token_id == self.pad_token_id: continue 
                seq_str += id_to_token_map.get(token_id, "?") 
            decoded_seqs.append(seq_str)
        return decoded_seqs

def load_classifier_model(model_path, input_dim, device, transformer_layers=3, transformer_heads=4, transformer_dropout=0.1):
    classifier = AntioxidantPredictor(input_dim=input_dim,
                                      transformer_layers=transformer_layers,
                                      transformer_heads=transformer_heads,
                                      transformer_dropout=transformer_dropout)
    try:
        state_dict = torch.load(model_path, map_location=device)
        classifier.load_state_dict(state_dict)
        print(f"二分类器模型 {model_path} 加载成功。")
        if hasattr(classifier, 'get_temperature') and callable(classifier.get_temperature):
            print(f"  其校准温度 T = {classifier.get_temperature():.4f}")
    except Exception as e: print(f"加载二分类器模型 {model_path} 失败: {e}"); raise
    classifier.to(device); classifier.eval()
    return classifier

def transform_rewards(prob_list, lower=0.2, upper=0.8, gamma=1.2):
    valid_probs = [p if p is not None and not np.isnan(p) else lower for p in prob_list] 
    prob_array = np.clip(np.array(valid_probs), lower, upper)
    return np.power((prob_array - lower) / (upper - lower + 1e-9), gamma).tolist()


def batch_predict_inference_generator(classifier_model, seq_list,
                                      feature_extractor_prott5_instance, scaler_instance, device,
                                      L_fixed_val=29, d_model_pe_val=16):
    if not seq_list: return []
    all_features, valid_indices = [], []
    for idx, seq_str in enumerate(seq_list):
        if not seq_str: continue 
        try:
            feat = extract_features(seq_str, feature_extractor_prott5_instance,
                                    L_fixed=L_fixed_val, d_model_pe=d_model_pe_val)
            all_features.append(feat); valid_indices.append(idx)
        except TypeError as te: print(f"TypeError: 特征提取失败 for '{seq_str[:20]}...': {te}.")
        except Exception as e: print(f"特征提取失败 for '{seq_str[:20]}...': {e}")
            
    if not all_features: return [float('nan')] * len(seq_list) 
    features_np = np.array(all_features, dtype=np.float32)
    try: features_scaled = scaler_instance.transform(features_np)
    except Exception as e: print(f"特征归一化失败: {e}"); return [float('nan')] * len(seq_list)
    
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    raw_probs_valid = []
    with torch.no_grad():
        scaled_logits = classifier_model(features_tensor)
        probs_tensor = torch.sigmoid(scaled_logits)
        if probs_tensor.ndim == 0: raw_probs_valid = [probs_tensor.item()] 
        elif probs_tensor.ndim >= 1 : raw_probs_valid = probs_tensor.squeeze().cpu().numpy().tolist()
        if not isinstance(raw_probs_valid, list): raw_probs_valid = [raw_probs_valid] 

    final_probs = [float('nan')] * len(seq_list) 
    for i, original_idx in enumerate(valid_indices):
        if i < len(raw_probs_valid): final_probs[original_idx] = raw_probs_valid[i]
    return final_probs

def get_bigram_diversity(seq_str):
    if len(seq_str) < 2: return 0.0
    bigrams = [seq_str[i:i+2] for i in range(len(seq_str)-1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0

def compute_reward_with_novelty(gen_seq_ids_batch, 
                                classifier_model, feature_prott5, scaler, device,
                                id_to_token_map, train_data_set,
                                min_peptide_len_reward=4, # MODIFIED for v3.6
                                max_peptide_len_reward_cap=20, 
                                target_len_gaussian=17.0, # MODIFIED for v3.6
                                std_dev_len_gaussian=5.5, # MODIFIED for v3.6
                                w_clf=0.5, w_novel_train=1.0, w_div=0.2, w_bigram_div=0.2,
                                w_len=0.1, w_internal_novel=0.15
                                ):
    eos_id_val = _token2id.get("<EOS>", 1); pad_id_val = _token2id.get("<PAD>", 0)
    decoded_seq_list, actual_lengths = [], []
    for ids_tensor in gen_seq_ids_batch:
        seq_str, current_len = "", 0
        for token_id in ids_tensor.tolist()[1:]:
            if token_id == eos_id_val: break
            if token_id == pad_id_val: continue
            seq_str += id_to_token_map.get(token_id, "?"); current_len +=1
        decoded_seq_list.append(seq_str); actual_lengths.append(current_len)

    classifier_probs = batch_predict_inference_generator(classifier_model, decoded_seq_list, feature_prott5, scaler, device)
    transformed_rewards_clf = transform_rewards(classifier_probs)
    
    final_rewards = []
    for idx, seq_str in enumerate(decoded_seq_list):
        length = actual_lengths[idx]
        
        if length < min_peptide_len_reward or np.isnan(transformed_rewards_clf[idx]):
            final_rewards.append(0.0); continue
        
        reward_clf_component = transformed_rewards_clf[idx]
        diversity_score = len(set(seq_str))/length if length>0 else 0
        bigram_diversity_score = get_bigram_diversity(seq_str)
        
        length_bonus_score = np.exp(-0.5 * ((length - target_len_gaussian) / std_dev_len_gaussian) ** 2)
        length_bonus_score = np.clip(length_bonus_score, 0.0, 1.0) 

        rep_rate = (length - len(set(seq_str)))/length if length>0 else 0
        novel_internal_factor = np.exp(-3 * rep_rate) 
        novel_vs_train_factor = 0.0 if seq_str in train_data_set else 1.0 
        
        final_r = (w_clf * reward_clf_component * novel_vs_train_factor + 
                   w_div * diversity_score +
                   w_bigram_div * bigram_diversity_score +
                   w_len * length_bonus_score + 
                   w_internal_novel * novel_internal_factor)
        final_r = np.clip(final_r, 0, 2.5) 
        final_rewards.append(final_r)
        
    return torch.tensor(final_rewards, dtype=torch.float32, device=device)


def compute_generation_stats(decoded_seqs, predicted_probs_raw):
    if not decoded_seqs: return {"average_length":0, "average_diversity":0, "unique_ratio":0, "average_bigram_diversity":0, "predicted_prob_mean":0, "predicted_prob_std":0}
    lengths = [len(s) for s in decoded_seqs]; diversity = [len(set(s))/len(s) if len(s)>0 else 0 for s in decoded_seqs]
    unique_ratio = len(set(decoded_seqs))/len(decoded_seqs) if decoded_seqs else 0
    bigram_divs = [get_bigram_diversity(s) for s in decoded_seqs]
    valid_probs = [p for p in predicted_probs_raw if p is not None and not np.isnan(p)]
    return {"average_length":np.mean(lengths) if lengths else 0, "average_diversity":np.mean(diversity) if diversity else 0,
            "unique_ratio":unique_ratio, "average_bigram_diversity":np.mean(bigram_divs) if bigram_divs else 0,
            "predicted_prob_mean":np.mean(valid_probs) if valid_probs else 0, "predicted_prob_std":np.std(valid_probs) if valid_probs else 0}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"使用设备: {device}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    finetuned_prott5_for_generator_path = os.path.join(args.checkpoint_dir, "finetuned_prott5_for_generator.bin")
    if args.run_prott5_finetune or not os.path.exists(finetuned_prott5_for_generator_path):
        print("微调 ProtT5 (ProtT5ForFineTuning)...")
        if not os.path.exists(args.protein_csv_for_finetune): print(f"错误: ProtT5微调CSV {args.protein_csv_for_finetune} 不存在。"); exit(1)
        finetune_params = {
            "max_length": args.prott5_finetune_max_length, "mask_prob": 0.15, "batch_size": args.prott5_finetune_batch_size,
            "embed_dim": args.generator_embed_dim, "num_layers": args.generator_num_layers, "num_heads": args.generator_num_heads, 
            "dropout": args.generator_dropout, "learning_rate": args.prott5_finetune_lr, "num_epochs": args.prott5_finetune_epochs, 
            "device": device, "num_workers": args.num_workers
        }
        finetune_protT5_on_antioxidant_proteins(args.protein_csv_for_finetune, finetune_params, model_path=args.checkpoint_dir)
        default_finetune_save_name = os.path.join(args.checkpoint_dir, "finetuned_prott5.bin") 
        if os.path.exists(default_finetune_save_name): os.rename(default_finetune_save_name, finetuned_prott5_for_generator_path)

    print("初始化 ProtT5Generator..."); generator = ProtT5Generator(GENERATOR_VOCAB_SIZE, args.generator_embed_dim, args.generator_num_layers, args.generator_num_heads, args.generator_dropout, _token2id["<PAD>"], _token2id["<EOS>"])
    if os.path.exists(finetuned_prott5_for_generator_path):
        source_state_dict = torch.load(finetuned_prott5_for_generator_path, map_location=device); generator_state_dict = generator.state_dict()
        new_state_dict, loaded_count = {}, 0
        for k_s, v_s in source_state_dict.items():
            k_t = None
            if k_s.startswith("transformer_encoder."): k_t = "encoder." + k_s[len("transformer_encoder."):]
            if k_t and k_t in generator_state_dict and generator_state_dict[k_t].shape == v_s.shape: new_state_dict[k_t] = v_s; loaded_count +=1
        if new_state_dict: 
            missing, unexpected = generator.load_state_dict(new_state_dict, strict=False)
            print(f"  生成器加载 {loaded_count} encoder参数: 缺失 {len(missing)}, 意外 {len(unexpected)}")
    else: print(f"警告: 微调源模型 {finetuned_prott5_for_generator_path} 未找到。生成器将从随机权重开始。")
    generator.to(device); gen_optimizer = optim.Adam(generator.parameters(), lr=args.generator_lr)

    print(f"加载监督数据: {args.train_peptides_fasta}")
    sup_dataset = PeptideDataset(fasta_file=args.train_peptides_fasta, max_len=args.max_seq_len + 1, token_to_id_map=_token2id) 
    if len(sup_dataset) == 0: print("错误：监督数据集为空！"); exit(1)
    sup_dataloader = DataLoader(sup_dataset, batch_size=args.sup_batch_size, shuffle=True, num_workers=args.num_workers)
    train_sequences_for_novelty = set(sup_dataset.sequences) 

    print(f"加载分类器: {args.classifier_model_path}")
    classifier = load_classifier_model(args.classifier_model_path, args.classifier_input_dim, device, args.classifier_transformer_layers, args.classifier_transformer_heads, args.classifier_transformer_dropout)
    print(f"加载Scaler: {args.scaler_path}"); scaler_instance = joblib.load(args.scaler_path)
    print(f"初始化特征提取ProtT5: {args.feature_prott5_model_path}")
    feature_prott5_instance = FeatureProtT5Model(args.feature_prott5_model_path, args.feature_finetuned_prott5_file)

    baseline_ma_reward = torch.tensor(0.0, device=device) 
    epoch_performance_log = []
    print(f"开始训练 {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        print(f"\n[Epoch {epoch+1}/{args.num_epochs}]"); generator.train()       
        current_epoch_rl_losses = []
        avg_batch_rewards_rl_epoch = [] 
        if args.num_rl_steps_per_epoch > 0:
            print("  RL 更新..."); 
            for step in range(args.num_rl_steps_per_epoch):
                gen_optimizer.zero_grad()
                gen_seq_ids, log_probs, entropies = generator.sample(
                    args.rl_batch_size, args.max_seq_len, device, 
                    args.sampling_temperature, args.min_gen_len, 
                    args.repetition_penalty, args.top_k, args.top_p
                )
                rewards = compute_reward_with_novelty(
                    gen_seq_ids, classifier, feature_prott5_instance, scaler_instance, 
                    device, _id2token, train_sequences_for_novelty, 
                    min_peptide_len_reward=args.min_gen_len, 
                    max_peptide_len_reward_cap=args.max_seq_len, 
                    target_len_gaussian=args.target_len_gaussian, 
                    std_dev_len_gaussian=args.std_dev_len_gaussian, 
                    w_clf=args.reward_w_clf, w_novel_train=args.reward_w_novel_train, 
                    w_div=args.reward_w_div, w_bigram_div=args.reward_w_bigram_div, 
                    w_len=args.reward_w_len, w_internal_novel=args.reward_w_internal_novel
                )
                
                current_reward_mean = rewards.mean().detach(); baseline_ma_reward = baseline_ma_reward * 0.9 + 0.1 * current_reward_mean; avg_batch_rewards_rl_epoch.append(current_reward_mean.item())
                rl_loss = - (log_probs * (rewards - baseline_ma_reward)).mean() - args.entropy_coef * entropies.mean()
                rl_loss.backward(); torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip); gen_optimizer.step()
                current_epoch_rl_losses.append(rl_loss.item())
        avg_rl_loss_epoch = np.mean(current_epoch_rl_losses) if current_epoch_rl_losses else 0
        avg_epoch_reward_value = np.mean(avg_batch_rewards_rl_epoch) if avg_batch_rewards_rl_epoch else 0.0
        if args.num_rl_steps_per_epoch > 0: print(f"    RL阶段平均奖励: {avg_epoch_reward_value:.4f}")

        current_epoch_sup_losses = []
        if len(sup_dataloader) > 0 and args.sup_train_freq > 0 and (epoch + 1) % args.sup_train_freq == 0 : 
            print("  监督微调...");
            for batch_idx, batch in enumerate(sup_dataloader):
                gen_optimizer.zero_grad(); input_ids, labels = batch[0].to(device), batch[1].to(device)
                logits = generator(input_ids); loss_fn_sup = nn.CrossEntropyLoss(ignore_index=_token2id["<PAD>"])
                sup_loss = loss_fn_sup(logits.view(-1, GENERATOR_VOCAB_SIZE), labels.view(-1))
                sup_loss.backward(); torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip); gen_optimizer.step()
                current_epoch_sup_losses.append(sup_loss.item())
        avg_sup_loss_epoch = np.mean(current_epoch_sup_losses) if current_epoch_sup_losses else 0
        
        print(f"  Epoch {epoch+1} 完成: Avg RL Loss={avg_rl_loss_epoch:.4f}, Avg Sup Loss={avg_sup_loss_epoch:.4f}")
        epoch_performance_log.append({"epoch": epoch+1, "rl_loss": avg_rl_loss_epoch, "sup_loss": avg_sup_loss_epoch, "avg_reward": avg_epoch_reward_value})

        generator.eval() 
        with torch.no_grad():
            eval_gen_ids, _, _ = generator.sample(args.eval_batch_size, args.max_seq_len, device, args.sampling_temperature, args.min_gen_len, args.repetition_penalty, args.top_k, args.top_p)
        decoded_eval_seqs = generator.decode(eval_gen_ids, _id2token)
        eval_classifier_probs = batch_predict_inference_generator(classifier, decoded_eval_seqs, feature_prott5_instance, scaler_instance, device)
        eval_stats = compute_generation_stats(decoded_eval_seqs, eval_classifier_probs)
        print(f"  [评估 Epoch {epoch+1}] AvgLen={eval_stats['average_length']:.2f}, AvgDiv={eval_stats['average_diversity']:.2f}, Uniq%={eval_stats['unique_ratio']*100:.1f}, AvgBigramDiv={eval_stats['average_bigram_diversity']:.2f}, AvgProb={eval_stats['predicted_prob_mean']:.3f}")
        
        if (epoch + 1) % args.save_every_epochs == 0:
            torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f"generator_model_epoch_{epoch+1}.pth"))
    
    print("\n--- 训练性能总结 ---"); [print(f"Epoch {log['epoch']}: RL Loss={log['rl_loss']:.4f}, Sup Loss={log['sup_loss']:.4f}, Avg Reward={log['avg_reward']:.4f}") for log in epoch_performance_log]
    final_gen_model_path = os.path.join(args.checkpoint_dir, "final_generator_model.pth")
    torch.save(generator.state_dict(), final_gen_model_path); print(f"\n生成器最终参数保存至: {final_gen_model_path}")
    print("\n--- 开始最终生成和评估 ---"); generator.eval()
    with torch.no_grad():
        final_sample_ids,_,_ = generator.sample(args.num_final_samples,args.max_seq_len,device,args.sampling_temperature,args.min_gen_len,args.repetition_penalty,args.top_k,args.top_p)
    final_decoded_seqs = generator.decode(final_sample_ids, _id2token)
    final_clf_probs = batch_predict_inference_generator(classifier,final_decoded_seqs,feature_prott5_instance,scaler_instance,device)
    results_df = pd.DataFrame({'sequence':final_decoded_seqs, 'predicted_probability':final_clf_probs}).dropna(subset=['predicted_probability']).sort_values(by='predicted_probability',ascending=False)
    output_csv_path = os.path.join(args.checkpoint_dir, "generated_peptides_ranked_v3.6.csv") # New filename for v3.6
    results_df.to_csv(output_csv_path, index=False); print(f"\n最终生成 {len(results_df)} 条有效序列保存至: {output_csv_path}")
    print("\nTop 10 生成序列:"); [print(f"  {results_df.iloc[i]['sequence']} (Prob: {results_df.iloc[i]['predicted_probability']:.4f})") for i in range(min(10,len(results_df)))]
    final_stats = compute_generation_stats(results_df['sequence'].tolist(), results_df['predicted_probability'].tolist())
    print("\n最终生成序列评价指标："); [print(f"  {k.replace('_',' ').capitalize()}: {v:.4f}") for k,v in final_stats.items()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="肽序列生成模型训练脚本 v3.6 (强力鼓励更长序列)")
    # File Paths
    parser.add_argument("--protein_csv_for_finetune", type=str, default="/root/shared-nvme/chshhan/diffusion/2_5rl_env/AODB_PROTEIN.csv")
    parser.add_argument("--train_peptides_fasta", type=str, default="/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/clean_data/clean_positive.fasta")
    parser.add_argument("--classifier_model_path", type=str, default="checkpoints/final_rl_model_logitp0.01_calibrated.pth") 
    parser.add_argument("--scaler_path", type=str, default="checkpoints/scaler.pkl") 
    parser.add_argument("--feature_prott5_model_path", type=str, default="/root/shared-nvme/chshhan/diffusion/prott5/model/")
    parser.add_argument("--feature_finetuned_prott5_file", type=str, default=None, help="Path to finetuned ProtT5 weights if feature_extract.ProtT5Model needs it.")
    parser.add_argument("--checkpoint_dir", type=str, default="generator_checkpoints_v3.6") # **NEW DIR for v3.6**
    
    # ProtT5 Fine-tuning for Generator Backbone
    parser.add_argument("--run_prott5_finetune", action='store_true', help="Run ProtT5 fine-tuning for the generator backbone.")
    parser.add_argument("--prott5_finetune_max_length", type=int, default=256)
    parser.add_argument("--prott5_finetune_batch_size", type=int, default=8)
    parser.add_argument("--prott5_finetune_lr", type=float, default=3e-5) 
    parser.add_argument("--prott5_finetune_epochs", type=int, default=3) 

    # Generator Architecture
    parser.add_argument("--generator_embed_dim", type=int, default=512)
    parser.add_argument("--generator_num_layers", type=int, default=6)
    parser.add_argument("--generator_num_heads", type=int, default=8)
    parser.add_argument("--generator_dropout", type=float, default=0.1)
    
    # Training Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=35)
    parser.add_argument("--num_rl_steps_per_epoch", type=int, default=5) 
    parser.add_argument("--generator_lr", type=float, default=3e-5) 
    parser.add_argument("--sup_batch_size", type=int, default=32)
    parser.add_argument("--sup_train_freq", type=int, default=1, help="Frequency of supervised training epochs.")
    parser.add_argument("--rl_batch_size", type=int, default=32)
    parser.add_argument("--entropy_coef", type=float, default=0.025) 
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every_epochs", type=int, default=5)

    # Sampling Hyperparameters
    parser.add_argument("--sampling_temperature", type=float, default=1.3, help="Sampling temperature. (v3.5 was 1.5)") # **MODIFIED**
    parser.add_argument("--max_seq_len", type=int, default=20, help="生成序列的最大长度 (不含起始token, 但包含EOS)")
    parser.add_argument("--min_gen_len", type=int, default=2, help="生成肽序列的最小实际长度 (不含起始token). (v3.5 was 3)") # **MODIFIED**
    parser.add_argument("--repetition_penalty", type=float, default=1.6) 
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering. 0 to disable.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus (top-p) filtering. 1.0 to disable.")

    # Reward Function Parameters & Weights
    parser.add_argument("--target_len_gaussian", type=float, default=10.0, help="Target mean length for Gaussian reward. (v3.5 was 15.0)") # **MODIFIED**
    parser.add_argument("--std_dev_len_gaussian", type=float, default=5.5, help="Standard deviation for Gaussian length reward. (v3.5 was 5.0)") # **MODIFIED**
    parser.add_argument("--reward_w_clf", type=float, default=0.40, help="分类器奖励权重")
    parser.add_argument("--reward_w_novel_train", type=float, default=1.0, help="与训练集比较的新颖性权重 (0 if seen, 1 if novel)")
    parser.add_argument("--reward_w_div", type=float, default=0.35, help="字符级多样性权重") 
    parser.add_argument("--reward_w_bigram_div", type=float, default=0.35, help="Bigram多样性权重") 
    parser.add_argument("--reward_w_len", type=float, default=0.60, help="长度奖励权重 (v3.5 was 0.50)") # **MODIFIED**
    parser.add_argument("--reward_w_internal_novel", type=float, default=0.30, help="内部重复惩罚因子权重")

    # Classifier and Feature Extractor Details
    parser.add_argument("--classifier_input_dim", type=int, default=1914, help="Input dimension for the AntioxidantPredictor.")
    parser.add_argument("--classifier_transformer_layers", type=int, default=3)
    parser.add_argument("--classifier_transformer_heads", type=int, default=4)
    parser.add_argument("--classifier_transformer_dropout", type=float, default=0.1)
    
    # Evaluation & Misc
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_final_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")

    cli_args = parser.parse_args()
    main(cli_args)
