# lora_generator_train_final.py
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
from tqdm import tqdm
import transformers
from transformers import T5EncoderModel, T5Tokenizer
from peft import PeftModel

# 确保所有自定义模块都可以被导入
try:
    from feature_extract import ProtT5Model as FeatureProtT5Model, extract_features
    from antioxidant_predictor_5 import AntioxidantPredictor
except ImportError as e:
    print(f"导入自定义模块失败: {e}")
    print("请确保 feature_extract.py, antioxidant_predictor_5.py 文件在PYTHONPATH中或当前目录。")
    exit()

# --- 词汇表定义 ---
AMINO_ACIDS_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
_token2id = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS_VOCAB)}
_token2id["<PAD>"] = 0
_token2id["<EOS>"] = 1
_id2token = {i: t for t, i in _token2id.items()}
GENERATOR_VOCAB_SIZE = len(_token2id)


# ==============================================================================
# --- 1. 核心模型与辅助函数定义 ---
# ==============================================================================
class AdvancedProtT5Generator(nn.Module):
    """
    使用LoRA增强的ProtT5作为骨干网络的生成器模型。
    """
    def __init__(self, base_model_path, lora_adapter_path, vocab_size):
        super(AdvancedProtT5Generator, self).__init__()
        
        print(f"  - 正在加载基础ProtT5模型: {base_model_path}")
        base_model = T5EncoderModel.from_pretrained(base_model_path)

        print(f"  - 正在加载并应用LoRA适配器: {lora_adapter_path}")
        self.backbone = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        embed_dim = self.backbone.config.d_model
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.eos_token_id = _token2id["<EOS>"]
        self.pad_token_id = _token2id["<PAD>"]
        print("  - 高级生成器初始化完成。")

    def forward(self, input_ids):
        attention_mask = (input_ids != self.pad_token_id).int()
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits

    def sample(self, batch_size, max_length=20, device="cpu", temperature=1.0,
               min_peptide_len=4, repetition_penalty=1.2, top_k=0, top_p=0.9):
        start_token = torch.randint(2, self.vocab_size, (batch_size, 1), device=device)
        generated_ids = start_token
        log_probs_list, entropy_list = [], []
        eos_generated = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_length - 1):
            logits = self.forward(generated_ids)
            next_token_logits = logits[:, -1, :] / (temperature + 1e-8)
            
            if generated_ids.size(1) < min_peptide_len:
                next_token_logits[:, self.eos_token_id] = -float("inf")

            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id_in_seq in generated_ids[b]:
                        if next_token_logits[b, token_id_in_seq] < 0:
                            next_token_logits[b, token_id_in_seq] *= repetition_penalty
                        else:
                            next_token_logits[b, token_id_in_seq] /= repetition_penalty
            
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            
            log_prob = dist.log_prob(next_token)
            entropy = dist.entropy()
            log_probs_list.append(log_prob * (~eos_generated))
            entropy_list.append(entropy * (~eos_generated))
            
            generated_ids = torch.cat((generated_ids, next_token.unsqueeze(1)), dim=1)
            eos_generated = eos_generated | (next_token == self.eos_token_id)
            if eos_generated.all(): break
        
        total_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1) if log_probs_list else torch.zeros(batch_size, device=device)
        avg_entropy = torch.stack(entropy_list, dim=1).mean(dim=1) if entropy_list else torch.zeros(batch_size, device=device)
        return generated_ids, total_log_prob, avg_entropy

    def decode(self, token_ids_batch):
        decoded_seqs = []
        for ids_tensor in token_ids_batch:
            seq_str = ""
            for token_id in ids_tensor.tolist()[1:]:
                if token_id == self.eos_token_id: break
                if token_id == self.pad_token_id: continue
                seq_str += _id2token.get(token_id, "?")
            decoded_seqs.append(seq_str)
        return decoded_seqs

class PeptideDataset(Dataset):
    def __init__(self, fasta_file, max_len=21):
        try:
            with open(fasta_file, "r") as f: self.sequences = [line.strip() for line in f if line.strip() and not line.startswith(">")]
        except FileNotFoundError: self.sequences = []
        self.max_len = max_len; self.token2id = _token2id
        self.pad_token_id = self.token2id["<PAD>"]; self.eos_token_id = self.token2id["<EOS>"]

    def encode(self, seq_str):
        ids = [self.token2id.get(aa.upper(), self.pad_token_id) for aa in seq_str]
        ids.append(self.eos_token_id)
        if len(ids) < self.max_len: ids.extend([self.pad_token_id] * (self.max_len - len(ids)))
        else:
            ids = ids[:self.max_len]
            if ids[-1] != self.eos_token_id: ids[-1] = self.eos_token_id
        return ids

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        seq_str = self.sequences[idx]; token_ids = self.encode(seq_str)
        return torch.tensor(token_ids[:-1], dtype=torch.long), torch.tensor(token_ids[1:], dtype=torch.long)

def load_classifier_model(model_path, input_dim, device):
    classifier = AntioxidantPredictor(input_dim)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    return classifier.to(device).eval()

def batch_predict(classifier_model, seq_list, feature_extractor, scaler, device):
    if not seq_list: return []
    try:
        features = np.array([extract_features(s, feature_extractor) for s in seq_list])
        scaled_features = scaler.transform(features)
        tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = classifier_model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy().tolist()
        return [probs] if not isinstance(probs, list) else probs
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        return [None] * len(seq_list)

def get_bigram_diversity(seq_str):
    if len(seq_str) < 2: return 0.0
    bigrams = {seq_str[i:i+2] for i in range(len(seq_str)-1)}
    return len(bigrams) / (len(seq_str) - 1)

def compute_reward(decoded_seqs, probs, train_seq_set, args):
    rewards = []
    for seq, prob in zip(decoded_seqs, probs):
        if not seq or prob is None or np.isnan(prob):
            rewards.append(0.0)
            continue
        length = len(seq)
        reward_clf = (prob - 0.5) * 2 # Scale to [-1, 1]
        len_reward = np.exp(-0.5 * ((length - args.target_len_gaussian) / args.std_dev_len_gaussian) ** 2)
        novelty_reward = 1.0 if seq not in train_seq_set else 0.0
        char_div = len(set(seq)) / length if length > 0 else 0.0
        bigram_div = get_bigram_diversity(seq)
        total_reward = (args.reward_w_clf * reward_clf +
                        args.reward_w_len * len_reward +
                        args.reward_w_novel_train * novelty_reward +
                        args.reward_w_div * char_div +
                        args.reward_w_bigram_div * bigram_div)
        rewards.append(total_reward)
    return torch.tensor(rewards, dtype=torch.float32, device=args.device)

def compute_generation_stats(decoded_seqs, predicted_probs_raw):
    if not decoded_seqs: return {"average_length":0, "average_diversity":0, "unique_ratio":0, "average_bigram_diversity":0, "predicted_prob_mean":0, "predicted_prob_std":0}
    lengths = [len(s) for s in decoded_seqs]; diversity = [len(set(s))/len(s) if len(s)>0 else 0 for s in decoded_seqs]
    unique_ratio = len(set(decoded_seqs))/len(decoded_seqs) if decoded_seqs else 0
    bigram_divs = [get_bigram_diversity(s) for s in decoded_seqs]
    valid_probs = [p for p in predicted_probs_raw if p is not None and not np.isnan(p)]
    return {"average_length":np.mean(lengths) if lengths else 0, "average_diversity":np.mean(diversity) if diversity else 0,
            "unique_ratio":unique_ratio, "average_bigram_diversity":np.mean(bigram_divs) if bigram_divs else 0,
            "predicted_prob_mean":np.mean(valid_probs) if valid_probs else 0, "predicted_prob_std":np.std(valid_probs) if valid_probs else 0}

class LoRAFeatureExtractorWrapper:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
    def encode(self, seq):
        sequence_spaced = " ".join(list(seq))
        encoded_input = self.tokenizer(sequence_spaced, return_tensors='pt', padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            embedding = self.model(**encoded_input).last_hidden_state
        return embedding.squeeze(0).cpu().numpy()

# ==============================================================================
# --- 主执行流程 ---
# ==============================================================================
def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {args.device}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== 步骤 1: 初始化带LoRA的生成器模型 ===")
    generator = AdvancedProtT5Generator(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        vocab_size=GENERATOR_VOCAB_SIZE
    ).to(args.device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.generator_lr)
    
    print("\n=== 步骤 2: 加载用于奖励计算的验证器 ===")
    validator = load_classifier_model(args.classifier_model_path, args.classifier_input_dim, args.device)
    scaler = joblib.load(args.scaler_path)
    
    feature_extractor_wrapper = LoRAFeatureExtractorWrapper(generator.backbone, T5Tokenizer.from_pretrained(args.base_model_path), args.device)

    print(f"\n=== 步骤 3: 强化学习与监督学习交替训练 ===")
    sup_dataset = PeptideDataset(fasta_file=args.train_peptides_fasta, max_len=args.max_seq_len + 1)
    sup_dataloader = DataLoader(sup_dataset, batch_size=args.sup_batch_size, shuffle=True)
    train_sequences_for_novelty = set(sup_dataset.sequences)

    baseline_reward = torch.tensor(0.0, device=args.device)
    print(f"开始训练 {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        print(f"\n[Epoch {epoch+1}/{args.num_epochs}]")
        generator.train()
        
        # --- RL Step ("Yang") ---
        for _ in tqdm(range(args.num_rl_steps_per_epoch), desc=f"Epoch {epoch+1} RL"):
            optimizer.zero_grad()
            gen_ids, log_probs, entropies = generator.sample(args.rl_batch_size, args.max_seq_len, args.device, args.sampling_temperature, args.min_gen_len, args.repetition_penalty, args.top_k, args.top_p)
            decoded_seqs = generator.decode(gen_ids)
            probs = batch_predict(validator, decoded_seqs, feature_extractor_wrapper, scaler, args.device)
            rewards = compute_reward(decoded_seqs, probs, train_sequences_for_novelty, args)
            
            baseline_reward = 0.9 * baseline_reward + 0.1 * rewards.mean()
            rl_loss = - (log_probs * (rewards - baseline_reward)).mean() - args.entropy_coef * entropies.mean()
            rl_loss.backward()
            optimizer.step()

        # --- SL Step ("Yin") ---
        if sup_dataloader and (epoch + 1) % args.sup_train_freq == 0:
            for input_ids, labels in tqdm(sup_dataloader, desc=f"Epoch {epoch+1} SL"):
                optimizer.zero_grad()
                logits = generator(input_ids.to(args.device))
                loss_fn = nn.CrossEntropyLoss(ignore_index=_token2id["<PAD>"])
                loss = loss_fn(logits.view(-1, GENERATOR_VOCAB_SIZE), labels.view(-1).to(args.device))
                loss.backward()
                optimizer.step()

        # --- Per-Epoch Evaluation ---
        generator.eval()
        with torch.no_grad():
            eval_ids, _, _ = generator.sample(args.eval_batch_size, args.max_seq_len, args.device)
        eval_decoded = generator.decode(eval_ids)
        eval_probs = batch_predict(validator, eval_decoded, feature_extractor_wrapper, scaler, args.device)
        eval_stats = compute_generation_stats(eval_decoded, eval_probs)
        print(f"  [评估 Epoch {epoch+1}] AvgLen={eval_stats['average_length']:.2f}, AvgProb={eval_stats['predicted_prob_mean']:.3f}, Uniq%={eval_stats['unique_ratio']*100:.1f}")
        
        if (epoch + 1) % args.save_every_epochs == 0:
            output_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            generator.backbone.save_pretrained(output_path)
            torch.save(generator.lm_head.state_dict(), os.path.join(output_path, "lm_head.pth"))
            print(f"  - 已保存 checkpoint 至 {output_path}")

    print("\n=== 步骤 4: 保存并评估最终模型 ===")
    final_output_path = os.path.join(args.output_dir, "final_lora_generator")
    generator.backbone.save_pretrained(final_output_path)
    torch.save(generator.lm_head.state_dict(), os.path.join(final_output_path, "lm_head.pth"))
    print(f"最终模型（适配器+LM头）已保存至: {final_output_path}")
    
    # Final Large Scale Generation and Evaluation
    print("\n--- 开始最终生成和评估 ---")
    generator.eval()
    with torch.no_grad():
        final_sample_ids,_,_ = generator.sample(args.num_final_samples, args.max_seq_len, args.device)
    final_decoded_seqs = generator.decode(final_sample_ids)
    final_clf_probs = batch_predict(validator, final_decoded_seqs, feature_extractor_wrapper, scaler, args.device)
    
    results_df = pd.DataFrame({'sequence':final_decoded_seqs, 'predicted_probability':final_clf_probs}).dropna().sort_values('predicted_probability', ascending=False)
    output_csv_path = os.path.join(args.output_dir, "generated_peptides_lora_ranked.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"\n最终生成 {len(results_df)} 条有效序列保存至: {output_csv_path}")

    final_stats = compute_generation_stats(results_df['sequence'].tolist(), results_df['predicted_probability'].tolist())
    print("\n最终生成序列评价指标：")
    for k,v in final_stats.items():
        print(f"  - {k.replace('_',' ').capitalize()}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LoRA微调后的ProtT5作为骨干进行序列生成训练。")
    
    # --- 路径参数 ---
    parser.add_argument("--base_model_path", type=str, default="./prott5/model/")
    parser.add_argument("--lora_adapter_path", type=str, default="./lora_finetuned_prott5")
    parser.add_argument("--classifier_model_path", type=str, default="./predictor_with_lora_checkpoints/final_predictor_with_lora.pth")
    parser.add_argument("--scaler_path", type=str, default="./predictor_with_lora_checkpoints/scaler_lora.pkl")
    parser.add_argument("--train_peptides_fasta", type=str, default="./data/remaining_positive.fasta")
    parser.add_argument("--output_dir", type=str, default="./generator_with_lora_output")
    
    # --- 训练和采样参数 ---
    parser.add_argument("--generator_lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--num_rl_steps_per_epoch", type=int, default=10)
    parser.add_argument("--sup_batch_size", type=int, default=32)
    parser.add_argument("--sup_train_freq", type=int, default=10)
    parser.add_argument("--rl_batch_size", type=int, default=32)
    parser.add_argument("--entropy_coef", type=float, default=0.025)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every_epochs", type=int, default=5)
    parser.add_argument("--sampling_temperature", type=float, default=1.3)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_gen_len", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--reward_w_clf", type=float, default=0.60)
    parser.add_argument("--reward_w_novel_train", type=float, default=1.0)
    parser.add_argument("--reward_w_div", type=float, default=0.35)
    parser.add_argument("--reward_w_bigram_div", type=float, default=0.35)
    parser.add_argument("--reward_w_len", type=float, default=0.60)
    parser.add_argument("--target_len_gaussian", type=float, default=17.0)
    parser.add_argument("--std_dev_len_gaussian", type=float, default=5.5)
    parser.add_argument("--classifier_input_dim", type=int, default=1914)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_final_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
