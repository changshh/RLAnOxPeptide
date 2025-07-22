# use_lora_generator.py (FINAL CORRECTED)
import os
import argparse
import torch
import numpy as np
import joblib
from sklearn.cluster import KMeans
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from peft import PeftModel

# 确保所有自定义模块都可以被导入
try:
    from antioxidant_predictor_5 import AntioxidantPredictor
    from feature_extract import extract_features
except ImportError as e:
    print(f"导入模块失败: {e}")
    exit()

# --- 词汇表定义 ---
AMINO_ACIDS_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
_token2id = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS_VOCAB)}
_token2id["<PAD>"] = 0
_token2id["<EOS>"] = 1
_id2token = {i: t for t, i in _token2id.items()}
VOCAB_SIZE = len(_token2id)

# =====================================================================================
# --- 核心模型与辅助函数定义 ---
# =====================================================================================
class AdvancedProtT5Generator(nn.Module):
    def __init__(self, base_model_path, lora_adapter_path, vocab_size):
        super(AdvancedProtT5Generator, self).__init__()
        
        print(f"  - 正在加载基础ProtT5模型: {base_model_path}")
        base_model = T5EncoderModel.from_pretrained(base_model_path)

        print(f"  - 正在加载并应用LoRA适配器: {lora_adapter_path}")
        self.backbone = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        # !! 关键修正点 1 !!
        # 将 embed_tokens 层暴露出来，方便后续聚类使用
        self.embed_tokens = self.backbone.get_input_embeddings()
        
        embed_dim = self.backbone.config.d_model
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.eos_token_id = _token2id["<EOS>"]
        self.pad_token_id = _token2id["<PAD>"]
        print("  - 高级生成器框架初始化完成。")

    def forward(self, input_ids):
        # 注意：forward现在直接使用backbone，它已经是PeftModel
        attention_mask = (input_ids != self.pad_token_id).int()
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits

    def sample(self, batch_size, max_length=20, device="cpu", temperature=2.5, min_decoded_length=3):
        start_token = torch.randint(2, self.vocab_size, (batch_size, 1), device=device)
        generated = start_token
        for _ in range(max_length - 1):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / temperature
            if generated.size(1) < min_decoded_length:
                next_logits[:, self.eos_token_id] = -float("inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if (generated == self.eos_token_id).any(dim=1).all():
                break
        return generated

    def decode(self, token_ids_batch):
        seqs = []
        for ids_tensor in token_ids_batch:
            seq = ""
            for token_id in ids_tensor.tolist()[1:]:
                if token_id == self.eos_token_id: break
                if token_id == _token2id["<PAD>"]: continue
                seq += _id2token.get(token_id, "?")
            seqs.append(seq)
        return seqs

class LoRAProtT5Extractor:
    def __init__(self, base_model_path, lora_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = T5EncoderModel.from_pretrained(base_model_path)
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        self.model = lora_model.merge_and_unload().to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.model.eval()
    def encode(self, sequence):
        sequence_spaced = " ".join(list(sequence))
        encoded_input = self.tokenizer(sequence_spaced, return_tensors='pt', padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            embedding = self.model(**encoded_input).last_hidden_state
        return embedding.squeeze(0).cpu().numpy()

def generate_unique_sequences(generator, device, num_seq, min_length, max_length, temperature, batch_size):
    generator.eval()
    unique_seqs = set()
    with tqdm(total=num_seq, desc="Generating unique sequences") as pbar:
        while len(unique_seqs) < num_seq:
            needed = num_seq - len(unique_seqs)
            current_batch_size = min(batch_size, needed * 2 if needed > 1 else 2)
            generated_tokens = generator.sample(
                batch_size=current_batch_size, max_length=max_length, device=device,
                temperature=temperature, min_decoded_length=min_length
            )
            decoded = generator.decode(generated_tokens.cpu())
            initial_count = len(unique_seqs)
            for seq in decoded:
                if min_length <= len(seq) <= max_length:
                    unique_seqs.add(seq)
                    if len(unique_seqs) >= num_seq: break
            pbar.update(len(unique_seqs) - initial_count)
    return list(unique_seqs)[:num_seq]

def run_validation_batch(classifier_model, seq_list, feature_prott5_instance, scaler_instance, device):
    if not seq_list: return [None] * len(seq_list)
    try:
        all_features = [extract_features(s, feature_prott5_instance) for s in seq_list]
        features_scaled = scaler_instance.transform(np.array(all_features))
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            scaled_logits = classifier_model(features_tensor)
            probs_tensor = torch.sigmoid(scaled_logits).squeeze()
            if probs_tensor.ndim == 0: return [probs_tensor.item()]
            else: return probs_tensor.cpu().numpy().tolist()
    except Exception as e:
        print(f"验证批次时出错: {e}")
        return [None] * len(seq_list)

def cluster_sequences(generator, sequences, num_clusters, device):
    if not sequences or len(sequences) < num_clusters:
        return sequences[:num_clusters]
    with torch.no_grad():
        token_ids_list = []
        max_len = max(len(seq) for seq in sequences) + 2
        for seq in sequences:
            ids = [np.random.randint(2, VOCAB_SIZE)] + [_token2id.get(aa, 0) for aa in seq] + [generator.eos_token_id]
            ids += [_token2id["<PAD>"]] * (max_len - len(ids))
            token_ids_list.append(ids)
        input_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)
        
        # !! 关键修正点 2 !!
        # 直接调用我们暴露出来的 embed_tokens 层
        embeddings = generator.embed_tokens(input_ids)

        mask = (input_ids != _token2id["<PAD>"]).unsqueeze(-1).float()
        embeddings = embeddings * mask
        lengths = mask.sum(dim=1)
        seq_embeds = embeddings.sum(dim=1) / (lengths + 1e-9)
        seq_embeds_np = seq_embeds.cpu().numpy()
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=42, n_init='auto').fit(seq_embeds_np)
    reps = []
    for i in range(int(num_clusters)):
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

# =====================================================================================
# --- 主执行流程 ---
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description="使用LoRA增强的生成器，生成高质量抗氧化肽。")
    parser.add_argument("--target_count", type=int, default=1000)
    parser.add_argument("--diversity_factor", type=float, default=1.2)
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--output_file", type=str, default="generated_peptides_with_lora.fasta")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--base_model_path", type=str, default="./prott5/model/")
    parser.add_argument("--generator_lora_path", type=str, default="./generator_with_lora_output/final_lora_generator")
    parser.add_argument("--validator_lora_path", type=str, default="./lora_finetuned_prott5")
    parser.add_argument("--predictor_checkpoint", type=str, default="./predictor_with_lora_checkpoints/final_predictor_with_lora.pth")
    parser.add_argument("--scaler_path", type=str, default="./predictor_with_lora_checkpoints/scaler_lora.pkl")
    parser.add_argument("--input_dim", type=int, default=1914)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- 步骤一：加载所有模型 ---")
    print("正在加载LoRA增强的生成器...")
    generator = AdvancedProtT5Generator(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.generator_lora_path,
        vocab_size=VOCAB_SIZE
    )
    lm_head_path = os.path.join(args.generator_lora_path, "lm_head.pth")
    if not os.path.exists(lm_head_path):
        raise FileNotFoundError(f"错误: 未找到生成器的lm_head权重文件: {lm_head_path}")
    generator.lm_head.load_state_dict(torch.load(lm_head_path, map_location=device))
    generator.to(device)
    print("生成器加载成功。")

    print("正在加载验证器...")
    scaler = joblib.load(args.scaler_path)
    feature_extractor = LoRAProtT5Extractor(args.base_model_path, args.validator_lora_path)
    predictor = AntioxidantPredictor(args.input_dim, args.transformer_layers, args.transformer_heads, args.transformer_dropout)
    predictor.load_state_dict(torch.load(args.predictor_checkpoint, map_location=device)); predictor.to(device)
    print("验证器所有组件加载成功。")

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
