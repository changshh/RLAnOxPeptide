#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler 
import torch
from transformers import T5EncoderModel, T5Tokenizer 


class ProtT5Model:
  
    def __init__(self, model_path, finetuned_model_file=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 尝试加载本地文件，如果失败，transformers库可能会尝试从hub下载（取决于配置）
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
            self.model = T5EncoderModel.from_pretrained(model_path, local_files_only=True)
        except OSError: # OSError: Can't load tokenizer for '...'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '...' is the correct path to a directory containing all relevant files for a T5Tokenizer tokenizer.
            print(f"警告: 无法从本地路径 {model_path} 加载ProtT5模型/分词器。尝试从HuggingFace Hub下载（如果transformers配置允许）。")
            self.tokenizer = T5Tokenizer.from_pretrained(model_path.split('/')[-1] if '/' in model_path else model_path, do_lower_case=False) # 尝试使用模型名下载
            self.model = T5EncoderModel.from_pretrained(model_path.split('/')[-1] if '/' in model_path else model_path)


        if finetuned_model_file is not None and os.path.exists(finetuned_model_file):
            try:
                state_dict = torch.load(finetuned_model_file, map_location=self.device)
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                print(f"加载微调权重 {finetuned_model_file}：缺失键 {missing_keys}, 意外键 {unexpected_keys}")
            except Exception as e:
                print(f"加载微调权重 {finetuned_model_file} 失败: {e}")
        
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sequence):
        if not sequence or not isinstance(sequence, str): # 增加对空序列或非字符串的检查
            print(f"警告: ProtT5Model.encode 接收到无效序列: {sequence}")
            return np.zeros((1, 1024), dtype=np.float32) # (1, hidden_dim)

        seq_spaced = " ".join(list(sequence)) # 修改变量名以避免覆盖外部seq
        try:
            encoded_input = self.tokenizer(seq_spaced, return_tensors='pt', padding=True, truncation=True, max_length=1022) # ProtT5通常最大长度1024，tokenized后可能更长
        except Exception as e:
            print(f"分词失败序列 '{sequence[:30]}...': {e}")
            return np.zeros((1, 1024), dtype=np.float32)

        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            try:
                embedding = self.model(**encoded_input).last_hidden_state  # (batch_size, seq_len, hidden_dim)
            except Exception as e:
                print(f"ProtT5模型推理失败序列 '{sequence[:30]}...': {e}")
                return np.zeros((1, 1024), dtype=np.float32)

        emb = embedding.squeeze(0).cpu().numpy()  # (seq_len, hidden_dim)
        if emb.shape[0] == 0: # 如果由于某种原因序列长度为0
             return np.zeros((1, 1024), dtype=np.float32)
        return emb

def load_fasta(fasta_file):
    # (您的 load_fasta 实现)
    sequences = []
    try:
        with open(fasta_file, 'r') as f:
            current_seq_lines = []
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if current_seq_lines: sequences.append("".join(current_seq_lines))
                    current_seq_lines = []
                else: current_seq_lines.append(line)
            if current_seq_lines: sequences.append("".join(current_seq_lines))
    except FileNotFoundError: print(f"文件未找到: {fasta_file}"); return []
    return sequences

def load_fasta_with_labels(fasta_file):
    # (您的 load_fasta_with_labels 实现)
    sequences, labels = [], []
    try:
        with open(fasta_file, 'r') as f:
            current_seq_lines, current_label = [], None
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if current_seq_lines:
                        sequences.append("".join(current_seq_lines))
                        labels.append(current_label if current_label is not None else 0) # Default label 0
                    current_seq_lines = []
                    current_label = int(line[1]) if len(line) > 1 and line[1] in ['0', '1'] else 0
                else: current_seq_lines.append(line)
            if current_seq_lines:
                sequences.append("".join(current_seq_lines))
                labels.append(current_label if current_label is not None else 0)
    except FileNotFoundError: print(f"文件未找到: {fasta_file}"); return [],[]
    return sequences, labels


def compute_amino_acid_composition(seq):
    if not seq: return {aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    # (您的 compute_amino_acid_composition 实现)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_len = len(seq)
    return {aa: seq.upper().count(aa) / seq_len for aa in amino_acids}


def compute_reducing_aa_ratio(seq):
    if not seq: return 0.0
    # (您的 compute_reducing_aa_ratio 实现)
    reducing = ['C', 'M', 'W']
    return sum(seq.upper().count(aa) for aa in reducing) / len(seq) if len(seq) > 0 else 0.0

def compute_physicochemical_properties(seq):
    if not seq or not all(c.upper() in "ACDEFGHIKLMNPQRSTVWYXUBZ" for c in seq): # ProteinAnalysis might fail on invalid chars
        return 0.0, 0.0, 0.0 # Default values
    try:
        analysis = ProteinAnalysis(str(seq).upper().replace('X','A').replace('U','C').replace('B','N').replace('Z','Q')) # Replace non-standard with common ones for analysis
        return analysis.gravy(), analysis.isoelectric_point(), analysis.molecular_weight()
    except Exception: # Catch any error from ProteinAnalysis
        return 0.0, 7.0, 110.0 * len(seq) # Rough defaults

def compute_electronic_features(seq):
    if not seq: return 0.0, 0.0
    # (您的 compute_electronic_features 实现)
    electronegativity = {'A':1.8,'C':2.5,'D':3.0,'E':3.2,'F':2.8,'G':1.6,'H':2.4,'I':4.5,'K':3.0,'L':4.2,'M':4.5,'N':2.0,'P':3.5,'Q':3.5,'R':2.5,'S':1.8,'T':2.5,'V':4.0,'W':5.0,'Y':4.0}
    values = [electronegativity.get(aa.upper(), 2.5) for aa in seq]
    avg_val = sum(values) / len(values) if values else 2.5
    return avg_val + 0.1, avg_val - 0.1


def compute_dimer_frequency(seq):
    if len(seq) < 2: return np.zeros(400) # 20*20
    # (您的 compute_dimer_frequency 实现)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dimer_counts = {aa1+aa2: 0 for aa1 in amino_acids for aa2 in amino_acids}
    for i in range(len(seq) - 1):
        dimer = seq[i:i+2].upper()
        if dimer in dimer_counts: dimer_counts[dimer] += 1
    total = max(len(seq) - 1, 1)
    for key in dimer_counts: dimer_counts[key] /= total
    return np.array([dimer_counts[d] for d in sorted(dimer_counts.keys())])


def positional_encoding(seq_len_actual, L_fixed=29, d_model=16): # Pass actual sequence length or use L_fixed
    # (您的 positional_encoding 实现)
    # This PE is fixed length, not dependent on actual seq len if L_fixed is used.
    # For random short sequences, this fixed PE might be an issue.
    # A more dynamic PE or no PE for very short sequences might be better.
    # However, to match current model input, we keep it.
    pos_enc = np.zeros((L_fixed, d_model))
    for pos in range(L_fixed):
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            pos_enc[pos, i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)
    return pos_enc.flatten()


def perturb_sequence(seq, perturb_rate=0.1, critical=['C', 'M', 'W']):
    # (您的 perturb_sequence 实现)
    if not seq: return ""
    seq_list = list(seq)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for i, aa in enumerate(seq_list):
        if aa.upper() not in critical and random.random() < perturb_rate:
            seq_list[i] = random.choice([x for x in amino_acids if x != aa.upper()])
    return "".join(seq_list)


def extract_features(seq, prott5_model_instance, L_fixed=29, d_model_pe=16): # Renamed d_model to d_model_pe
    if not seq or not isinstance(seq, str) or len(seq) == 0:
        print(f"警告: extract_features 接收到空或无效序列。返回零特征。")
        # 返回一个与预期特征维度匹配的零向量
        # 1024 (protT5) + 20 (aac) + 1 (red_ratio) + 3 (phys) + 2 (elec) + 400 (dimer) + L_fixed*d_model_pe (pos_enc)
        # Example: 1024 + 20 + 1 + 3 + 2 + 400 + 29*16 = 1024 + 20 + 1 + 3 + 2 + 400 + 464 = 1914
        return np.zeros(1024 + 20 + 1 + 3 + 2 + 400 + (L_fixed * d_model_pe))


    embedding = prott5_model_instance.encode(seq) # prott5_model is now an instance
    prot_embed = np.mean(embedding, axis=0) if embedding.shape[0] > 0 else np.zeros(embedding.shape[1] if embedding.ndim > 1 else 1024) # Handle empty embedding
    if prot_embed.shape[0] != 1024: # Ensure consistent ProtT5 embedding dim
        # print(f"警告: ProtT5 嵌入维度异常 ({prot_embed.shape[0]}) for seq '{seq[:20]}...'. 使用零向量。")
        prot_embed = np.zeros(1024)


    aa_comp = compute_amino_acid_composition(seq)
    aa_comp_vector = np.array([aa_comp[aa] for aa in "ACDEFGHIKLMNPQRSTVWY"])
    red_ratio = np.array([compute_reducing_aa_ratio(seq)])
    gravy, pI, mol_weight = compute_physicochemical_properties(seq)
    phys_props = np.array([gravy, pI, mol_weight])
    HOMO, LUMO = compute_electronic_features(seq)
    electronic = np.array([HOMO, LUMO])
    dimer_vector = compute_dimer_frequency(seq)
    pos_enc = positional_encoding(len(seq), L_fixed, d_model_pe) # Pass actual length, though current PE uses L_fixed
    
    features = np.concatenate([prot_embed, aa_comp_vector, red_ratio, phys_props, electronic, dimer_vector, pos_enc])
    return features

##############################################
# 主接口函数 prepare_features
##############################################
def prepare_features(neg_fasta, pos_fasta, prott5_model_path, additional_params=None):
    neg_seqs = load_fasta(neg_fasta)
    pos_seqs = load_fasta(pos_fasta)
    
    if not neg_seqs and not pos_seqs:
        raise ValueError("未能从FASTA文件加载任何序列。请检查文件路径和内容。")

    neg_labels = [0] * len(neg_seqs)
    pos_labels = [1] * len(pos_seqs)
    sequences = neg_seqs + pos_seqs
    labels = neg_labels + pos_labels

    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)
    sequences = list(sequences)
    labels = list(labels)

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.1, random_state=42, stratify=labels if len(np.unique(labels)) > 1 else None
    )
    print("训练集原始样本数：", len(train_seqs))
    print("验证集原始样本数：", len(val_seqs))

    if additional_params is not None and additional_params.get("augment", False):
        # (数据增强逻辑 - 如果启用)
        augmented_seqs, augmented_labels = [], []
        perturb_rate = additional_params.get("perturb_rate", 0.1)
        for seq, label in zip(train_seqs, train_labels):
            aug_seq = perturb_sequence(seq, perturb_rate=perturb_rate)
            augmented_seqs.append(aug_seq)
            augmented_labels.append(label)
        train_seqs.extend(augmented_seqs)
        train_labels.extend(augmented_labels)
        print("数据增强后训练集样本数：", len(train_seqs))


    finetuned_model_file = additional_params.get("finetuned_model_file") if additional_params else None
    # 创建 ProtT5Model 实例
    prott5_model_instance = ProtT5Model(prott5_model_path, finetuned_model_file=finetuned_model_file)

    def process_data(seqs_list): # Renamed seqs to seqs_list
        feature_list = []
        for s_item in seqs_list: # Renamed s to s_item
            # 将 ProtT5Model 实例传递给 extract_features
            features = extract_features(s_item, prott5_model_instance) 
            feature_list.append(features)
        return np.array(feature_list)

    X_train = process_data(train_seqs)
    X_val = process_data(val_seqs)
    
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("特征提取后训练集或验证集为空。请检查序列数据和特征提取过程。")


    # --- **关键修改：使用 RobustScaler** ---
    # scaler = StandardScaler() # 原来的 StandardScaler
    scaler = RobustScaler() 
    print("使用 RobustScaler 进行特征归一化。")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled, np.array(train_labels), np.array(val_labels), scaler

if __name__ == "__main__"
    neg_fasta_test = "dummy_data/test_neg.fasta"
    pos_fasta_test = "dummy_data/test_pos.fasta"
    prott5_path_test = "dummy_prott5_model/" # 需要一个包含config.json, pytorch_model.bin等的目录结构
    
    os.makedirs("dummy_data", exist_ok=True)
    os.makedirs(prott5_path_test, exist_ok=True) # 创建虚拟模型目录

    if not os.path.exists(neg_fasta_test):
        with open(neg_fasta_test, "w") as f: f.write(">neg1\nKALKALKALK\n>neg2\nPEPTPEPT\n")
    if not os.path.exists(pos_fasta_test):
        with open(pos_fasta_test, "w") as f: f.write(">pos1\nAOPPEPTIDE\n>pos2\nTRYTRYTRY\n")
    

    if not os.listdir(prott5_path_test): # 如果目录为空
        print(f"警告: {prott5_path_test} 为空。ProtT5Model可能尝试从HuggingFace Hub下载模型。")
        print(f"请确保您已下载Rostlab/ProstT5-XL-UniRef50或类似模型到该路径，或使用其HuggingFace名称。")
      
    additional_params_test = {
        "augment": False, 
        "perturb_rate": 0.1, 
        "finetuned_model_file": None # 测试时不使用微调模型
    }
    
    print("开始测试 prepare_features (使用RobustScaler)...")
    try:
        X_train_t, X_val_t, y_train_t, y_val_t, scaler_t = prepare_features(
            neg_fasta_test, 
            pos_fasta_test, 
            "Rostlab/ProstT5-XL-UniRef50", # 使用HuggingFace模型名称，如果本地路径无效
            additional_params_test
        )
        print("prepare_features 测试完成。")
        print("训练集样本数：", X_train_t.shape[0])
        print("验证集样本数：", X_val_t.shape[0])
        if X_train_t.shape[0] > 0:
            print("训练集特征维度:", X_train_t.shape[1])
            print("一个缩放后的训练样本 (前5个特征):", X_train_t[0, :5])
        if scaler_t:
            print("Scaler类型:", type(scaler_t))
    except Exception as e:
        print(f"prepare_features 测试失败: {e}")
        print("这可能是由于无法加载ProtT5模型或FASTA文件处理问题。请检查路径和文件内容。")

