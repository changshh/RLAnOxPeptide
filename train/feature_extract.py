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

# ProtT5Model, load_fasta, load_fasta_with_labels,
# compute_amino_acid_composition, compute_reducing_aa_ratio,
# compute_physicochemical_properties, compute_electronic_features,
# compute_dimer_frequency, positional_encoding, perturb_sequence,





class ProtT5Model:
    """Wrapper for loading ProtT5 and generating sequence embeddings."""
    def __init__(self, model_path, finetuned_model_file=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
            self.model = T5EncoderModel.from_pretrained(model_path, local_files_only=True)
        except OSError: # OSError: Can't load tokenizer for '...'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '...' is the correct path to a directory containing all relevant files for a T5Tokenizer tokenizer.
            print(f"Could not load ProtT5 locally from {model_path}; trying remote model resolution.")
            self.tokenizer = T5Tokenizer.from_pretrained(model_path.split('/')[-1] if '/' in model_path else model_path, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(model_path.split('/')[-1] if '/' in model_path else model_path)


        if finetuned_model_file is not None and os.path.exists(finetuned_model_file):
            try:
                state_dict = torch.load(finetuned_model_file, map_location=self.device)
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded fine-tuned weights from {finetuned_model_file}; missing keys={missing_keys}, unexpected keys={unexpected_keys}")
            except Exception as e:
                print(f"Failed to load fine-tuned weights from {finetuned_model_file}: {e}")

        self.model.to(self.device)
        self.model.eval()

    def encode(self, sequence):
        if not sequence or not isinstance(sequence, str):
            print(f"Invalid sequence received by ProtT5Model.encode: {sequence}")
            return np.zeros((1, 1024), dtype=np.float32)

        seq_spaced = " ".join(list(sequence))
        try:
            encoded_input = self.tokenizer(seq_spaced, return_tensors='pt', padding=True, truncation=True, max_length=1022)
        except Exception as e:
            print(f"Feature extraction failed for sequence {sequence[:30]}...: {e}")
            return np.zeros((1, 1024), dtype=np.float32)

        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            try:
                embedding = self.model(**encoded_input).last_hidden_state  # (batch_size, seq_len, hidden_dim)
            except Exception as e:
                print(f"Feature extraction failed for sequence {sequence[:30]}...: {e}")
                return np.zeros((1, 1024), dtype=np.float32)

        emb = embedding.squeeze(0).cpu().numpy()  # (seq_len, hidden_dim)
        if emb.shape[0] == 0:
             return np.zeros((1, 1024), dtype=np.float32)
        return emb


# load_fasta, load_fasta_with_labels, compute_amino_acid_composition, ... extract_features



def load_fasta(fasta_file):

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
    except FileNotFoundError:
        print(f"File not found: {fasta_file}")
        return []
    return sequences

def load_fasta_with_labels(fasta_file):

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
    except FileNotFoundError:
        print(f"File not found: {fasta_file}")
        return [], []
    return sequences, labels


def compute_amino_acid_composition(seq):
    if not seq: return {aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_len = len(seq)
    return {aa: seq.upper().count(aa) / seq_len for aa in amino_acids}


def compute_reducing_aa_ratio(seq):
    if not seq: return 0.0

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

    electronegativity = {'A':1.8,'C':2.5,'D':3.0,'E':3.2,'F':2.8,'G':1.6,'H':2.4,'I':4.5,'K':3.0,'L':4.2,'M':4.5,'N':2.0,'P':3.5,'Q':3.5,'R':2.5,'S':1.8,'T':2.5,'V':4.0,'W':5.0,'Y':4.0}
    values = [electronegativity.get(aa.upper(), 2.5) for aa in seq]
    avg_val = sum(values) / len(values) if values else 2.5
    return avg_val + 0.1, avg_val - 0.1


def compute_dimer_frequency(seq):
    if len(seq) < 2: return np.zeros(400) # 20*20

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dimer_counts = {aa1+aa2: 0 for aa1 in amino_acids for aa2 in amino_acids}
    for i in range(len(seq) - 1):
        dimer = seq[i:i+2].upper()
        if dimer in dimer_counts: dimer_counts[dimer] += 1
    total = max(len(seq) - 1, 1)
    for key in dimer_counts: dimer_counts[key] /= total
    return np.array([dimer_counts[d] for d in sorted(dimer_counts.keys())])


def positional_encoding(seq_len_actual, L_fixed=29, d_model=16): # Pass actual sequence length or use L_fixed

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

    if not seq: return ""
    seq_list = list(seq)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for i, aa in enumerate(seq_list):
        if aa.upper() not in critical and random.random() < perturb_rate:
            seq_list[i] = random.choice([x for x in amino_acids if x != aa.upper()])
    return "".join(seq_list)


def extract_features(seq, prott5_model_instance, L_fixed=29, d_model_pe=16): # Renamed d_model to d_model_pe
    if not seq or not isinstance(seq, str) or len(seq) == 0:
        print("Invalid sequence received by extract_features; returning zero features.")
        # 1024 (protT5) + 20 (aac) + 1 (red_ratio) + 3 (phys) + 2 (elec) + 400 (dimer) + L_fixed*d_model_pe (pos_enc)
        # Example: 1024 + 20 + 1 + 3 + 2 + 400 + 29*16 = 1024 + 20 + 1 + 3 + 2 + 400 + 464 = 1914
        return np.zeros(1024 + 20 + 1 + 3 + 2 + 400 + (L_fixed * d_model_pe))


    embedding = prott5_model_instance.encode(seq) # prott5_model is now an instance
    prot_embed = np.mean(embedding, axis=0) if embedding.shape[0] > 0 else np.zeros(embedding.shape[1] if embedding.ndim > 1 else 1024) # Handle empty embedding
    if prot_embed.shape[0] != 1024: # Ensure consistent ProtT5 embedding dim

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

##############################################
def prepare_features(neg_fasta, pos_fasta, prott5_model_path, additional_params=None):
    neg_seqs = load_fasta(neg_fasta)
    pos_seqs = load_fasta(pos_fasta)

    if not neg_seqs and not pos_seqs:
        raise ValueError('Invalid input data.')

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
    print("Raw train size:", len(train_seqs))
    print("Raw validation size:", len(val_seqs))

    if additional_params is not None and additional_params.get("augment", False):

        augmented_seqs, augmented_labels = [], []
        perturb_rate = additional_params.get("perturb_rate", 0.1)
        for seq, label in zip(train_seqs, train_labels):
            aug_seq = perturb_sequence(seq, perturb_rate=perturb_rate)
            augmented_seqs.append(aug_seq)
            augmented_labels.append(label)
        train_seqs.extend(augmented_seqs)
        train_labels.extend(augmented_labels)
        print("Augmented train size:", len(train_seqs))


    finetuned_model_file = additional_params.get("finetuned_model_file") if additional_params else None

    prott5_model_instance = ProtT5Model(prott5_model_path, finetuned_model_file=finetuned_model_file)

    def process_data(seqs_list): # Renamed seqs to seqs_list
        feature_list = []
        for s_item in seqs_list: # Renamed s to s_item

            features = extract_features(s_item, prott5_model_instance)
            feature_list.append(features)
        return np.array(feature_list)

    X_train = process_data(train_seqs)
    X_val = process_data(val_seqs)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError('Invalid input data.')




    scaler = RobustScaler()
    print("Scaling features with RobustScaler.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, np.array(train_labels), np.array(val_labels), scaler
