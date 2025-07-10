#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import copy
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

try:
    from feature_extract import prepare_features, ProtT5Model, extract_features, load_fasta_with_labels
    from antioxidant_predictor_5 import AntioxidantPredictor
    from rl_policy_try import RLPolicyNetwork, AntioxidantRLEnv
    from pretrain_finetune import finetune_protT5_on_antioxidant_proteins, train_feature_fusion_classifier
except ImportError as e: print(f"导入模块失败: {e}"); exit()

class TemperatureCalibrator:
    def __init__(self, model, device): self.model = model; self.device = device
    def calibrate(self, X_val, y_val, initial_temp=1.0, lr=0.01, n_iters=100):
        self.model.set_temperature(initial_temp, self.device) # 确保使用传入的initial_temp
        print(f"开始温度校准 (Adam, lr={lr}, iters={n_iters}, T_initial_set={initial_temp:.4f}, T_model_current={self.model.get_temperature():.4f})...")
        self.model.to(self.device); self.model.eval(); self.model.temperature.requires_grad = True
        criterion_cal = nn.BCEWithLogitsLoss().to(self.device)
        optimizer_temp = optim.Adam([self.model.temperature], lr=lr)
        val_inputs = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        val_labels_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        print(f"  校准开始时模型内部 T = {self.model.temperature.item():.4f}")
        for iter_num in range(n_iters):
            optimizer_temp.zero_grad(); calibrated_logits = self.model(val_inputs)
            loss = criterion_cal(calibrated_logits, val_labels_tensor); loss.backward(); optimizer_temp.step()
            with torch.no_grad():
                if self.model.temperature.item() < 0.1: self.model.temperature.data.fill_(0.1)
            if (iter_num + 1) % (n_iters // 10 if n_iters >=10 else 1) == 0: print(f"  校准迭代 {iter_num+1}: Temp={self.model.temperature.item():.4f}, Loss={loss.item():.4f}")
        print(f"温度校准完成。优化后 T = {self.model.temperature.item():.4f}")
        self.model.temperature.requires_grad = False; return self.model

def evaluate_model_with_threshold_custom(model, X, y, threshold, model_outputs_logits=True):
    model.eval(); device = next(model.parameters()).device
    current_temp = model.get_temperature() if hasattr(model,'get_temperature') and callable(model.get_temperature) else 1.0
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy() if model_outputs_logits else outputs.squeeze().cpu().numpy()
        if probs.ndim == 0: probs = np.array([probs.item()]) # Handle scalar output
        preds = (probs > threshold).astype(int)
    unique_y = np.unique(y); auc = roc_auc_score(y,probs) if len(unique_y)>1 else float('nan')
    # Handle cases where all predictions are the same or y is all one class for F1, Precision, Recall
    if len(unique_y) > 1:
        f1 = f1_score(y,preds,zero_division=0)
        prec = precision_score(y,preds,zero_division=0)
        rec = recall_score(y,preds,zero_division=0)
    elif len(y) > 0 and np.array_equal(y, preds): # All correct and single class
        f1, prec, rec = 1.0, 1.0, 1.0
    else: # All incorrect or empty y (though latter shouldn't happen with checks)
        f1, prec, rec = 0.0, 0.0, 0.0
    acc = accuracy_score(y,preds)
    print(f"评估 (T={current_temp:.2f}, thr={threshold}): AUC={auc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Acc={acc:.4f}")
    return {"AUC-ROC":auc, "F1-Score":f1, "Precision":prec, "Recall":rec, "Accuracy":acc, "Temperature":current_temp}


def supervised_training(X_train, y_train, X_val, y_val, params, calibrate_temp=True):
    device = params["device"]; print("开始纯监督训练...")
    model = train_feature_fusion_classifier(X_train, y_train, X_val, y_val, params)
    print("纯监督训练结束。")
    if calibrate_temp:
        print("对纯监督模型进行温度校准...");
        calibrator = TemperatureCalibrator(model, device)
        model = calibrator.calibrate(X_val, y_val, initial_temp=1.0, lr=0.01, n_iters=100)
        print("纯监督模型温度校准完成。")
    val_metrics = evaluate_model_with_threshold_custom(model, X_val, y_val, params["threshold"])
    return model, val_metrics

def rl_joint_training(X_train_rl, y_train_rl, X_val_rl, y_val_rl, base_classifier_params, rl_params, calibrate_temp_after_rl=True):
    device = base_classifier_params["device"]; input_dim = X_train_rl.shape[1]
    model = AntioxidantPredictor(input_dim, base_classifier_params.get('transformer_layers',3),
                                 base_classifier_params.get('transformer_heads',4),
                                 base_classifier_params.get('transformer_dropout',0.1)).to(device)
    model.load_state_dict(copy.deepcopy(base_classifier_params["state_dict"]))
    print(f"RL开始时，继承自监督阶段的 T = {model.get_temperature():.4f}")
    policy_net = RLPolicyNetwork(X_train_rl.shape[1]+1, 128).to(device)

    logit_penalty_weight_rl = base_classifier_params.get("logit_penalty_weight", 0.01)
    print(f"RL阶段监督损失的Logit惩罚权重 = {logit_penalty_weight_rl}")

    opt_model = optim.Adam(model.parameters(), lr=base_classifier_params.get("learning_rate_rl",5e-5), weight_decay=base_classifier_params.get("weight_decay",1e-5))
    opt_policy = optim.Adam(policy_net.parameters(), lr=rl_params["policy_lr"], weight_decay=rl_params.get("policy_weight_decay",1e-5))
    crit_bce_rl = nn.BCEWithLogitsLoss()

    num_epochs_rl = rl_params["num_epochs_rl"]; w_rl = rl_params["w_rl"]; threshold = rl_params["threshold"]
    batch_size = base_classifier_params.get("batch_size", 64); patience_rl = rl_params.get("patience_rl", 10)
    best_val_f1 = 0.0; patience_counter = 0; best_model_state, best_policy_state = None, None
    env = AntioxidantRLEnv(X_train_rl, y_train_rl, model)

    print(f"\n开始RL联合训练 (w_rl={w_rl}, logit_penalty_rl={logit_penalty_weight_rl})...")
    for epoch in range(num_epochs_rl):
        model.train(); policy_net.train(); model.temperature.requires_grad = False
        sup_loss_ep, rl_loss_ep, total_loss_ep, num_b, bce_loss_rl_ep, logit_reg_rl_ep = 0.0,0.0,0.0,0,0.0,0.0

        indices = np.arange(X_train_rl.shape[0]); np.random.shuffle(indices)
        for i in range(0, X_train_rl.shape[0], batch_size):
            num_b+=1; batch_idx = indices[i:i+batch_size]
            batch_X, batch_y = torch.tensor(X_train_rl[batch_idx],dtype=torch.float32).to(device), torch.tensor(y_train_rl[batch_idx],dtype=torch.float32).unsqueeze(1).to(device)

            scaled_logits = model(batch_X)
            bce_loss = crit_bce_rl(scaled_logits, batch_y)
            logit_reg = torch.mean(scaled_logits**2) if logit_penalty_weight_rl > 0 else torch.tensor(0.0,device=device)
            sup_loss = bce_loss + logit_penalty_weight_rl * logit_reg

            state = env.reset(); state_t = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(device)
            class_lg, expl_lg, _ = policy_net(state_t)
            act_cl = torch.distributions.Categorical(logits=class_lg).sample()
            act_ex = torch.distributions.Categorical(logits=expl_lg).sample()
            log_p = torch.distributions.Categorical(logits=class_lg).log_prob(act_cl) + torch.distributions.Categorical(logits=expl_lg).log_prob(act_ex)
            _, rew, _, _ = env.step({"class":act_cl.item(),"explanation":act_ex.item()})
            rl_l = -log_p * rew

            loss = sup_loss + w_rl * rl_l
            opt_model.zero_grad(); opt_policy.zero_grad(); loss.backward(); opt_model.step(); opt_policy.step()

            sup_loss_ep+=sup_loss.item(); rl_loss_ep+=rl_l.item(); total_loss_ep+=loss.item()
            bce_loss_rl_ep += bce_loss.item()
            if logit_penalty_weight_rl > 0 and isinstance(logit_reg, torch.Tensor): logit_reg_rl_ep += logit_reg.item()

        avg_s = sup_loss_ep/num_b if num_b > 0 else 0
        avg_r = rl_loss_ep/num_b if num_b > 0 else 0
        avg_t = total_loss_ep/num_b if num_b > 0 else 0
        avg_bce_r = bce_loss_rl_ep/num_b if num_b > 0 else 0
        avg_l_reg_r = logit_reg_rl_ep/num_b if num_b > 0 and logit_penalty_weight_rl > 0 else 0

        print(f"RL Ep{epoch+1}: TotalL={avg_t:.4f} (Sup(BCE={avg_bce_r:.4f},LogReg={avg_l_reg_r:.4f}), RL={avg_r:.4f}), T={model.get_temperature():.2f}")

        val_met = evaluate_model_with_threshold_custom(model,X_val_rl,y_val_rl,threshold)
        current_f1_rl = val_met["F1-Score"] if val_met["F1-Score"] is not None and not np.isnan(val_met["F1-Score"]) else 0.0

        if current_f1_rl > best_val_f1:
            best_val_f1=current_f1_rl; best_model_state=copy.deepcopy(model.state_dict()); best_policy_state=copy.deepcopy(policy_net.state_dict()); patience_counter=0
        else:
            patience_counter+=1
            if patience_counter>=patience_rl: print(f"RL早停 Ep{epoch+1}"); break

    if best_model_state: model.load_state_dict(best_model_state); print(f"加载RL最佳模型 (T={model.get_temperature():.4f})")
    if best_policy_state: policy_net.load_state_dict(best_policy_state); print("加载RL最佳策略")

    if calibrate_temp_after_rl and best_model_state:
        print("RL后模型温度校准...");
        print(f"校准前T={model.get_temperature():.4f}")
        calibrator = TemperatureCalibrator(model,device)
        model = calibrator.calibrate(X_val_rl,y_val_rl, initial_temp=model.get_temperature(), lr=0.01,n_iters=100)
        # 根据是否使用微调模型，修改保存文件名
        model_type_suffix = "_FINETUNED_PROTT5" if base_classifier_params.get("finetuned_model_in_use", False) else "_ORIGINAL_PROTT5"
        path = os.path.join("checkpoints",f"final_rl_model_logitp{logit_penalty_weight_rl}_calibrated{model_type_suffix}.pth")
        torch.save(model.state_dict(),path); print(f"最终RL模型保存至 {path} (T={model.get_temperature():.4f})")
    return model, policy_net

def add_explainability(model, X_sample, output_path="shap_summary.png", model_outputs_logits=True):
    try: import shap; import matplotlib.pyplot as plt
    except ImportError: print("请安装shap和matplotlib"); return
    device=next(model.parameters()).device
    def pred_wrap(x_np_arr):
        if not isinstance(x_np_arr, np.ndarray): x_np_arr = np.array(x_np_arr)
        if x_np_arr.ndim == 1: x_np_arr = x_np_arr.reshape(1, -1)

        out=model(torch.tensor(x_np_arr,dtype=torch.float32).to(device))
        res = torch.sigmoid(out).squeeze().cpu().numpy() if model_outputs_logits else out.squeeze().cpu().numpy()
        return res if res.ndim > 0 else res.reshape(1)

    if X_sample.shape[0]==0: print("SHAP:样本为空"); return
    bg_k = min(50, X_sample.shape[0])
    bg_data = shap.sample(X_sample, bg_k)
    if bg_data.shape[0]==0: print("SHAP:背景数据为空(采样后)"); return

    try:
        print("尝试使用 KernelExplainer 进行SHAP分析...")
        explainer=shap.KernelExplainer(pred_wrap, bg_data);
        shap_vals=explainer.shap_values(X_sample)

        plt.figure()
        shap.summary_plot(shap_vals, X_sample, show=False)
        plt.tight_layout();plt.savefig(output_path);plt.close()
        print(f"SHAP图保存至: {output_path}")
    except Exception as e: print(f"SHAP分析/绘图失败: {e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"设备: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # --- ProtT5 微调路径 ---
    print("\n--- ProtT5 微调阶段 (如果需要) ---")
    protein_csv_for_finetune = "/root/shared-nvme/chshhan/diffusion/2_5rl_env/AODB_PROTEIN.csv"
    finetuned_prott5_save_dir = "/root/shared-nvme/chshhan/diffusion/prott5/model/"
    finetuned_model_file_name = "finetuned_prott5.bin" # 定义微调模型文件名
    finetuned_model_file_path = os.path.join(finetuned_prott5_save_dir, finetuned_model_file_name)


    finetune_prot_params = {
        "max_length": 512, "mask_prob": 0.15, "batch_size": 16, "embed_dim": 512,
        "num_layers": 6, "num_heads": 8, "dropout": 0.1, "learning_rate": 1e-4,
        "num_epochs": 5, "device": device, "num_workers": 0
    }
    
    # 决定是否实际执行微调，或者只是检查文件是否存在
    PERFORM_FINETUNING = True # 设置为 True 以执行微调，False 则跳过实际微调过程
    finetuned_model_in_use = True # 标记是否使用了微调模型进行特征提取

    if PERFORM_FINETUNING:
        if not os.path.exists(finetuned_model_file_path):
            print(f"微调的ProtT5模型 {finetuned_model_file_path} 不存在，开始微调...")
            if os.path.exists(protein_csv_for_finetune):
                finetune_protT5_on_antioxidant_proteins(protein_csv_for_finetune, finetune_prot_params, model_path=finetuned_prott5_save_dir, output_filename=finetuned_model_file_name)
                print(f"ProtT5 微调完成。模型保存为 {finetuned_model_file_path}")
                finetuned_model_in_use = True # 微调后，标记为使用
            else:
                print(f"错误: ProtT5微调的CSV文件 {protein_csv_for_finetune} 不存在。跳过微调。")
                print("将使用基础 ProtT5 模型进行特征提取。")
        else:
            print(f"已找到预先微调的ProtT5模型: {finetuned_model_file_path}")
            finetuned_model_in_use = True # 找到了预训练模型，标记为使用
    else:
        print("PERFORM_FINETUNING 设置为 False，跳过 ProtT5 微调检查和执行。")
        if os.path.exists(finetuned_model_file_path):
            print(f"注意: 存在预微调模型 {finetuned_model_file_path}，但本次运行可能不使用它进行特征提取（取决于后续逻辑）。")
            # 如果希望即使PERFORM_FINETUNING=False也强制使用已存在的微调模型，可以在下面 additional_params 中设置
            # finetuned_model_in_use = True # 取消注释此行以在找到时使用
        else:
            print("未找到预微调模型，将使用基础 ProtT5 模型进行特征提取。")


    # --- 数据准备与特征工程路径 ---
    print("\n--- 数据准备与特征工程 ---")
    # 根据 finetuned_model_in_use 决定 additional_params 中的 finetuned_model_file
    # 如果想强制使用基础模型进行消融，即使微调模型存在，也应将 finetuned_model_file 设为 None
    # current_finetuned_path_for_features = finetuned_model_file_path if finetuned_model_in_use and os.path.exists(finetuned_model_file_path) else None
    
    # 为了明确是使用微调还是原始模型，我们可以在这里做一个选择，或者根据之前的逻辑
    # 如果是进行“不使用微调模型”的消融实验，这里应该强制为 None
    # 如果是“使用微调模型”的实验，这里应该是 finetuned_model_file_path (如果存在)
    
    # 假设这是“使用微调模型（如果存在并执行了微调）”的流程
    # 如果要强制进行“不使用微调”的消融，请将下面的 path_to_use_for_prott5_features 设为 None
    path_to_use_for_prott5_features = None 
    if finetuned_model_in_use and os.path.exists(finetuned_model_file_path):
        print(f"特征提取将使用微调后的 ProtT5 模型: {finetuned_model_file_path}")
        path_to_use_for_prott5_features = finetuned_model_file_path
        model_type_suffix_for_saving = "_FINETUNED_PROTT5" # 用于保存文件名
    else:
        print(f"特征提取将使用原始 ProtT5 模型。")
        path_to_use_for_prott5_features = None # 明确不使用微调文件
        model_type_suffix_for_saving = "_ORIGINAL_PROTT5" # 用于保存文件名
        finetuned_model_in_use = False # 确保标记正确

    additional_params = {
        "augment": False,
        "perturb_rate": 0.1,
        "finetuned_model_file": path_to_use_for_prott5_features
    }
    neg_fasta_path = "/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/clean_data/remaining_negative.fasta"
    pos_fasta_path = "/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/clean_data/remaining_positive.fasta"
    prott5_model_base_path = "/root/shared-nvme/chshhan/diffusion/prott5/model/"

    if not (os.path.exists(neg_fasta_path) and os.path.exists(pos_fasta_path)):
        print(f"错误: 训练FASTA文件未找到。路径:\nNeg: {neg_fasta_path}\nPos: {pos_fasta_path}"); exit(1)
    if not os.path.exists(prott5_model_base_path):
        print(f"错误: ProtT5基础模型路径 {prott5_model_base_path} 未找到。"); exit(1)

    X_train, X_val, y_train, y_val, scaler = prepare_features(neg_fasta_path, pos_fasta_path, prott5_model_base_path, additional_params)
    print(f"训练集样本数：{X_train.shape[0]}, 验证集样本数：{X_val.shape[0]}")
    scaler_save_path = os.path.join("checkpoints", f"scaler{model_type_suffix_for_saving}.pkl")
    joblib.dump(scaler, scaler_save_path); print(f"Scaler ({type(scaler).__name__}) 已保存至: {scaler_save_path}")

    # --- 独立测试集加载路径 ---
    print("\n--- 独立测试集加载 ---")
    independent_fasta_path = "/root/shared-nvme/chshhan/diffusion/2_11rl_new/2_7rl_env/clean_data/independent_test_cleaned.fasta"
    X_ind_scaled, labels_ind = None, None
    if os.path.exists(independent_fasta_path):
        seqs_ind, labels_ind_list = load_fasta_with_labels(independent_fasta_path)
        if seqs_ind:
            labels_ind = np.array(labels_ind_list)
            X_ind_list = []
            # 特征提取时使用的ProtT5模型应与训练时一致
            print(f"为独立测试集初始化 ProtT5Model (使用与训练时相同的设置: {'微调模型' if path_to_use_for_prott5_features else '原始模型'})")
            protT5_for_ind_test = ProtT5Model(prott5_model_base_path, finetuned_model_file=path_to_use_for_prott5_features)
            for s_val in seqs_ind:
                try: X_ind_list.append(extract_features(s_val, protT5_for_ind_test))
                except Exception as e: print(f"独立测试集特征提取失败 for '{s_val[:20]}...': {e}")
            if X_ind_list:
                X_ind = np.array(X_ind_list)
                if X_ind.shape[0] == labels_ind.shape[0] and X_ind.size > 0 :
                    X_ind_scaled = scaler.transform(X_ind); print(f"独立测试集样本数：{X_ind_scaled.shape[0]}")
                else: X_ind_scaled, labels_ind = None, None; print(f"独立测试集特征数与标签数不匹配或为空。")
            else: X_ind_scaled, labels_ind = None, None; print("独立测试集特征提取后为空。")
        else: print(f"从 {independent_fasta_path} 未解析到序列。")
    else: print(f"警告：独立测试集FASTA文件 {independent_fasta_path} 未找到。跳过独立测试评估。")

    logit_penalty_sup = 0.05
    logit_penalty_rl = 0.1

    sup_params = {
        'input_dim': X_train.shape[1], 'transformer_layers': 3, 'transformer_heads': 4, 'transformer_dropout': 0.1,
        'batch_size': 64, 'learning_rate': 1e-4, 'num_epochs': 10, 'device': device, 'threshold': 0.5,
        'patience': 15, 'weight_decay': 1e-5, 'label_smoothing': 0.1,
        'logit_penalty_weight': logit_penalty_sup
    }
    print(f"\n=== 监督训练 (LogitPenalty={logit_penalty_sup}, ProtT5: {'微调' if finetuned_model_in_use else '原始'}) ===")
    sup_model, sup_val_metrics = supervised_training(X_train,y_train,X_val,y_val,sup_params,calibrate_temp=True)
    print(f"监督模型(校准)验证指标 ({'微调ProtT5' if finetuned_model_in_use else '原始ProtT5'}):", sup_val_metrics)
    
    sup_model_save_path = os.path.join("checkpoints",f"sup_model_lp{logit_penalty_sup}_cal{model_type_suffix_for_saving}.pth")
    torch.save(sup_model.state_dict(), sup_model_save_path)
    print(f"监督模型已保存至: {sup_model_save_path} (T={sup_model.get_temperature():.4f})")

    # +++++ 新增：纯监督模型在独立测试集上的评估 +++++
    if X_ind_scaled is not None and labels_ind is not None and X_ind_scaled.size > 0 and labels_ind.size > 0:
        print(f"\n=== 纯监督模型独立测试集评估 (ProtT5: {'微调' if finetuned_model_in_use else '原始'}) ===")
        print(f"使用训练和校准后的纯监督模型 (T={sup_model.get_temperature():.4f}) 进行独立测试评估")
        sup_ind_test_metrics = evaluate_model_with_threshold_custom(sup_model, X_ind_scaled, labels_ind, threshold=sup_params["threshold"], model_outputs_logits=True)
        print(f"纯监督模型(已校准, ProtT5: {'微调' if finetuned_model_in_use else '原始'})独立测试集指标：", sup_ind_test_metrics)
    else:
        print("未找到有效独立测试集数据，跳过纯监督模型的独立测试评估。")
    # ++++++++++++++++++++++++++++++++++++++++++++++

    rl_base_params = sup_params.copy()
    rl_base_params["state_dict"] = sup_model.state_dict()
    rl_base_params["learning_rate_rl"] = 5e-5
    rl_base_params["logit_penalty_weight"] = logit_penalty_rl
    rl_base_params["finetuned_model_in_use"] = finetuned_model_in_use # 传递标记给RL训练，用于文件名

    rl_config = {'w_rl':0.01, 'num_epochs_rl':30, 'policy_lr':5e-4, 'policy_weight_decay':1e-5, 'threshold':0.5, 'patience_rl':10}
    print(f"\n=== RL训练 (LogitPenalty={logit_penalty_rl}, ProtT5: {'微调' if finetuned_model_in_use else '原始'}) ===")
    rl_model, policy_net = rl_joint_training(X_train,y_train,X_val,y_val,rl_base_params,rl_config,calibrate_temp_after_rl=True)

    policy_net_save_path = os.path.join("checkpoints",f"policy_net_lp{logit_penalty_rl}{model_type_suffix_for_saving}.pth")
    torch.save(policy_net.state_dict(), policy_net_save_path)
    print(f"RL联合训练后的策略网络已保存至 {policy_net_save_path}。")
    print(f"RL联合训练后的模型最终 T={rl_model.get_temperature():.4f}")

    if X_ind_scaled is not None and labels_ind is not None and X_ind_scaled.size > 0 and labels_ind.size > 0:
        print(f"\n=== 最终RL模型独立测试集评估 (ProtT5: {'微调' if finetuned_model_in_use else '原始'}) ===")
        final_model_path = os.path.join("checkpoints",f"final_rl_model_logitp{logit_penalty_rl}_calibrated{model_type_suffix_for_saving}.pth")

        if os.path.exists(final_model_path):
            eval_rl_model = AntioxidantPredictor(
                input_dim=X_ind_scaled.shape[1],
                transformer_layers=sup_params['transformer_layers'],
                transformer_heads=sup_params['transformer_heads'],
                transformer_dropout=sup_params['transformer_dropout']
            ).to(device)
            eval_rl_model.load_state_dict(torch.load(final_model_path, map_location=device))
            print(f"加载最终RL模型 {final_model_path} 进行评估 (T={eval_rl_model.get_temperature():.4f})")
            final_metrics = evaluate_model_with_threshold_custom(eval_rl_model, X_ind_scaled, labels_ind, threshold=rl_config["threshold"], model_outputs_logits=True)
            print(f"最终RL模型(已校准, ProtT5: {'微调' if finetuned_model_in_use else '原始'})独立测试集指标：", final_metrics)
        else:
            print(f"错误: 未找到最终校准的RL模型 {final_model_path} 进行评估。")

    if X_val.shape[0]>0: # Ensure validation set is not empty for SHAP
        print(f"\n=== SHAP分析 (最终RL模型, ProtT5: {'微调' if finetuned_model_in_use else '原始'}) ===")
        model_for_shap = rl_model
        if 'eval_rl_model' in locals() and eval_rl_model is not None:
            model_for_shap = eval_rl_model

        num_shap_samples = min(50, X_val.shape[0])
        idx_shap = np.random.choice(X_val.shape[0], num_shap_samples, replace=False)
        X_val_sample_for_shap = X_val[idx_shap]

        shap_output_path = f"shap_summary_final_rl_lp{logit_penalty_rl}_T{model_for_shap.get_temperature():.2f}{model_type_suffix_for_saving}.png"
        add_explainability(model_for_shap, X_val_sample_for_shap, output_path=shap_output_path, model_outputs_logits=True)

if __name__ == "__main__":
    main()
