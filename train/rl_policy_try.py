#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn

# 候选解释及对应的动作数 (保持不变)
CANDIDATE_EXPLANATIONS = [
    "", "自由基中和", "抗氧化", "自由基中和 抗氧化", "抗氧化 自由基中和", "其他"
]
NUM_EXPLANATION_ACTIONS = len(CANDIDATE_EXPLANATIONS)

class AntioxidantRLEnv(gym.Env):
    def __init__(self, features, labels, baseline_model):
        super(AntioxidantRLEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.baseline_model = baseline_model
        self.input_dim = features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dim + 1,), dtype=np.float32)
        self.action_space = spaces.Dict({
            "class": spaces.Discrete(2),
            "explanation": spaces.Discrete(NUM_EXPLANATION_ACTIONS)
        })
        self.current_idx = None
        self.current_label = None

    def reset(self):
        self.current_idx = np.random.randint(0, len(self.features))
        feature = self.features[self.current_idx]
        self.current_label = int(self.labels[self.current_idx])
        
        self.baseline_model.eval()
        with torch.no_grad():
            device = next(self.baseline_model.parameters()).device
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
            raw_logits = self.baseline_model(feature_tensor) # 模型输出校准后/原始 logits
            
            # pred_value 仍然是类别1的概率，用于状态构建
            # 注意：如果模型内部已应用T，这里的sigmoid是在T-scaled logits上操作
            prob_class1 = torch.sigmoid(raw_logits.squeeze()) 
            pred_value = prob_class1.item()

        state = np.concatenate([feature, np.array([pred_value])])
        return state.astype(np.float32)

    def step(self, action):
        # action 是一个字典: {"class": predicted_class_idx, "explanation": explanation_idx}
        
        # 【关键修改】RL奖励函数解耦
        # 不再使用 baseline_model 的 confidence 来缩放奖励
        
        is_correct = (action["class"] == self.current_label)
        
        # 固定的分类奖励
        if is_correct:
            reward_class = 1.0  # 正确分类奖励
        else:
            reward_class = -1.0 # 错误分类惩罚
            # 可以考虑对高置信度错误给予更大惩罚，但这需要获取置信度
            # 如果需要，可以从 baseline_model 重新获取 logits 并计算置信度，但这里我们先简化
            # self.baseline_model.eval()
            # with torch.no_grad():
            #     device = next(self.baseline_model.parameters()).device
            #     feature_tensor = torch.tensor(self.features[self.current_idx], dtype=torch.float32).unsqueeze(0).to(device)
            #     raw_logits = self.baseline_model(feature_tensor)
            #     prob_predicted_class = torch.sigmoid(raw_logits.squeeze()) if action["class"] == 1 else 1 - torch.sigmoid(raw_logits.squeeze())
            #     if prob_predicted_class.item() > 0.8: # 如果对错误分类非常自信
            #         reward_class -= 0.5 # 额外惩罚

        # 解释奖励 (与原逻辑相同或可调整)
        explanation_text = CANDIDATE_EXPLANATIONS[action["explanation"]]
        count_keywords = 0
        if "自由基中和" in explanation_text: count_keywords += 1
        if "抗氧化" in explanation_text: count_keywords += 1
        
        if count_keywords >= 2: reward_explanation = 0.5
        elif count_keywords == 0 and explanation_text != "": reward_explanation = -0.5
        elif explanation_text == "": reward_explanation = -0.2
        else: reward_explanation = 0.0
            
        reward = reward_class + reward_explanation
        done = True 
        info = {"true_label": self.current_label, "action_taken": action}
        
        next_state = self.reset() 
        return next_state, reward, done, info

# Attention 和 RLPolicyNetwork 类保持不变 (与您之前版本一致)
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.fc_attention = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        attention_scores = torch.softmax(self.fc_attention(x), dim=-1)
        return x * attention_scores

class RLPolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(RLPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier_head = nn.Linear(hidden_dim, 2)
        self.explanation_head = nn.Linear(hidden_dim, NUM_EXPLANATION_ACTIONS)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        residual = x
        x = torch.relu(self.fc2(x))
        x = self.attention(x)
        x_att = x
        x = torch.relu(self.fc3(x))
        x = x + residual
        class_logits = self.classifier_head(x)
        explanation_logits = self.explanation_head(x)
        state_value = self.value_head(x)
        return class_logits, explanation_logits, state_value

if __name__ == '__main__':
    # 简单测试 (与之前类似)
    num_dummy_samples = 10
    feature_dim_dummy = 1914 
    dummy_features = np.random.randn(num_dummy_samples, feature_dim_dummy).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, size=num_dummy_samples)

    from antioxidant_predictor_5 import AntioxidantPredictor 
    dummy_baseline_model = AntioxidantPredictor(input_dim=feature_dim_dummy)
    device_test = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_baseline_model.to(device_test)

    env = AntioxidantRLEnv(dummy_features, dummy_labels, dummy_baseline_model)
    initial_state = env.reset()
    action_sample = env.action_space.sample()
    next_state, reward, done, info = env.step(action_sample)
    print(f"Sample action: {action_sample}, Reward: {reward}, Done: {done}, Info: {info}")
    print("RL Environment with new reward logic basic tests passed.")
