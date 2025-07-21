import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig

# 定义一个包装模型，它将 lm_head 作为固定的一部分
class T5EncoderForMLM(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.encoder = base_model
        # 将 lm_head 定义为模型的持久部分
        self.lm_head = nn.Linear(base_model.config.d_model, len(tokenizer), bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None):
        # 从基础编码器获取输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        # 使用固定的 lm_head 进行预测
        logits = self.lm_head(encoder_outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() # 默认 ignore_index=-100
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def main(args):
    # --- 1. 加载基础模型和分词器 ---
    print(f"Loading base model from: {args.base_model_path}")
    model = T5EncoderModel.from_pretrained(args.base_model_path, device_map='auto')
    tokenizer = T5Tokenizer.from_pretrained(args.base_model_path, do_lower_case=False)

    # --- 为Tokenizer添加并设置mask_token ---
    print("Adding mask token to tokenizer...")
    mask_token = "<MASK>"
    if mask_token not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'mask_token': mask_token})
        model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.mask_token = mask_token
    tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    print(f"Tokenizer's mask token set to '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    
    # --- 2. 配置LoRA ---
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    
    lora_encoder = get_peft_model(model, lora_config)
    print("\nLoRA model configured. Trainable parameters:")
    lora_encoder.print_trainable_parameters()
    
    mlm_model = T5EncoderForMLM(lora_encoder, tokenizer)

    # --- 3. 准备数据集 ---
    print(f"Loading and processing dataset from: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    if 'Sequence' not in df.columns:
        if 'sequence' in df.columns:
            df.rename(columns={'sequence': 'Sequence'}, inplace=True)
            print("Renamed column 'sequence' to 'Sequence'.")
        else:
            raise ValueError("CSV file must contain a 'Sequence' column.")
        
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        sequences_spaced = [" ".join(list(seq)) for seq in examples["Sequence"]]
        return tokenizer(
            sequences_spaced,
            max_length=args.max_length,
            padding='max_length',
            truncation=True
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    print("Dataset tokenized.")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # --- 5. 设置训练参数并启动训练 ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=mlm_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n--- Starting LoRA fine-tuning ---")
    trainer.train()

    # --- 6. 保存LoRA适配器 ---
    print("\nFine-tuning complete. Saving LoRA adapters...")
    mlm_model.encoder.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapters and tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LoRA对ProtT5模型进行参数高效性微调（MLM任务）。")
    
    parser.add_argument("--base_model_path", type=str, default="/root/shared-nvme/chshhan/diffusion/prott5/model/", help="存放【原始】ProtT5模型文件的目录路径。")
    parser.add_argument("--dataset_path", type=str, default="/root/shared-nvme/chshhan/diffusion/2_5rl_env/AODB_PROTEIN.csv", help="用于微调的蛋白质序列CSV文件路径。")
    
    # --- (!! 关键修改点 !!) ---
    # 将 add_button 修正为正确的 add_argument
    parser.add_argument("--output_dir", type=str, default="./lora_finetuned_prott5", help="保存LoRA适配器权重和配置的输出目录。")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="微调时的学习率。")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮次。")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小。")
    parser.add_argument("--max_length", type=int, default=512, help="序列的最大长度。")
    
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA的秩 (rank)。")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA的alpha值。")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA层的dropout率。")
    
    args = parser.parse_args()
    main(args)
