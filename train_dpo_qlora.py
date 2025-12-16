"""
DPO Training with QLoRA for Llama-3-8B
Using TRL, PEFT, and HH-RLHF dataset
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import DPOTrainer, DPOConfig
import wandb

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Base model
OUTPUT_DIR = "./llama3-8b-dpo-qlora"
DATASET_NAME = "Anthropic/hh-rlhf"  # Can change to "openbmb/UltraFeedback" 

# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Configuration
peft_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# DPO Training Configuration
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=2,
    remove_unused_columns=False,

    local_rank=-1,  # torchrun will handle this
    ddp_find_unused_parameters=False,
    report_to="wandb",

    beta=0.1,  # DPO beta parameter
    max_prompt_length=512,
    max_length=1024,
)

def load_and_prepare_dataset(dataset_name):
    """Load and format the preference dataset"""
    print(f"Loading dataset: {dataset_name}")
    
    if "hh-rlhf" in dataset_name.lower():
        # Load HH-RLHF dataset
        dataset = load_dataset(dataset_name)
        
        def format_hh_rlhf(example):
            """Format HH-RLHF to DPO format"""
            return {
                "prompt": example["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:",
                "chosen": example["chosen"].split("\n\nAssistant:")[-1].strip(),
                "rejected": example["rejected"].split("\n\nAssistant:")[-1].strip(),
            }
        
        dataset = dataset.map(format_hh_rlhf)
        
    elif "ultrafeedback" in dataset_name.lower():
        # Load UltraFeedback dataset
        dataset = load_dataset(dataset_name)
        
        def format_ultrafeedback(example):
            """Format UltraFeedback to DPO format"""
            # UltraFeedback has instruction, chosen, rejected format
            return {
                "prompt": example["instruction"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }
        
        dataset = dataset.map(format_ultrafeedback)
    
    return dataset

def main():
    # Initialize wandb
    #wandb.init(project="llama3-dpo-qlora", name="llama3-8b-dpo-run")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for DPO
    
    print("Loading model with QLoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        #device_map="auto", #This tells accelerate to place model layers across devices, might make data parallelism harder
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load reference model (frozen, for DPO)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        #device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(DATASET_NAME)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else dataset["train"].select(range(100))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Print example
    print("\nExample training sample:")
    print(f"Prompt: {train_dataset[0]['prompt'][:100]}...")
    print(f"Chosen: {train_dataset[0]['chosen'][:100]}...")
    print(f"Rejected: {train_dataset[0]['rejected'][:100]}...")
    
    # Initialize DPO Trainer
    print("\nInitializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    #wandb.finish()

if __name__ == "__main__":
    main()
