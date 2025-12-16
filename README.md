# DPO Training with QLoRA for Llama-3-8B

This project implements Direct Preference Optimization (DPO) training for Llama-3-8B using QLoRA (Quantized LoRA) with the TRL and PEFT libraries.

## Overview

**What this does:**
- Fine-tunes Llama-3-8B using preference data (chosen vs rejected responses)
- Uses QLoRA for memory-efficient 4-bit quantized training
- Supports HH-RLHF and UltraFeedback datasets
- Produces a model aligned with human preferences

## Requirements

- GPU with at least 16GB VRAM (24GB recommended)
- Python 3.8+
- CUDA-compatible GPU

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for Llama-3 access)
huggingface-cli login

# Optional: Login to Weights & Biases for experiment tracking
wandb login
```

## Quick Start

### 1. Basic Training

```bash
python train_dpo_qlora.py
```

This will:
- Load Llama-3-8B with 4-bit quantization
- Train on HH-RLHF dataset
- Save checkpoints to `./llama3-8b-dpo-qlora`

### 2. Switch to UltraFeedback Dataset

Edit `train_dpo_qlora.py` line 18:
```python
DATASET_NAME = "openbmb/UltraFeedback"
```

Or use the config file:
```python
# In config.py, change:
ACTIVE_DATASET = "ultrafeedback"
```

### 3. Run Inference

After training completes:
```bash
python inference.py
```

## Project Structure

```
.
├── train_dpo_qlora.py    # Main training script
├── inference.py          # Inference and testing script
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

### Key Hyperparameters

**LoRA Settings** (in `config.py`):
- `r`: 16 (LoRA rank - increase to 32/64 for better quality)
- `lora_alpha`: 32 (typically 2x rank)
- `lora_dropout`: 0.05

**DPO Settings**:
- `beta`: 0.1 (DPO temperature - higher = stronger preference learning)
- `learning_rate`: 5e-5
- `batch_size`: 2 (per device)
- `gradient_accumulation_steps`: 4 (effective batch size = 8)

**Memory Optimization**:
- 4-bit quantization (NF4)
- Gradient checkpointing enabled
- BF16 mixed precision training

### Adjusting for Your Hardware

**If you get OOM (Out of Memory) errors:**
```python
# In train_dpo_qlora.py, reduce batch size:
per_device_train_batch_size=1  # instead of 2

# Or increase gradient accumulation:
gradient_accumulation_steps=8  # instead of 4
```

**If training is too slow:**
```python
# Increase batch size if you have VRAM:
per_device_train_batch_size=4

# Or reduce gradient accumulation:
gradient_accumulation_steps=2
```

## Understanding DPO

DPO (Direct Preference Optimization) trains models to prefer "chosen" responses over "rejected" ones without requiring a separate reward model. 

**How it works:**
1. Takes pairs of responses: (chosen, rejected) for the same prompt
2. Trains the model to increase probability of chosen responses
3. Uses a reference model to prevent over-optimization
4. Controlled by beta parameter (strength of preference learning)

**Dataset Format:**
```python
{
    "prompt": "Human: How do I learn Python?\n\nAssistant:",
    "chosen": "Here's a structured approach: start with basics...",
    "rejected": "Just Google it."
}
```

## Datasets

### HH-RLHF (Anthropic's Helpful & Harmless)
- 160K preference pairs
- Focuses on helpfulness and harmlessness
- Well-formatted dialogue data
- Default in this project

### UltraFeedback
- 64K examples with multiple responses rated
- More diverse instruction types
- Good for general instruction following

## Training Process

1. **Initialization**: Model loads in 4-bit, LoRA adapters added
2. **Training**: DPO loss optimizes preference alignment
3. **Checkpointing**: Saves every 100 steps
4. **Evaluation**: Runs every 100 steps on eval set
5. **Completion**: Final model saved with adapters

## Monitoring Training

If using Weights & Biases:
```bash
# View your run at:
https://wandb.ai/<your-username>/llama3-dpo-qlora
```

Key metrics to watch:
- `train/loss`: Should decrease over time
- `train/rewards/chosen`: Should increase
- `train/rewards/rejected`: Should decrease
- `train/rewards/margins`: Should increase

## Inference

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = PeftModel.from_pretrained(model, "./llama3-8b-dpo-qlora")
tokenizer = AutoTokenizer.from_pretrained("./llama3-8b-dpo-qlora")

# Generate
prompt = "Human: What is machine learning?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using the Inference Script

```bash
# Automatic testing
python inference.py

# Interactive mode (built into the script)
# Just run and follow prompts
```

## Expected Results

After 1 epoch on HH-RLHF:
- Model should prefer helpful, harmless responses
- Better instruction following
- More aligned with human preferences
- Less likely to give harmful/unhelpful responses

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size to 1
- Increase gradient_accumulation_steps to 8 or 16
- Reduce max_length to 768 or 512

### "Cannot find model 'meta-llama/Meta-Llama-3-8B'"
- You need access to Llama-3 on Hugging Face
- Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B
- Accept the license agreement
- Run `huggingface-cli login` with your token

### Training is very slow
- Normal for 8B model with QLoRA
- Expect ~1-2 samples/second on a single GPU
- Full epoch may take 12-24 hours depending on hardware

### Loss not decreasing
- Check that data is loading correctly
- Try increasing learning rate to 1e-4
- Increase beta to 0.2 for stronger preference learning
- Ensure you have enough training data

## Advanced Usage

### Custom Dataset

```python
def format_custom_dataset(example):
    return {
        "prompt": example["your_prompt_field"],
        "chosen": example["your_chosen_field"],
        "rejected": example["your_rejected_field"],
    }

dataset = load_dataset("your/dataset").map(format_custom_dataset)
```

### Merge Adapters for Deployment

```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = PeftModel.from_pretrained(model, "./llama3-8b-dpo-qlora")
model = model.merge_and_unload()
model.save_pretrained("./llama3-8b-dpo-merged")
```

### Change LoRA Rank for Better Quality

```python
# In config.py or train_dpo_qlora.py:
peft_config = LoraConfig(
    r=64,  # Higher rank = better quality, more memory
    lora_alpha=128,  # 2x rank
    ...
)
```

## References

- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## License

This code is provided as-is for educational purposes. Llama-3 model usage is subject to Meta's license agreement.

## Support

For issues or questions, please check:
1. This README
2. TRL GitHub issues
3. Hugging Face forums
