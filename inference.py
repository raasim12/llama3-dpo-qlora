"""
Inference script for DPO-trained Llama-3-8B model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name, adapter_path):
    """Load the base model with LoRA adapters"""
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # Merge adapters for faster inference
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part (remove prompt)
    response = response[len(prompt):].strip()
    return response

def main():
    # Configuration
    BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
    ADAPTER_PATH = "./llama3-8b-dpo-qlora"
    
    # Load model
    model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "Human: What are the benefits of exercise?\n\nAssistant:",
        "Human: How can I improve my communication skills?\n\nAssistant:",
        "Human: Explain quantum computing in simple terms.\n\nAssistant:",
    ]
    
    print("\n" + "="*80)
    print("Testing DPO-trained Llama-3-8B model")
    print("="*80 + "\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}")
        print(f"{'='*80}")
        print(f"Prompt:\n{prompt}\n")
        
        response = generate_response(model, tokenizer, prompt)
        print(f"Response:\n{response}\n")
    
    # Interactive mode
    print("\n" + "="*80)
    print("Interactive mode (type 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        user_input = input("Enter your prompt (or 'quit'): ")
        if user_input.lower() == 'quit':
            break
        
        prompt = f"Human: {user_input}\n\nAssistant:"
        response = generate_response(model, tokenizer, prompt)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()
