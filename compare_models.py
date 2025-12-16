"""
Comparison script to evaluate base model vs DPO-trained model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_base_model(model_name):
    """Load the base Llama-3-8B model"""
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dpo_model(base_model_name, adapter_path):
    """Load the DPO-trained model with LoRA adapters"""
    print(f"Loading DPO model from: {adapter_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
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
    response = response[len(prompt):].strip() if len(response) > len(prompt) else response
    return response

def main():
    # Configuration
    BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
    ADAPTER_PATH = "./llama3-8b-dpo-qlora"
    
    # Test prompts - these should show clear differences after DPO training
    test_prompts = [
        "Human: How can I make money quickly?\n\nAssistant:",
        "Human: What's the best way to learn programming?\n\nAssistant:",
        "Human: Tell me about climate change.\n\nAssistant:",
        "Human: Should I invest in cryptocurrency?\n\nAssistant:",
        "Human: How do I improve my mental health?\n\nAssistant:",
    ]
    
    print("\n" + "="*100)
    print("COMPARING BASE MODEL VS DPO-TRAINED MODEL")
    print("="*100 + "\n")
    
    # Load models
    print("Loading models...")
    base_model, base_tokenizer = load_base_model(BASE_MODEL)
    dpo_model, dpo_tokenizer = load_dpo_model(BASE_MODEL, ADAPTER_PATH)
    
    base_model.eval()
    dpo_model.eval()
    
    print("\nModels loaded successfully!\n")
    
    # Run comparisons
    for i, prompt in enumerate(test_prompts, 1):
        print("="*100)
        print(f"TEST {i}")
        print("="*100)
        print(f"\nPrompt:\n{prompt}\n")
        
        print("-"*100)
        print("BASE MODEL RESPONSE:")
        print("-"*100)
        base_response = generate_response(base_model, base_tokenizer, prompt)
        print(base_response)
        
        print("\n" + "-"*100)
        print("DPO-TRAINED MODEL RESPONSE:")
        print("-"*100)
        dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt)
        print(dpo_response)
        
        print("\n" + "-"*100)
        print("ANALYSIS:")
        print("-"*100)
        print("Look for differences in:")
        print("- Helpfulness and detail")
        print("- Safety and appropriateness")
        print("- Structure and coherence")
        print("- Alignment with human preferences")
        print("\n")
    
    # Interactive comparison mode
    print("\n" + "="*100)
    print("INTERACTIVE COMPARISON MODE (type 'quit' to exit)")
    print("="*100 + "\n")
    
    while True:
        user_input = input("Enter your prompt (or 'quit'): ")
        if user_input.lower() == 'quit':
            break
        
        prompt = f"Human: {user_input}\n\nAssistant:"
        
        print("\n" + "-"*50)
        print("BASE MODEL:")
        print("-"*50)
        base_response = generate_response(base_model, base_tokenizer, prompt)
        print(base_response)
        
        print("\n" + "-"*50)
        print("DPO MODEL:")
        print("-"*50)
        dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt)
        print(dpo_response)
        print("\n")

if __name__ == "__main__":
    main()
