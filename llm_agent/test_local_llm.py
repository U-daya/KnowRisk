import sys
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_inference():
    print("=" * 60)
    print("KnowRisk AMD GPU (ROCm) Direct LLM Test")
    print("=" * 60)
    
    # 1. Verify CUDA / ROCm availability in PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA/ROCm available: {cuda_available}")
    if not cuda_available:
        print("❌ Error: PyTorch does not detect ROCm GPU!")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    print(f"Device Count: {device_count}")
    for i in range(device_count):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading model: {model_id} ...")
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Model loaded in {time.time() - t0:.2f} seconds.")
    print(f"Model device: {model.device}")
    
    # 2. Run a test prompt
    prompt = "Explain in one sentence why single-source semiconductor suppliers represent a risk."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("\nRunning inference...")
    t1 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=True
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    duration = time.time() - t1
    
    print("\n" + "=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Answer: {response}")
    print(f"Inference completed in {duration:.2f} seconds.")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()
