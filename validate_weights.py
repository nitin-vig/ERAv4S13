import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import SmolLMForCausalLM, SmolLMConfig
import numpy as np

def validate_model():
    print("Downloading official model from HuggingFace...")
    model_id = "HuggingFaceTB/SmolLM-135M"
    
    # Download official model
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Initializing custom SmolLM implementation...")
    config = SmolLMConfig()
    custom_model = SmolLMForCausalLM(config)
    
    print("Copying weights from official model to custom model...")
    # Get state dicts
    hf_sd = hf_model.state_dict()
    custom_sd = custom_model.state_dict()
    
    # Map keys if necessary, but since we followed Llama architecture, 
    # many keys should match 1:1 if the hierarchy is the same.
    # HF Model structure: model.layers.0...
    # Custom Model structure: model.layers.0...
    
    # Let's count matches
    matched = 0
    skipped = 0
    
    for key in custom_sd.keys():
        if key in hf_sd:
            # Check shapes
            if custom_sd[key].shape == hf_sd[key].shape:
                with torch.no_grad():
                    custom_sd[key].copy_(hf_sd[key])
                matched += 1
            else:
                print(f"Shape mismatch for {key}: Custom {custom_sd[key].shape} != HF {hf_sd[key].shape}")
                skipped += 1
        else:
            # Sometimes bias terms or rotary embeddings buffers might mismatch
            if "rotary_emb" not in key: # Rotary buffers are often generated on the fly or named differently
                print(f"Key missing in official model: {key}")
                skipped += 1
            
    print(f"Weights transfer complete. Matched: {matched}, Skipped/Missing: {skipped}")
    
    # Validation
    print("\nRunning Validation...")
    custom_model.eval()
    hf_model.eval()
    
    input_text = "Hello, how are you? "
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        print("Running forward pass on official model...")
        hf_outputs = hf_model(**inputs).logits
        
        print("Running forward pass on custom model...")
        custom_outputs = custom_model(inputs["input_ids"])
        
    # Compare
    diff = (hf_outputs - custom_outputs).abs().max().item()
    print(f"\nMax difference between logits: {diff:.6f}")
    
    if diff < 1e-4:
        print("SUCCESS! The custom implementation matches the official model weights.")
    else:
        print("WARNING: Difference is larger than expected. There might be subtle implementation differences (e.g. RoPE calculation, LayerNorm eps, standard deviation in initialization).")

if __name__ == "__main__":
    validate_model()
