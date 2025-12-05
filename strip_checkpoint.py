import torch
import sys
import os

def strip_checkpoint(input_path, output_path):
    print(f"Loading checkpoint from {input_path}...")
    # Load on CPU to avoid memory issues
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Extract only model weights
    if 'model_state_dict' in checkpoint:
        print("Found 'model_state_dict', extracting weights...")
        weights = checkpoint['model_state_dict']
    else:
        print("Checkpoint format unknown or already stripped. Saving as is...")
        weights = checkpoint

    # Removing potential compile prefixes if they exist (cleaner for inference)
    clean_weights = {}
    for k, v in weights.items():
        new_k = k.replace("_orig_mod.", "")
        clean_weights[new_k] = v

    print(f"Saving stripped weights to {output_path}...")
    torch.save(clean_weights, output_path)
    
    # Check sizes
    orig_size = os.path.getsize(input_path) / (1024*1024)
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"Done! Reduced size from {orig_size:.2f} MB to {new_size:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python strip_checkpoint.py <input_checkpoint> <output_model_path>")
        print("Example: python strip_checkpoint.py checkpoints/stage1/checkpoint_step_5000.pt hf_space/model.pt")
    else:
        strip_checkpoint(sys.argv[1], sys.argv[2])
