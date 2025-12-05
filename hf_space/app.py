import torch
import gradio as gr
from transformers import AutoTokenizer
from model import SmolLMForCausalLM, SmolLMConfig
import os

# 1. Configuration constants
MODEL_CHECKPOINT = "model.pt" # Expects the model weights to be in this file
TOKENIZER_ID = "HuggingFaceTB/SmolLM-135M" # Using the standard tokenizer
DEVICE = "cpu" # HF Spaces free tier usually is CPU. Change to 'cuda' if GPU is available.

# 2. Load Model and Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

print("Initializing model...")
config = SmolLMConfig()
model = SmolLMForCausalLM(config)

# 3. Load Weights
if os.path.exists(MODEL_CHECKPOINT):
    print(f"Loading weights from {MODEL_CHECKPOINT}...")
    try:
        # Map location to CPU to be safe
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))
        
        # Check if it's a full checkpoint (dict with 'model_state_dict') or just weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Handle any prefix issues (e.g. if saved from compiled model with '_orig_mod.')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Running with initialized (random) weights for demonstration.")
else:
    print(f"Warning: {MODEL_CHECKPOINT} not found! Running with random weights.")

model.to(DEVICE)
model.eval()

# 4. Generation Function
def generate_text(prompt, max_new_tokens, temperature, top_k):
    if not prompt:
        return "Please enter a prompt."
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Text Generation Loop
    # We implement a simple loop similar to the training script's generate function
    # but added temperature and top-k sampling for better variety in the demo.
    
    curr_input_ids = input_ids
    
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            # Get logits
            logits = model(curr_input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Apply Temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            else:
                # Greedy decoding if temperature is 0 (or very close)
                # Just take argmax, but for code simplicity we'll let multinomial handle it with very high conf or Argmax
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                curr_input_ids = torch.cat([curr_input_ids, next_token_id], dim=1)
                continue

            # Apply Top-K
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Sample
            next_token_id = torch.multinomial(probs, num_samples=1)
            curr_input_ids = torch.cat([curr_input_ids, next_token_id], dim=1)
            
            # optional: stop if EOS token is generated (if we had one defined and training used it)
            # if next_token_id == tokenizer.eos_token_id:
            #     break

    output_text = tokenizer.decode(curr_input_ids[0].tolist(), skip_special_tokens=True)
    return output_text

# 5. Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# SmolLM-135M Implementation Demo")
    gr.Markdown("This is a demo of the 135M parameter transformer model trained from scratch.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="Once upon a time...", lines=3)
            with gr.Row():
                max_tokens = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-K")
            generate_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(label="Generated Text", lines=10)
            
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens, temperature, top_k],
        outputs=output
    )
    
    gr.Markdown("### Note on inputs")
    gr.Markdown("Because this model is small (135M) and trained on a specific dataset, it may not follow instructions like ChatGPT. It is best at completing text/stories.")

if __name__ == "__main__":
    demo.launch()
