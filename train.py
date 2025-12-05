import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import SmolLMForCausalLM, SmolLMConfig
import os

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        self.block_size = block_size
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print(f"Loaded text file with {len(text)} characters")
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)
        print(f"Encoded into {len(self.tokens)} tokens")
        
        # Determine number of samples
        # We need block_size + 1 tokens per sample (input + target)
        self.num_samples = len(self.tokens) // (block_size + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Taking non-overlapping chunks for simplicity, or sliding window
        # Here we do simple chunks
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1
        
        chunk = self.tokens[start_idx:end_idx]
        
        # If we run out of data (should be handled by __len__ but handling edge case)
        if len(chunk) < self.block_size + 1:
            chunk = self.tokens[:self.block_size + 1] # Fallback to beginning
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

def train(max_steps=5000, resume_from=None, run_name="run1"):
    # Hyperparameters
    BATCH_SIZE = 8
    BLOCK_SIZE = 256
    LEARNING_RATE = 3e-4
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    # Dataset
    dataset = TextDataset("input.txt", tokenizer, BLOCK_SIZE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    config = SmolLMConfig()
    model = SmolLMForCausalLM(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    start_step = 0
    
    # Resume if requested
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # If start_step is saved, use it, otherwise 0 or inferred
            start_step = checkpoint.get('step', 0)
            print(f"Resumed at step {start_step}")
        else:
            print(f"Checkpoint {resume_from} not found. Starting from scratch.")

    model.train()
    
    # Create checkpoints directory
    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
    checkpoints = []
    
    print(f"Starting training for {max_steps} steps...")
    
    step = start_step
    data_iter = iter(train_loader)
    
    while step < start_step + max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
            
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        # Logging
        if step % 10 == 0:
             print(f"Step {step} | Loss: {loss.item():.4f}")
             
        # Generate sample and save checkpoint
        if step % 50 == 0:
            print(f"\n[Step {step}] Generating sample text...")
            generate(model, tokenizer, DEVICE)
            model.train()
            
            checkpoint_path = f"checkpoints/{run_name}/checkpoint_step_{step}.pt"
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            
            # Manage checkpoints: Keep only last 3
            checkpoints.append(checkpoint_path)
            if len(checkpoints) > 3:
                oldest_ckpt = checkpoints.pop(0)
                if os.path.exists(oldest_ckpt):
                    try:
                        os.remove(oldest_ckpt)
                        print(f"Removed old checkpoint: {oldest_ckpt}")
                    except OSError as e:
                        print(f"Error deleting checkpoint {oldest_ckpt}: {e}")
            
    # Save final checkpoint for this run
    final_path = f"checkpoints/{run_name}/checkpoint_step_{step}.pt"
    print(f"Process complete. Saving checkpoint to {final_path}...")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, final_path)
    
    return final_path

def generate(model, tokenizer, device, max_new_tokens=50):
    model.eval()
    context = torch.tensor([[tokenizer.bos_token_id or 1]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
            
    decoded = tokenizer.decode(context[0].tolist())
    print(f"Generated: {decoded}")
    print("-" * 50)

if __name__ == "__main__":
    # Stage 1: Train for 5000 steps
    print("=== STAGE 1: Training for 5000 steps ===")
    ckpt_path = train(max_steps=5000, run_name="stage1")
    
    # Stage 2: Load and train for 50 more steps
    print("\n=== STAGE 2: Resuming for 50 more steps ===")
    train(max_steps=50, resume_from=ckpt_path, run_name="stage2")
