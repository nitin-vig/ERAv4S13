import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SmolLMConfig:
    """
    Configuration class for SmolLM.
    This holds all the hyperparameters that define the model architecture.
    """
    vocab_size: int = 49152  # Size of vocabulary (number of unique tokens)
    hidden_size: int = 576   # Dimension of the embedding vectors
    intermediate_size: int = 1536  # Dimension of the inner layer in the MLP
    num_hidden_layers: int = 30    # Number of Transformer blocks (depth)
    num_attention_heads: int = 9   # Number of heads for the query
    num_key_value_heads: int = 3   # Number of heads for keys and values (GQA)
    hidden_act: str = "silu"       # Activation function
    max_position_embeddings: int = 2048 # Maximum sequence length
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    tie_word_embeddings: bool = True # Share weights between input embedding and output layer
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        # Calculate dimension per head
        self.head_dim = self.hidden_size // self.num_attention_heads
        # Calculate how many Query heads share one Key/Value head (Grouped Query Attention)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    A simpler version of LayerNorm that re-scales inputs based on their RMS.
    It stabilizes training and is used in Llama-based models instead of standard LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate RMS: sqrt(mean(x^2) + epsilon)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Normalize and then scale by a learnable parameter
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embeddings (RoPE) to queries and keys.
    RoPE rotates the query and key vectors to inject relative positional information.
    """
    # q, k: [bs, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim] or projected
    
    # Rotate function: [-x2, x1]
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
    
    # Apply rotation: (x * cos) + (rotate_90(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(nn.Module):
    """
    Pre-computes the cosine and sine values for RoPE.
    These are fixed values based on position indices, used to modulate Q and K.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Calculate inverse frequencies for the rotations
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from standard position embeddings, we concat freq to itself to cover both halves
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.float32), persistent=False)


    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class LlamaMLP(nn.Module):
    """
    Feed-Forward Network (FFN) utilizing the SwiGLU activation.
    Structure:
    x -> GateProj -> SiLU \
                           -> Multiply -> DownProj -> output
    x -> UpProj_________/
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU: (SiLU(Gate(x)) * Up(x)) -> Down(x)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class LlamaAttention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA).
    GQA uses fewer Key/Value heads than Query heads to save memory and KV cache during inference.
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, x, position_ids=None, attention_mask=None):
        bsz, q_len, _ = x.size()

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for multi-head attention
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Embeddings
        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Handle GQA (Grouped Query Attention)
        # If we have fewer KV heads than Q heads, we repeat K and V to match Q's dimensions
        if self.num_key_value_groups > 1:
            k = k[:, :, None, :, :].expand(bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim).reshape(bsz, self.num_heads, q_len, self.head_dim)
            v = v[:, :, None, :, :].expand(bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim).reshape(bsz, self.num_heads, q_len, self.head_dim)

        # 5. Calculate Attention Scores
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask

        # 6. Apply Softmax and Compute Weighted Sum
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # 7. Reshape back and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class LlamaDecoderLayer(nn.Module):
    """
    A single Transformer block.
    Consists of:
    1. Pre-Norm -> Attention -> Add Residual
    2. Pre-Norm -> MLP (Feed Forward) -> Add Residual
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_ids=None, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        # Self Attention block
        x = self.self_attn(x, position_ids=position_ids, attention_mask=attention_mask)
        x = residual + x # Residual connection
        
        residual = x
        x = self.post_attention_layernorm(x)
        # MLP block
        x = self.mlp(x)
        x = residual + x # Residual connection
        return x

class SmolLMModel(nn.Module):
    """
    Main Transformer model (the "trunk").
    Embeddings -> N x Decoder Layers -> Final Norm
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids):
        # 1. Lookup Embeddings
        x = self.embed_tokens(input_ids)
        
        seq_len = x.shape[1]
        
        # 2. Key Concept: Causal Mask
        # We want the model to predict the NEXT token, so it shouldn't see future tokens.
        # We create a mask of -inf for upper triangle to block future positions.
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        
        # 3. Pass through all Transformer Layers
        for layer in self.layers:
            x = layer(x, attention_mask=mask)
            
        x = self.norm(x)
        return x

class SmolLMForCausalLM(nn.Module):
    """
    The full Causal Language Model.
    Wraps the trunk (SmolLMModel) and adds the Language Model Head (Linear Layer) to project to accumulation logic.
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.model = SmolLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        x = self.model(input_ids)
        logits = self.lm_head(x)
        return logits

def test_model():
    config = SmolLMConfig()
    print(f"Initializing SmolLM-135M with config: {config}")
    
    model = SmolLMForCausalLM(config)
    print(f"Model keys: {model.state_dict().keys().__len__()}")
    
    # Test forward pass
    dummy_input = torch.randint(0, config.vocab_size, (1, 32)) # Batch size 1, seq len 32
    print(f"Running forward pass with input shape {dummy_input.shape}")
    
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}") # Should be [1, 32, 49152]
    
    assert logits.shape == (1, 32, config.vocab_size)
    print("Test passed!")

if __name__ == "__main__":
    test_model()
