# SmolLM-135M Training

This repository contains the implementation and training scripts for a **SmolLM-135M** language model, trained from scratch on a custom dataset.

## Model Architecture

The model is based on the **Llama architecture** with approximately **135 million parameters**. Key architectural features include:

*   **Transformer-based Causal Language Model**: Standard decoder-only transformer for next-token prediction.
*   **Grouped Query Attention (GQA)**: optimized attention mechanism where multiple Query heads share a single Key/Value head (9 query heads, 3 KV heads). This reduces memory usage for the KV cache during inference.
*   **Rotary Positional Embeddings (RoPE)**: Relative positional encoding applied to queries and keys for better handling of sequence lengths.
*   **RMSNorm**: Root Mean Square Layer Normalization used for better training stability compared to standard LayerNorm.
*   **SwiGLU Activation**: Uses the SwiGLU activation function in the MLP layers (SiLU gated linear unit) for improved performance.
*   **Tied Embeddings**: Input and output embeddings share weights to reduce parameter count.

**Configuration:**
- **Hidden Size**: 576
- **Layers**: 30
- **Heads**: 9 (Query), 3 (Key/Value)
- **Vocab Size**: 49,152
- **Context Length**: 2048

## Speed Optimizations

The training script utilizes several modern PyTorch optimizations to maximize training throughput:

1.  **Mixed Precision Training**: Uses `torch.amp.autocast` with **bfloat16** (on supported CUDA devices) or **float16** (MPS/CUDA). This reduces memory bandwidth usage and speeds up matrix multiplications while maintaining training stability.
2.  **Flash Attention**: The model uses `F.scaled_dot_product_attention` which automatically selects optimized attention kernels (like FlashAttentionv2) when available, significantly reducing memory usage and computation time for attention layers.
3.  **Torch Compile**: On CUDA devices, the model is compiled using `torch.compile(model)`, which optimizes the computational graph, fuses kernels, and reduces Python overhead.
4.  **TF32 Support**: Enabled TensorFloat-32 (TF32) on Ampere+ GPUs for faster FP32 matrix multiplications.

## Training Logs

The full training log, including loss metrics and text generation samples at various checkpoints, can be found in:

**[Training_log.txt](Training_log.txt)**

**Highlights:**
*   **Initial Loss**: ~66.0
*   **Final Loss**: ~3.5
*   **Samples**: The log contains generated samples every 50 steps, showing the model's progression from random noise to coherent text.

## Usage

To train the model:

```bash
python train.py
```

This runs a 2-stage training process:
1.  **Stage 1**: Train from scratch for 5000 steps.
2.  **Stage 2**: Resume and fine-tune for an additional 50 steps.
