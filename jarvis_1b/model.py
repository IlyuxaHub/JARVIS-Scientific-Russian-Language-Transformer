"""JARVIS Scientific: pre-norm, RoPE, SwiGLU, causal PyTorch SDPA."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig


def parameter_count(c: ModelConfig):
    return c.vocab_size * c.d_model + c.n_layers * (4 * c.d_model**2 + 3 * c.d_model * c.d_ff + 2 * c.d_model) + c.d_model


def parameter_breakdown(c: ModelConfig):
    """Unique trainable parameters; the tied LM head adds no second matrix."""
    embedding = c.vocab_size * c.d_model
    attention = 4 * c.d_model**2
    feed_forward = 3 * c.d_model * c.d_ff
    block_norms = 2 * c.d_model
    per_block = attention + feed_forward + block_norms
    result = {
        'token_embedding': embedding,
        'attention_per_block': attention,
        'feed_forward_per_block': feed_forward,
        'norms_per_block': block_norms,
        'transformer_blocks': c.n_layers * per_block,
        'final_norm': c.d_model,
        'lm_head_additional': 0,
        'tied_embedding_parameters_avoided': embedding,
    }
    result['total'] = embedding + result['transformer_blocks'] + c.d_model
    return result


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.head_dim = c.n_heads, c.d_model // c.n_heads
        self.dropout = c.dropout
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (u.transpose(1, 2) for u in (q, k, v))
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(x.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_up = nn.Linear(c.d_model, 2 * c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.ffn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.attn, self.ffn = Attention(c), SwiGLU(c)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin):
        x = x + self.dropout(self.attn(self.attn_norm(x), cos, sin))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class ScientificLM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        c.validate()
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        # F.linear with the embedding matrix ties weights without duplicate registration/init.
        head_dim = c.d_model // c.n_heads
        freq = 1.0 / c.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.outer(torch.arange(c.context).float(), freq)
        self.register_buffer('rope_cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('rope_sin', angles.sin()[None, None], persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=c.init_std)
        # Scale only the two residual output projections, not every matrix.
        for block in self.blocks:
            for weight in (block.attn.out.weight, block.ffn.down.weight):
                nn.init.normal_(weight, std=c.init_std / math.sqrt(2 * c.n_layers))

    def _loss_chunk(self, h, y):
        return F.cross_entropy(F.linear(h, self.embedding.weight).float(), y, reduction='sum')

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.context or tokens.shape[1] < 1:
            raise ValueError('Expected [batch, sequence] within configured context')
        t = tokens.shape[1]
        cos, sin = self.rope_cos[:, :, :t], self.rope_sin[:, :, :t]
        x = self.embedding(tokens)
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.norm(x)
        if targets is None:
            return F.linear(x, self.embedding.weight), None
        if targets.shape != tokens.shape:
            raise ValueError('Targets must have the same shape as inputs and already be shifted by one')
        h, y = x.reshape(-1, x.shape[-1]), targets.reshape(-1)
        # Recompute the small LM-head chunks in backward. Splitting without checkpointing
        # would retain all softmax activations and would not reduce peak memory enough.
        loss = x.new_zeros((), dtype=torch.float32)
        chunk = self.config.loss_chunk_tokens
        for offset in range(0, len(y), chunk):
            args = (h[offset:offset + chunk], y[offset:offset + chunk])
            if self.training and torch.is_grad_enabled():
                loss = loss + checkpoint(self._loss_chunk, *args, use_reentrant=False)
            else:
                loss = loss + self._loss_chunk(*args)
        return None, loss / y.numel()
