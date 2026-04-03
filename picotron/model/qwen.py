import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import picotron.process_group_manager as pgm


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def repeat_kv(x, num_repeat):
    if num_repeat == 1:
        return x
    return x.repeat_interleave(num_repeat, dim=1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_key_value_heads,
        dropout=0.0,
        rope_theta=1_000_000.0,
        attention_backend="eager",
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0
        assert num_heads % pgm.process_group_manager.tp_world_size == 0
        assert num_key_value_heads % pgm.process_group_manager.tp_world_size == 0
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires an even head dimension."
        self.local_num_heads = num_heads // pgm.process_group_manager.tp_world_size
        self.local_num_key_value_heads = num_key_value_heads // pgm.process_group_manager.tp_world_size
        assert self.local_num_heads % self.local_num_key_value_heads == 0
        self.num_key_value_groups = self.local_num_heads // self.local_num_key_value_heads
        self.dropout = float(dropout)
        self.rope_theta = float(rope_theta)
        self.attention_backend = attention_backend

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def _rope_cos_sin(self, position_ids, device, dtype):
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        freqs = torch.einsum("bt,d->btd", position_ids.to(torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).to(dtype=dtype)
        sin = emb.sin().unsqueeze(1).to(dtype=dtype)
        return cos, sin

    def _apply_rope(self, q, k, position_ids):
        cos, sin = self._rope_cos_sin(position_ids, q.device, q.dtype)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k

    def forward(self, x, position_ids=None):
        B, T, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        q = self.q_proj(x).view(B, T, self.local_num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = self.k_proj(x).view(B, T, self.local_num_key_value_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = self.v_proj(x).view(B, T, self.local_num_key_value_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        q, k = self._apply_rope(q, k, position_ids)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        if self.attention_backend == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
            attn_outputs = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_score = (q @ k.transpose(-2, -1).contiguous()) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device))
            attn_score = attn_score.masked_fill(mask == 0, float("-inf"))
            attention_weights = self.attn_dropout(torch.softmax(attn_score, dim=-1))
            attn_outputs = attention_weights @ v

        attn_outputs = attn_outputs.permute(0, 2, 1, 3).contiguous()
        attn_outputs = attn_outputs.view(B, T, self.local_num_heads * self.head_dim)
        return self.resid_dropout(self.out_proj(attn_outputs))


class MLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_size, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_key_value_heads,
        intermediate_size,
        dropout=0.0,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        attention_backend="eager",
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.attention = Attention(
            hidden_dim,
            num_heads,
            num_key_value_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            attention_backend=attention_backend,
        )
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.mlp = MLP(hidden_dim, intermediate_size, dropout=dropout)

    def forward(self, x, position_ids=None):
        x = x + self.attention(self.input_layernorm(x), position_ids=position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen(nn.Module):
    def __init__(
        self,
        vocab_size=151936,
        block_size=32768,
        embed_dim=1024,
        num_heads=16,
        num_key_value_heads=8,
        num_layers=28,
        intermediate_size=3072,
        dropout=0.0,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        attention_backend="eager",
        activation_checkpointing=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_dim=embed_dim,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    rms_norm_eps=rms_norm_eps,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.activation_checkpointing = activation_checkpointing
        self.block_size = block_size
        self.wte.weight = self.lm_head.weight

    def embed_tokens(self, input_ids, position_ids):
        del position_ids
        return self.drop(self.wte(input_ids))

    def forward(self, input_ids, position_ids=None):
        B, T = input_ids.size()
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed_tokens(input_ids, position_ids)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = activation_checkpoint(block, x, position_ids, use_reentrant=False)
            else:
                x = block(x, position_ids=position_ids)
        return self.lm_head(self.ln_f(x))
