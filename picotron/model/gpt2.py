import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import math
import picotron.process_group_manager as pgm

class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, attention_backend="eager"):
        super().__init__()

        assert hidden_dim % num_heads == 0
        assert num_heads % pgm.process_group_manager.tp_world_size == 0
        self.model_dim = hidden_dim // num_heads
        self.local_num_heads = num_heads // pgm.process_group_manager.tp_world_size
        self.dropout = float(dropout)
        # attention_backend 允许我们在“教学版 eager attention”和
        # “PyTorch 官方 SDPA 后端”之间切换。
        # 这样可以用最少代码把 L6 compiler/runtime 路线里的一个关键技术点体现出来：
        # 同样的数学表达，走不同后端会直接影响 kernel 选择、显存读写和性能上限。
        self.attention_backend = attention_backend
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, H = x.shape
        # 先做线性投影，再把通道维拆成多头。
        # 这里保留标准形状 `(B, H, T, D)`，是因为：
        # 1. eager attention 更容易写清楚；
        # 2. SDPA 也正好接受这种布局。
        q, k, v = [fn(x) for fn in (self.q_proj, self.k_proj, self.v_proj)]
        q, k, v = [tmp.view(B, T, self.local_num_heads, self.model_dim) for tmp in (q, k, v)]
        q, k, v = [tmp.permute(0, 2, 1, 3).contiguous() for tmp in (q, k, v)]

        if self.attention_backend == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
            # SDPA 会把 mask / softmax / dropout / matmul 打包到官方后端里。
            # 在 CUDA 上这通常会落到更优化的实现，是当前这份代码里最轻量的“后端升级”。
            attn_outputs = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # eager 路径保留最原始的数学表达，便于从 L0/L1 角度理解 attention。
            attn_score = (q @ k.transpose(-2, -1).contiguous()) / math.sqrt(self.model_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device))
            attn_score = attn_score.masked_fill(mask == 0, float("-inf"))
            attention_weights = self.attn_dropout(torch.softmax(attn_score, dim=-1))
            attn_outputs = attention_weights @ v

        # 多头注意力concat回去
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).contiguous()
        attn_outputs = attn_outputs.view(B, T, self.local_num_heads * self.model_dim)
        # 最后一层映射
        return self.resid_dropout(self.out_proj(attn_outputs))


class MLP(nn.Module):
    def __init__(self, H, dropout=0.1):
        super().__init__()
        self.up_proj = nn.Linear(H, 4 * H, bias=False)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(4*H, H, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.down_proj(self.gelu(self.up_proj(x))))


class Block(nn.Module):
    def __init__(self, H, num_heads, dropout=0.1, attention_backend="eager"):
        super().__init__()
        self.ln1 = nn.LayerNorm(H) # PreNorm before attention
        self.attention = Attention(H, num_heads, dropout=dropout, attention_backend=attention_backend)
        self.ln2 = nn.LayerNorm(H) # PreNorm before MLP
        self.mlp = MLP(H, dropout=dropout)

    def forward(self, x, position_ids=None):
        # Attention段
        x = x + self.attention(self.ln1(x)) # 注意力前 LayerNorm，结果 residual
        # Feedforward段
        x = x + self.mlp(self.ln2(x)) # MLP 前 LayerNorm，结果 residual
        return x


# 模型定义：embedding → attention → linear output 
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size=2048,
        block_size=512,
        embed_dim=None,
        H=None,
        num_heads=8,
        num_layers=8,
        dropout=0.1,
        attention_backend="eager",
        activation_checkpointing=False,
    ):
        super().__init__()
        if embed_dim is None and H is None:
            hidden_dim = 512
        elif embed_dim is None:
            hidden_dim = H
        elif H is None or H == embed_dim:
            hidden_dim = embed_dim
        else:
            raise ValueError("GPT received conflicting values for embed_dim and H.")

        self.embed_dim = hidden_dim
        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.wpe = nn.Embedding(block_size, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(hidden_dim, num_heads, dropout=dropout, attention_backend=attention_backend) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        # activation_checkpointing 是“小规模也值得保留”的一个关键技术点：
        # 它把前向中间激活换成“反向时重算”，用更少显存换更多算力。
        # 对教学来说，它正好能把 L0 训练循环、L1 autograd 和 L8 显存管理连起来。
        self.activation_checkpointing = activation_checkpointing
        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

    def embed_tokens(self, input_ids, position_ids):
        return self.drop(self.wte(input_ids) + self.wpe(position_ids))

    def forward(self, input_ids, position_ids=None): # x: (B, T)
        B, T = input_ids.size()
        # position_ids 显式透传出来，是为了和 PP 路径保持一致。
        # 首段 stage 可以自己构造 position_ids，中间 stage 则只处理 hidden states。
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed_tokens(input_ids, position_ids)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                # use_reentrant=False 是 PyTorch 当前更推荐的 checkpoint 路径，
                # 它对 autograd 行为更直观，也更适合和现代特性一起使用。
                x = activation_checkpoint(block, x, position_ids, use_reentrant=False)
            else:
                x = block(x, position_ids=position_ids)
        logits = self.lm_head(self.ln_f(x))
        return logits
