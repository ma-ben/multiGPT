import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import math
import picotron.process_group_manager as pgm

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_backend="eager"):
        super().__init__()

        assert embed_dim % num_heads == 0
        assert num_heads % pgm.process_group_manager.tp_world_size == 0
        self.model_dim = embed_dim // num_heads
        self.local_num_heads = num_heads // pgm.process_group_manager.tp_world_size
        # attention_backend 允许我们在“教学版 eager attention”和
        # “PyTorch 官方 SDPA 后端”之间切换。
        # 这样可以用最少代码把 L6 compiler/runtime 路线里的一个关键技术点体现出来：
        # 同样的数学表达，走不同后端会直接影响 kernel 选择、显存读写和性能上限。
        self.attention_backend = attention_backend
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, embed_dim = x.shape
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
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            # eager 路径保留最原始的数学表达，便于从 L0/L1 角度理解 attention。
            attn_score = (q @ k.transpose(-2, -1).contiguous()) / math.sqrt(self.model_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device))
            attn_score = attn_score.masked_fill(mask == 0, float("-inf"))
            attention_weights = torch.softmax(attn_score, dim=-1)
            attn_outputs = attention_weights @ v

        # 多头注意力concat回去
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).contiguous()
        attn_outputs = attn_outputs.view(B, T, self.local_num_heads * self.model_dim)
        # 最后一层映射
        return self.out_proj(attn_outputs)


class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, 4 * embed_dim, bias=False)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(4*embed_dim, embed_dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(self.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, attention_backend="eager"):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim) # PreNorm before attention
        self.attention = Attention(embed_dim, num_heads, attention_backend=attention_backend)
        self.ln2 = nn.LayerNorm(embed_dim) # PreNorm before MLP
        self.mlp = MLP(embed_dim)

    def forward(self, x):
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
        embed_dim=512,
        num_heads=8,
        num_layers=8,
        attention_backend="eager",
        activation_checkpointing=False,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, attention_backend=attention_backend) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # activation_checkpointing 是“小规模也值得保留”的一个关键技术点：
        # 它把前向中间激活换成“反向时重算”，用更少显存换更多算力。
        # 对教学来说，它正好能把 L0 训练循环、L1 autograd 和 L8 显存管理连起来。
        self.activation_checkpointing = activation_checkpointing
        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

    def forward(self, input_ids, position_ids=None): # x: (B, T)
        B, T = input_ids.size()
        # position_ids 显式透传出来，是为了和 PP 路径保持一致。
        # 首段 stage 可以自己构造 position_ids，中间 stage 则只处理 hidden states。
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                # use_reentrant=False 是 PyTorch 当前更推荐的 checkpoint 路径，
                # 它对 autograd 行为更直观，也更适合和现代特性一起使用。
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.lm_head(self.ln_f(x))
        return logits
    
    
    
    
