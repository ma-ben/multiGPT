import torch 
import torch.nn as nn
import math
import picotron.process_group_manager as pgm

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.model_dim = embed_dim // num_heads
        self.local_num_heads = num_heads // pgm.process_group_manager.tp_world_size
        
        self.q_proj = nn.Linear(embed_dim,embed_dim)
        self.k_proj = nn.Linear(embed_dim,embed_dim)
        self.v_proj = nn.Linear(embed_dim,embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, embed_dim = x.shape # (B, T, C)
        # 计算q, k, v
        q, k, v = [fn(x) for fn in (self.q_proj, self.k_proj, self.v_proj)] # (B, T, C)
        q, k, v = [tmp.view(B, T, self.local_num_heads, self.model_dim) for tmp in (q, k, v)] # 拆开嵌入维度
        # 计算注意力分数
        q, k, v = [tmp.permute(0, 2, 1, 3).contiguous() for tmp in(q, k, v)]  # (B, num_heads, T, model_dim)
        attn_score = (q @ k.transpose(-2, -1).contiguous()) / math.sqrt(self.model_dim) # (B, num_heads, T, T)
        # 掩码
        mask = torch.tril(torch.ones(T, T, device=x.device)) # NOTE:mask必须动态创建，因为使用了变量T,NOTE:我意识到T其实不是动态的:)
        attn_score = attn_score.masked_fill(mask == 0, float('-inf')) # (B, num_heads, T, T)
        # softmax + 对 v 加权
        attention_weights = torch.softmax(attn_score, dim=-1) # (B, num_heads, T, T)
        attn_outputs = attention_weights @ v # (B, num_heads, T, model_dim)
        # 多头注意力concat回去
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).contiguous() # (B, T, num_heads, model_dim)
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
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim) # PreNorm before attention
        self.attention = Attention(embed_dim, num_heads)
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
    def __init__(self, vocab_size = 2048, block_size=512, embed_dim=512, num_heads=8 , num_layers=8):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

    def forward(self, input_ids): # x: (B, T)
        B, T = input_ids.size()
        pos = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        return logits
    
    
    
    
