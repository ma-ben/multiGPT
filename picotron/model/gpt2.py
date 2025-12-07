import torch 
import torch.nn as nn
import math



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.d_model = embed_dim // num_heads
        self.num_heads = num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, embed_dim = x.shape # (B, T, C)
        # 1. 计算q, k, v
        qkv = self.qkv_proj(x) # (B, T,3*C)
        # 2. 拆分q, k, v
        qkv = qkv.view(B, T, self.num_heads, 3*self.d_model)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, num_heads, d_model)*3
        # 3. 计算注意力分数
        q, k, v = [x.permute(0, 2, 1, 3) for x in(q, k, v)]  # (B, num_heads, T, d_model)
        attn_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model) # (B, num_heads, T, T)
        # 4. 掩码
        mask = torch.tril(torch.ones(T, T, device=x.device)) # NOTE:mask必须动态创建，因为使用了变量T,NOTE:我意识到T其实不是动态的:)
        attn_score = attn_score.masked_fill(mask == 0, float('-inf')) # (B, num_heads, T, T)
        # 5. softmax + 对 v 加权
        attention_weights = torch.softmax(attn_score, dim=-1) # (B, num_heads, T, T)
        attn_outputs = attention_weights @ v # (B, num_heads, T, d_model)
        # 6. 多头注意力concat回去
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).contiguous() # (B, T, num_heads, d_model)
        attn_outputs = attn_outputs.view(B, T, embed_dim)
        # 7. 最后一层映射
        return self.out_proj(attn_outputs)


class MLP(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*embed_dim, embed_dim, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim) # PreNorm before attention
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim) # PreNorm before MLP
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        # Attention段
        x = x + self.attn(self.ln1(x)) # 注意力前 LayerNorm，结果 residual
        # Feedforward段
        x = x + self.mlp(self.ln2(x)) # MLP 前 LayerNorm，结果 residual
        return x


# 模型定义：embedding → attention → linear output 
class GPT(nn.Module):
    def __init__(self, vocab_size = 2048, block_size=512, embed_dim=512, num_heads=8 , num_layers=8):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(block_size, embed_dim)
        self.h = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

    def forward(self, input_ids): # x: (B, T)B, T = idx.size()
        B, T = input_ids.size()
        pos = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        return logits


# class GPT(nn.Module):
#     def __init__(self, config=None):
#         super().__init__()
#         self.gpt = GPTModel(
#             vocab_size=config.vocab_size,
#             block_size=config.block_size,
#             embed_dim=config.embed_dim,
#             num_heads=config.num_heads,
#             num_layers=config.num_layers,
#         ) if config is not None else GPTModel()
    
#     def forward(self, input_ids):
#         return self.gpt(input_ids)
    
    
    
    
