import numpy as np
import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.scale = d_model ** -0.5

        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = x * self.scale
        return x

class PositionalEmbeddingLearned(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pe = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pe
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v)
        
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*r_mlp, d_model)
        )
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x):
        res1 = self.mha(self.ln1(x))
        res1 = self.dropout1(res1)
        x = x + res1
        
        res2 = self.mlp(self.ln2(x))
        res2 = self.dropout2(res2)
        x = x + res2
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, dropout=0.2):
        super().__init__()

        assert img_size % patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.ln_f = nn.LayerNorm(d_model)

        self.n_patches = (self.img_size * self.img_size) // (self.patch_size * self.patch_size)
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEmbeddingLearned(self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(
                self.d_model, 
                self.n_heads, 
                r_mlp=4,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.n_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.ln_f(x[:,0])
        x = self.classifier(x)
        
        return x
