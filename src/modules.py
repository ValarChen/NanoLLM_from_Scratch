"""
Transformer 核心模块实现
包括：Scaled Dot-Product Attention, Multi-Head Attention, FFN, Positional Encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    计算缩放点积注意力
    
    Args:
        q (Tensor): 查询, shape (..., seq_len_q, d_k)
        k (Tensor): 键, shape (..., seq_len_k, d_k)
        v (Tensor): 值, shape (..., seq_len_k, d_v)
        mask (Tensor, optional): 掩码, shape (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
        output (Tensor): 加权后的值, shape (..., seq_len_q, d_v)
        attn (Tensor): 注意力权重, shape (..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    
    # 1. Q @ K.T / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Mask (optional)
    if mask is not None:
        # 将mask中为0的位置填充为一个极小的负数, softmax后这些位置的概率会接近0
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. Multiply by V
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影
        # q, k, v: (batch_size, seq_len, d_model)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. 拆分成h个头
        # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力
        # context shape: (batch_size, h, seq_len_q, d_k)
        # attn_weights shape: (batch_size, h, seq_len_q, seq_len_k)
        context, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 4. 拼接头
        # (batch_size, h, seq_len_q, d_k) -> (batch_size, seq_len_q, h * d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 输出线性投影
        output = self.w_o(context)
        
        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络 (FFN)
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    Transformer Encoder 层
    包含：
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Position-wise Feed-Forward Network
    4. Add & Norm
    """
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 层
    包含：
    1. Masked Multi-Head Self-Attention
    2. Add & Norm
    3. Multi-Head Cross-Attention (Query来自Decoder，Key和Value来自Encoder)
    4. Add & Norm
    5. Position-wise Feed-Forward Network
    6. Add & Norm
    """
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and layer norm
        # Query来自decoder，Key和Value来自encoder
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

