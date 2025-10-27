"""
完整的 Transformer 模型实现
包括 Encoder-Decoder 架构
"""

import torch
import torch.nn as nn
import math
from .modules import (
    EncoderLayer, DecoderLayer, PositionalEncoding
)


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    """
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_layers=6,
                 h=8,
                 d_ff=2048,
                 max_len=5000,
                 dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, src, tgt, pad_idx=0):
        """
        生成 mask
        
        Args:
            src: source tensor of shape (batch_size, src_len)
            tgt: target tensor of shape (batch_size, tgt_len)
            pad_idx: padding index
            
        Returns:
            src_mask: (batch_size, 1, 1, src_len)
            tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        """
        # Source padding mask
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Target padding mask
        tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
        
        # Target look-ahead mask (causal mask)
        tgt_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)
        
        # Combine tgt masks
        tgt_mask = tgt_padding_mask & (~look_ahead_mask)
        
        return src_mask, tgt_mask
    
    def encode(self, src, src_mask):
        """
        Encoder forward pass
        
        Args:
            src: (batch_size, src_len)
            src_mask: (batch_size, 1, 1, src_len)
            
        Returns:
            encoder_output: (batch_size, src_len, d_model)
        """
        # Embedding + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(src_emb)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """
        Decoder forward pass
        
        Args:
            tgt: (batch_size, tgt_len)
            encoder_output: (batch_size, src_len, d_model)
            src_mask: (batch_size, 1, 1, src_len)
            tgt_mask: (batch_size, 1, tgt_len, tgt_len)
            
        Returns:
            decoder_output: (batch_size, tgt_len, d_model)
        """
        # Embedding + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(tgt_emb)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, pad_idx=0):
        """
        Forward pass
        
        Args:
            src: source tensor of shape (batch_size, src_len)
            tgt: target tensor of shape (batch_size, tgt_len)
            pad_idx: padding index
            
        Returns:
            output: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Generate masks
        src_mask, tgt_mask = self.generate_mask(src, tgt, pad_idx)
        
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output

