"""
数据加载和预处理模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TranslationDataset(Dataset):
    """
    机器翻译数据集
    
    用于加载和处理翻译数据（如 IWSLT2017）
    """
    def __init__(self, source_texts, target_texts, src_vocab, tgt_vocab, max_len=128):
        """
        Args:
            source_texts: 源语言文本列表
            target_texts: 目标语言文本列表
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            max_len: 最大序列长度
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Convert text to indices
        src_indices = self.text_to_indices(source_text, self.src_vocab)
        tgt_indices = self.text_to_indices(target_text, self.tgt_vocab)
        
        # Add <BOS> and <EOS> tokens to target
        tgt_indices = [self.tgt_vocab['<BOS>']] + tgt_indices + [self.tgt_vocab['<EOS>']]
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)
    
    @staticmethod
    def text_to_indices(text, vocab):
        """Convert text to indices"""
        words = text.split()
        indices = [vocab.get(word, vocab.get('<UNK>', 0)) for word in words]
        return indices


def collate_fn(batch, pad_idx=0, max_len=None):
    """
    Collate function for DataLoader
    
    Args:
        batch: list of (src, tgt) tuples
        pad_idx: padding index
        max_len: maximum length for padding
        
    Returns:
        src_batch: (batch_size, padded_src_len)
        tgt_batch: (batch_size, padded_tgt_len)
    """
    sources, targets = zip(*batch)
    
    if max_len:
        # Truncate if necessary
        sources = [s[:max_len] for s in sources]
        targets = [t[:max_len] for t in targets]
    
    # Find max lengths
    max_src_len = max(len(s) for s in sources)
    max_tgt_len = max(len(t) for t in targets)
    
    # Pad sequences
    src_batch = []
    tgt_batch = []
    
    for src, tgt in zip(sources, targets):
        src_padded = torch.cat([src, torch.full((max_src_len - len(src),), pad_idx, dtype=torch.long)])
        tgt_padded = torch.cat([tgt, torch.full((max_tgt_len - len(tgt),), pad_idx, dtype=torch.long)])
        
        src_batch.append(src_padded)
        tgt_batch.append(tgt_padded)
    
    return torch.stack(src_batch), torch.stack(tgt_batch)


def create_vocab(texts, special_tokens=None):
    """
    创建词汇表
    
    Args:
        texts: 文本列表
        special_tokens: 特殊token列表，如 ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
    Returns:
        vocab: 词汇表字典
    """
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Build vocabulary from texts
    word_count = {}
    for text in texts:
        words = text.split()
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
    
    # Add words to vocab (excluding special tokens)
    for word in sorted(word_count.keys()):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab


def load_sample_data():
    """
    加载示例数据用于测试
    
    返回示例的source和target文本
    """
    # 这是一个简单的示例，实际使用时需要加载真实数据
    source_texts = [
        "The quick brown fox jumps over the lazy dog .",
        "I love artificial intelligence .",
        "Transformers are powerful models ."
    ]
    
    target_texts = [
        "敏捷的棕色狐狸跳过懒狗 。",
        "我喜欢人工智能 。",
        "Transformer 是强大的模型 。"
    ]
    
    return source_texts, target_texts

