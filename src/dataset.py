"""
数据加载和预处理模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from collections import Counter
import urllib.request
import tarfile
import zipfile


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


def download_iwslt2017(data_dir='data'):
    """
    下载IWSLT2017数据集（德语-英语）
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # IWSLT2017数据集URL
    base_url = "https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz"
    tgz_path = os.path.join(data_dir, 'de-en.tgz')
    extract_path = os.path.join(data_dir, 'iwslt2017')
    
    if not os.path.exists(extract_path):
        print("正在下载IWSLT2017数据集...")
        try:
            urllib.request.urlretrieve(base_url, tgz_path)
            print("下载完成，正在解压...")
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(data_dir)
            os.rename(os.path.join(data_dir, 'de-en'), extract_path)
            print("数据集准备完成！")
        except Exception as e:
            print(f"下载失败: {e}")
            print("使用备用数据集...")
            return load_sample_data()
    
    return extract_path


def load_iwslt2017(data_dir='data', max_samples=None):
    """
    加载IWSLT2017数据集
    
    Args:
        data_dir: 数据目录
        max_samples: 最大样本数（用于快速测试）
    
    Returns:
        source_texts, target_texts: 源语言和目标语言文本列表
    """
    extract_path = download_iwslt2017(data_dir)
    
    # 查找训练、验证和测试文件
    train_files = []
    val_files = []
    test_files = []
    
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if 'train' in file and file.endswith('.de'):
                train_files.append(os.path.join(root, file))
            elif 'dev' in file or 'valid' in file:
                if file.endswith('.de'):
                    val_files.append(os.path.join(root, file))
            elif 'test' in file and file.endswith('.de'):
                test_files.append(os.path.join(root, file))
    
    def read_file_pair(de_file, en_file):
        """读取德语-英语文件对"""
        source_texts = []
        target_texts = []
        
        try:
            with open(de_file, 'r', encoding='utf-8') as f_de, \
                 open(en_file, 'r', encoding='utf-8') as f_en:
                for de_line, en_line in zip(f_de, f_en):
                    de_line = de_line.strip()
                    en_line = en_line.strip()
                    if de_line and en_line:
                        source_texts.append(de_line)
                        target_texts.append(en_line)
                        if max_samples and len(source_texts) >= max_samples:
                            break
        except FileNotFoundError:
            print(f"文件未找到: {de_file} 或 {en_file}")
        
        return source_texts, target_texts
    
    # 加载训练数据
    train_src, train_tgt = [], []
    for de_file in train_files:
        en_file = de_file.replace('.de', '.en')
        if os.path.exists(en_file):
            src, tgt = read_file_pair(de_file, en_file)
            train_src.extend(src)
            train_tgt.extend(tgt)
    
    # 加载验证数据
    val_src, val_tgt = [], []
    for de_file in val_files:
        en_file = de_file.replace('.de', '.en')
        if os.path.exists(en_file):
            src, tgt = read_file_pair(de_file, en_file)
            val_src.extend(src)
            val_tgt.extend(tgt)
    
    # 如果没有找到数据，使用示例数据
    if not train_src:
        print("未找到IWSLT2017数据，使用示例数据...")
        return load_sample_data()
    
    print(f"加载了 {len(train_src)} 个训练样本")
    print(f"加载了 {len(val_src)} 个验证样本")
    
    return (train_src, train_tgt), (val_src, val_tgt)


def load_sample_data():
    """
    加载示例数据用于测试
    
    返回示例的source和target文本
    """
    # 这是一个简单的示例，实际使用时需要加载真实数据
    source_texts = [
        "The quick brown fox jumps over the lazy dog .",
        "I love artificial intelligence .",
        "Transformers are powerful models .",
        "Machine learning is fascinating .",
        "Deep neural networks can learn complex patterns .",
        "Natural language processing enables computers to understand text .",
        "Attention mechanisms improve model performance .",
        "The encoder processes input sequences .",
        "The decoder generates output sequences .",
        "Positional encoding adds location information ."
    ]
    
    target_texts = [
        "敏捷的棕色狐狸跳过懒狗 。",
        "我喜欢人工智能 。",
        "Transformer 是强大的模型 。",
        "机器学习很有趣 。",
        "深度神经网络可以学习复杂模式 。",
        "自然语言处理使计算机能够理解文本 。",
        "注意力机制提高了模型性能 。",
        "编码器处理输入序列 。",
        "解码器生成输出序列 。",
        "位置编码添加位置信息 。"
    ]
    
    return source_texts, target_texts

