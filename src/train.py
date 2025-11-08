"""
训练和评估脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import os
import json
from datetime import datetime

from .model import Transformer
from .dataset import collate_fn, create_vocab, load_sample_data, load_iwslt2017, TranslationDataset


class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = Adam(self.model.parameters(), 
                                 lr=config['learning_rate'], 
                                 betas=(0.9, 0.98), 
                                 eps=1e-9)
        elif config['optimizer'] == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), 
                                  lr=config['learning_rate'],
                                  weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
        # Setup learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def _get_scheduler(self):
        """创建学习率调度器"""
        def lr_lambda(step):
            # Warmup + decay
            warmup_steps = self.config['warmup_steps']
            total_steps = self.config['num_epochs'] * len(self.train_loader)
            
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return (total_steps - step) / (total_steps - warmup_steps)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (src, tgt) in enumerate(pbar):
            # Move to device
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Create input and target for decoder
            # Target input is tgt[:-1], target output is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt_input, pad_idx=self.config['pad_idx'])
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            # Calculate loss
            loss = self.criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('max_grad_norm', 1.0))
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # Create input and target for decoder
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward pass
                output = self.model(src, tgt_input, pad_idx=self.config['pad_idx'])
                
                # Reshape for loss calculation
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(output, tgt_output)
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def compute_perplexity(self, loss):
        """计算困惑度"""
        return np.exp(loss)
    
    def train(self):
        """完整的训练流程"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            train_perplexity = self.compute_perplexity(train_loss)
            
            # Validate
            val_loss = self.validate()
            val_perplexity = self.compute_perplexity(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_perplexities.append(train_perplexity)
            self.val_perplexities.append(val_perplexity)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_perplexity:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_perplexity:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('results/best_model.pt')
                print("Saved best model!")
            
            # Save checkpoint
            self.save_checkpoint(epoch, 'results/checkpoint.pt')
            
            # Plot training curves
            self.plot_training_curves()
            
            # Save results table
            self.save_results_table()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
    
    def save_checkpoint(self, epoch, path):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        return checkpoint['epoch']
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='训练损失', marker='o', linewidth=2)
        axes[0].plot(self.val_losses, label='验证损失', marker='s', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Perplexity plot
        axes[1].plot(self.train_perplexities, label='训练困惑度', marker='o', linewidth=2)
        axes[1].plot(self.val_perplexities, label='验证困惑度', marker='s', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Perplexity', fontsize=12)
        axes[1].set_title('训练和验证困惑度曲线', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results_table(self):
        """保存结果表格"""
        import pandas as pd
        
        results_data = {
            'Epoch': list(range(1, len(self.train_losses) + 1)),
            'Train Loss': [f"{loss:.4f}" for loss in self.train_losses],
            'Val Loss': [f"{loss:.4f}" for loss in self.val_losses],
            'Train PPL': [f"{ppl:.2f}" for ppl in self.train_perplexities],
            'Val PPL': [f"{ppl:.2f}" for ppl in self.val_perplexities]
        }
        
        df = pd.DataFrame(results_data)
        df.to_csv('results/training_results.csv', index=False, encoding='utf-8-sig')
        
        # 也保存为Markdown表格
        with open('results/training_results.md', 'w', encoding='utf-8') as f:
            f.write("# 训练结果表格\n\n")
            f.write(df.to_markdown(index=False))
        
        print("结果表格已保存到 results/training_results.csv 和 results/training_results.md")


def create_data_loaders(config):
    """创建数据加载器"""
    # 尝试加载真实数据集
    use_real_data = config.get('use_real_data', True)
    max_samples = config.get('max_samples', None)  # 限制样本数用于快速测试
    
    if use_real_data:
        try:
            (train_src, train_tgt), (val_src, val_tgt) = load_iwslt2017(
                data_dir=config.get('data_dir', 'data'),
                max_samples=max_samples
            )
            print(f"使用IWSLT2017数据集: {len(train_src)} 训练样本, {len(val_src)} 验证样本")
        except Exception as e:
            print(f"加载真实数据集失败: {e}")
            print("使用示例数据...")
            source_texts, target_texts = load_sample_data()
            split_idx = int(len(source_texts) * 0.8)
            train_src = source_texts[:split_idx]
            train_tgt = target_texts[:split_idx]
            val_src = source_texts[split_idx:]
            val_tgt = target_texts[split_idx:]
    else:
        # Load sample data (for testing)
        source_texts, target_texts = load_sample_data()
        split_idx = int(len(source_texts) * 0.8)
        train_src = source_texts[:split_idx]
        train_tgt = target_texts[:split_idx]
        val_src = source_texts[split_idx:]
        val_tgt = target_texts[split_idx:]
    
    # 限制词汇表大小
    max_vocab_size = config.get('max_vocab_size', 10000)
    
    # Create vocabularies
    src_vocab = create_vocab(train_src)
    tgt_vocab = create_vocab(train_tgt)
    
    # 限制词汇表大小（保留最常用的词）
    if len(src_vocab) > max_vocab_size:
        # 统计词频
        word_count = {}
        for text in train_src:
            for word in text.split():
                word_count[word] = word_count.get(word, 0) + 1
        # 保留最常用的词
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        src_vocab = {token: idx for idx, token in enumerate(['<PAD>', '<UNK>', '<BOS>', '<EOS>'])}
        for word, _ in sorted_words[:max_vocab_size - 4]:
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)
    
    if len(tgt_vocab) > max_vocab_size:
        word_count = {}
        for text in train_tgt:
            for word in text.split():
                word_count[word] = word_count.get(word, 0) + 1
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        tgt_vocab = {token: idx for idx, token in enumerate(['<PAD>', '<UNK>', '<BOS>', '<EOS>'])}
        for word, _ in sorted_words[:max_vocab_size - 4]:
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)
    
    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, config['max_len'])
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, config['max_len'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=config['pad_idx'], max_len=config['max_len'])
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=config['pad_idx'], max_len=config['max_len'])
    )
    
    return train_loader, val_loader, len(src_vocab), len(tgt_vocab)


def main():
    """主函数"""
    # Configuration
    config = {
        'src_vocab_size': 10000,  # Will be updated
        'tgt_vocab_size': 10000,  # Will be updated
        'd_model': 512,
        'num_layers': 6,
        'h': 8,
        'd_ff': 2048,
        'max_len': 128,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'num_epochs': 10,
        'warmup_steps': 1000,
        'pad_idx': 0,
        'max_grad_norm': 1.0,
        'use_real_data': True,
        'data_dir': 'data',
        'max_vocab_size': 10000,
        'max_samples': None,  # 设置为数字可限制样本数用于快速测试
    }
    
    # Create data loaders and get vocab sizes
    train_loader, val_loader, src_vocab_size, tgt_vocab_size = create_data_loaders(config)
    config['src_vocab_size'] = src_vocab_size
    config['tgt_vocab_size'] = tgt_vocab_size
    
    # Create model
    model = Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        h=config['h'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    trainer.train()
    
    print("\nTraining completed!")
    print("Results saved in 'results/' directory")


if __name__ == '__main__':
    main()

