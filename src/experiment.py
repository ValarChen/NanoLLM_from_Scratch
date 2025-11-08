"""
实验脚本：运行多个配置并生成对比结果
"""

import torch
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from .train import Trainer, create_data_loaders
from .model import Transformer


def run_experiment(config, experiment_name):
    """
    运行单个实验
    
    Args:
        config: 配置字典
        experiment_name: 实验名称
    
    Returns:
        results: 实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"{'='*60}")
    
    # 创建结果目录
    exp_dir = os.path.join('results', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 创建数据加载器
    train_loader, val_loader, src_vocab_size, tgt_vocab_size = create_data_loaders(config)
    config['src_vocab_size'] = src_vocab_size
    config['tgt_vocab_size'] = tgt_vocab_size
    
    # 创建模型
    model = Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        h=config['h'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    )
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 修改保存路径
    original_results_dir = trainer.config.get('results_dir', 'results')
    
    # 临时修改保存路径
    import src.train as train_module
    original_create_dir = os.makedirs
    
    def makedirs_exp(path, *args, **kwargs):
        if 'results' in path:
            path = path.replace('results', exp_dir)
        return original_create_dir(path, *args, **kwargs)
    
    # 重写保存方法
    class ExperimentTrainer(Trainer):
        def save_model(self, path):
            path = os.path.join(exp_dir, os.path.basename(path))
            return super().save_model(path)
        
        def save_checkpoint(self, epoch, path):
            path = os.path.join(exp_dir, os.path.basename(path))
            return super().save_checkpoint(epoch, path)
        
        def plot_training_curves(self):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(self.train_losses, label='训练损失', marker='o', linewidth=2)
            axes[0].plot(self.val_losses, label='验证损失', marker='s', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title(f'{experiment_name} - 训练和验证损失曲线', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(self.train_perplexities, label='训练困惑度', marker='o', linewidth=2)
            axes[1].plot(self.val_perplexities, label='验证困惑度', marker='s', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Perplexity', fontsize=12)
            axes[1].set_title(f'{experiment_name} - 训练和验证困惑度曲线', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        def save_results_table(self):
            results_data = {
                'Epoch': list(range(1, len(self.train_losses) + 1)),
                'Train Loss': [f"{loss:.4f}" for loss in self.train_losses],
                'Val Loss': [f"{loss:.4f}" for loss in self.val_losses],
                'Train PPL': [f"{ppl:.2f}" for ppl in self.train_perplexities],
                'Val PPL': [f"{ppl:.2f}" for ppl in self.val_perplexities]
            }
            df = pd.DataFrame(results_data)
            df.to_csv(os.path.join(exp_dir, 'training_results.csv'), index=False, encoding='utf-8-sig')
            with open(os.path.join(exp_dir, 'training_results.md'), 'w', encoding='utf-8') as f:
                f.write(f"# {experiment_name} 训练结果表格\n\n")
                f.write(df.to_markdown(index=False))
    
    # 创建新的训练器实例
    trainer = ExperimentTrainer(model, train_loader, val_loader, config)
    
    # 训练
    trainer.train()
    
    # 返回结果
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
        'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
        'final_train_ppl': trainer.train_perplexities[-1] if trainer.train_perplexities else None,
        'final_val_ppl': trainer.val_perplexities[-1] if trainer.val_perplexities else None,
        'best_val_loss': min(trainer.val_losses) if trainer.val_losses else None,
        'best_val_ppl': min(trainer.val_perplexities) if trainer.val_perplexities else None,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_perplexities': trainer.train_perplexities,
        'val_perplexities': trainer.val_perplexities
    }
    
    return results


def run_all_experiments():
    """运行所有实验配置"""
    
    # 定义实验配置
    experiments = [
        {
            'name': 'baseline',
            'config': {
                'd_model': 512,
                'num_layers': 6,
                'h': 8,
                'd_ff': 2048,
                'dropout': 0.1,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'optimizer': 'adam',
                'num_epochs': 10,
                'warmup_steps': 1000,
                'pad_idx': 0,
                'max_grad_norm': 1.0,
                'max_len': 128,
                'use_real_data': True,
                'data_dir': 'data',
                'max_vocab_size': 10000,
                'max_samples': 5000,  # 限制样本数用于快速实验
            }
        },
        {
            'name': 'small_model',
            'config': {
                'd_model': 256,
                'num_layers': 4,
                'h': 4,
                'd_ff': 1024,
                'dropout': 0.1,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'optimizer': 'adam',
                'num_epochs': 10,
                'warmup_steps': 1000,
                'pad_idx': 0,
                'max_grad_norm': 1.0,
                'max_len': 128,
                'use_real_data': True,
                'data_dir': 'data',
                'max_vocab_size': 10000,
                'max_samples': 5000,
            }
        },
        {
            'name': 'large_model',
            'config': {
                'd_model': 512,
                'num_layers': 8,
                'h': 8,
                'd_ff': 2048,
                'dropout': 0.1,
                'batch_size': 16,  # 减小batch size以适应更大的模型
                'learning_rate': 1e-4,
                'optimizer': 'adam',
                'num_epochs': 10,
                'warmup_steps': 1000,
                'pad_idx': 0,
                'max_grad_norm': 1.0,
                'max_len': 128,
                'use_real_data': True,
                'data_dir': 'data',
                'max_vocab_size': 10000,
                'max_samples': 5000,
            }
        },
        {
            'name': 'high_lr',
            'config': {
                'd_model': 512,
                'num_layers': 6,
                'h': 8,
                'd_ff': 2048,
                'dropout': 0.1,
                'batch_size': 32,
                'learning_rate': 5e-4,  # 更高的学习率
                'optimizer': 'adam',
                'num_epochs': 10,
                'warmup_steps': 1000,
                'pad_idx': 0,
                'max_grad_norm': 1.0,
                'max_len': 128,
                'use_real_data': True,
                'data_dir': 'data',
                'max_vocab_size': 10000,
                'max_samples': 5000,
            }
        },
        {
            'name': 'low_lr',
            'config': {
                'd_model': 512,
                'num_layers': 6,
                'h': 8,
                'd_ff': 2048,
                'dropout': 0.1,
                'batch_size': 32,
                'learning_rate': 5e-5,  # 更低的学习率
                'optimizer': 'adam',
                'num_epochs': 10,
                'warmup_steps': 1000,
                'pad_idx': 0,
                'max_grad_norm': 1.0,
                'max_len': 128,
                'use_real_data': True,
                'data_dir': 'data',
                'max_vocab_size': 10000,
                'max_samples': 5000,
            }
        },
    ]
    
    all_results = []
    
    # 运行所有实验
    for exp in experiments:
        try:
            results = run_experiment(exp['config'], exp['name'])
            all_results.append(results)
        except Exception as e:
            print(f"实验 {exp['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比表格
    generate_comparison_table(all_results)
    
    # 生成对比图表
    generate_comparison_plots(all_results)
    
    return all_results


def generate_comparison_table(results):
    """生成对比表格"""
    comparison_data = []
    
    for r in results:
        comparison_data.append({
            '实验名称': r['experiment_name'],
            'd_model': r['config']['d_model'],
            'num_layers': r['config']['num_layers'],
            'num_heads': r['config']['h'],
            'learning_rate': r['config']['learning_rate'],
            '最终训练损失': f"{r['final_train_loss']:.4f}" if r['final_train_loss'] else 'N/A',
            '最终验证损失': f"{r['final_val_loss']:.4f}" if r['final_val_loss'] else 'N/A',
            '最佳验证损失': f"{r['best_val_loss']:.4f}" if r['best_val_loss'] else 'N/A',
            '最终训练困惑度': f"{r['final_train_ppl']:.2f}" if r['final_train_ppl'] else 'N/A',
            '最终验证困惑度': f"{r['final_val_ppl']:.2f}" if r['final_val_ppl'] else 'N/A',
            '最佳验证困惑度': f"{r['best_val_ppl']:.2f}" if r['best_val_ppl'] else 'N/A',
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 保存为CSV
    df.to_csv('results/experiment_comparison.csv', index=False, encoding='utf-8-sig')
    
    # 保存为Markdown
    with open('results/experiment_comparison.md', 'w', encoding='utf-8') as f:
        f.write("# 实验对比结果表格\n\n")
        f.write(df.to_markdown(index=False))
    
    print("\n对比表格已保存到 results/experiment_comparison.csv 和 results/experiment_comparison.md")
    print("\n对比结果:")
    print(df.to_string(index=False))


def generate_comparison_plots(results):
    """生成对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 损失对比
    for r in results:
        axes[0, 0].plot(r['train_losses'], label=f"{r['experiment_name']} - 训练", linestyle='-', linewidth=2)
        axes[0, 0].plot(r['val_losses'], label=f"{r['experiment_name']} - 验证", linestyle='--', linewidth=2)
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('所有实验的损失对比', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 困惑度对比
    for r in results:
        axes[0, 1].plot(r['train_perplexities'], label=f"{r['experiment_name']} - 训练", linestyle='-', linewidth=2)
        axes[0, 1].plot(r['val_perplexities'], label=f"{r['experiment_name']} - 验证", linestyle='--', linewidth=2)
    
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Perplexity', fontsize=12)
    axes[0, 1].set_title('所有实验的困惑度对比', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 最佳验证损失对比（柱状图）
    exp_names = [r['experiment_name'] for r in results]
    best_val_losses = [r['best_val_loss'] for r in results if r['best_val_loss']]
    if best_val_losses:
        axes[1, 0].bar(exp_names[:len(best_val_losses)], best_val_losses, color='skyblue', edgecolor='navy', linewidth=1.5)
        axes[1, 0].set_ylabel('最佳验证损失', fontsize=12)
        axes[1, 0].set_title('最佳验证损失对比', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 最佳验证困惑度对比（柱状图）
    best_val_ppls = [r['best_val_ppl'] for r in results if r['best_val_ppl']]
    if best_val_ppls:
        axes[1, 1].bar(exp_names[:len(best_val_ppls)], best_val_ppls, color='lightcoral', edgecolor='darkred', linewidth=1.5)
        axes[1, 1].set_ylabel('最佳验证困惑度', fontsize=12)
        axes[1, 1].set_title('最佳验证困惑度对比', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("对比图表已保存到 results/experiment_comparison.png")


if __name__ == '__main__':
    print("开始运行所有实验...")
    results = run_all_experiments()
    print("\n所有实验完成！")
    print(f"共完成 {len(results)} 个实验")

