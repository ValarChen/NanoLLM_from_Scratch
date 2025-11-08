"""
快速测试脚本 - 使用小数据集快速验证功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import main

if __name__ == '__main__':
    # 修改配置为快速测试
    import src.train as train_module
    
    # 保存原始main函数
    original_main = train_module.main
    
    def quick_main():
        config = {
            'src_vocab_size': 10000,
            'tgt_vocab_size': 10000,
            'd_model': 256,  # 较小的模型
            'num_layers': 3,  # 较少的层数
            'h': 4,
            'd_ff': 1024,
            'max_len': 64,
            'dropout': 0.1,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'optimizer': 'adam',
            'num_epochs': 5,  # 较少的epoch
            'warmup_steps': 100,
            'pad_idx': 0,
            'max_grad_norm': 1.0,
            'use_real_data': False,  # 使用示例数据快速测试
            'data_dir': 'data',
            'max_vocab_size': 5000,
            'max_samples': None,
        }
        
        from src.train import create_data_loaders, Trainer
        from src.model import Transformer
        
        # Create data loaders
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
        
        print("\n快速测试完成！")
        print("结果保存在 results/ 目录")
    
    train_module.main = quick_main
    quick_main()

