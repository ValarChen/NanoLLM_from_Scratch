"""
数据集下载测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import download_iwslt2017, load_iwslt2017

def main():
    print("=" * 60)
    print("数据集下载测试")
    print("=" * 60)
    
    # 数据集会下载到 data/ 目录
    data_dir = 'data'
    
    print(f"\n数据集将下载到: {os.path.abspath(data_dir)}")
    print(f"完整路径: {os.path.abspath(os.path.join(data_dir, 'iwslt2017'))}")
    
    # 尝试下载
    print("\n开始下载数据集...")
    try:
        extract_path = download_iwslt2017(data_dir)
        print(f"\n✅ 数据集路径: {extract_path}")
        
        # 检查文件
        if os.path.exists(extract_path):
            print(f"\n✅ 数据集目录存在: {extract_path}")
            files = os.listdir(extract_path)
            print(f"✅ 找到 {len(files)} 个文件/目录")
            
            # 尝试加载数据
            print("\n尝试加载数据集...")
            (train_src, train_tgt), (val_src, val_tgt) = load_iwslt2017(data_dir, max_samples=10)
            print(f"✅ 成功加载数据")
            print(f"   - 训练样本: {len(train_src)}")
            print(f"   - 验证样本: {len(val_src)}")
            print(f"\n示例训练数据:")
            print(f"   源语言: {train_src[0][:100]}...")
            print(f"   目标语言: {train_tgt[0][:100]}...")
        else:
            print(f"\n❌ 数据集目录不存在: {extract_path}")
            print("可能下载失败，将使用示例数据")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n将使用示例数据进行训练")

if __name__ == '__main__':
    main()

