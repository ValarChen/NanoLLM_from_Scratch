#!/bin/bash

# 运行Transformer训练脚本

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python -m src.train

# 可选：使用配置文件
# python -m src.train --config configs/base_config.yaml --seed 42

echo "Training completed!"

