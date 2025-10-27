# NanoLLM from Scratch

A project for BJTU LLM class - Implementation of Transformer from scratch.

## Project Overview

This project implements a complete Transformer model from scratch for sequence-to-sequence tasks (e.g., machine translation). The implementation follows the architecture described in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Features

- **Complete Transformer Implementation**: Encoder-Decoder architecture with multi-head attention
- **Modular Design**: Clean separation of components (Attention, FFN, Positional Encoding, etc.)
- **Training Pipeline**: Full training loop with validation, checkpointing, and learning rate scheduling
- **Experiments**: Configurable experiments with results visualization

## Project Structure

```
NanoLLM_from_Scratch/
├── src/                      # Source code
│   ├── modules.py            # Core modules (Attention, FFN, etc.)
│   ├── model.py              # Transformer model definition
│   ├── train.py              # Training and evaluation scripts
│   └── dataset.py            # Data loading and preprocessing
├── scripts/
│   ├── run.sh                # Unix/Linux run script
│   └── run.bat               # Windows run script
├── configs/
│   └── base_config.yaml      # Configuration file
├── results/                  # Training results and plots
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Installation

### 方式 1: 使用 uv（推荐，更快速）

**在 WSL2 中设置（推荐）：**
详见 [WSL2_SETUP.md](WSL2_SETUP.md)

**快速开始：**
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/ValarChen/NanoLLM_from_Scratch.git
cd NanoLLM_from_Scratch

# 创建环境并安装依赖
uv sync

# 运行项目
uv run python -m src.train
```

### 方式 2: 使用传统 venv + pip

1. **Clone the repository**
```bash
git clone https://github.com/ValarChen/NanoLLM_from_Scratch.git
cd NanoLLM_from_Scratch
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the training script:

```bash
# 使用 uv (推荐，自动管理环境)
uv run python -m src.train

# 或使用脚本
# On Unix/Linux/Mac
bash scripts/run.sh

# On Windows
scripts\run.bat

# 传统方式（需要先激活环境）
source .venv/bin/activate  # 或 .venv\Scripts\activate on Windows
python -m src.train
```

### Training Configuration

The default configuration is defined in `src/train.py`. You can modify:

- Model parameters: `d_model`, `num_layers`, `num_heads`, `d_ff`
- Training parameters: `batch_size`, `learning_rate`, `num_epochs`, `optimizer`
- Data parameters: `max_len`

## Model Architecture

### Core Components

1. **Scaled Dot-Product Attention**
   - Formula: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
   - Includes masking support for padding and causal masks

2. **Multi-Head Attention**
   - Splits input into multiple heads for parallel attention computation
   - Enables model to attend to different representation subspaces

3. **Position-wise Feed-Forward Network**
   - Formula: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`
   - Adds non-linearity to the model

4. **Positional Encoding**
   - Sinusoidal position encodings to inject position information
   - Formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`

### Encoder-Decoder Architecture

- **Encoder**: Stack of N identical layers, each containing:
  - Multi-Head Self-Attention
  - Position-wise Feed-Forward Network
  - Residual connections and Layer Normalization

- **Decoder**: Stack of N identical layers, each containing:
  - Masked Multi-Head Self-Attention
  - Multi-Head Cross-Attention
  - Position-wise Feed-Forward Network
  - Residual connections and Layer Normalization

## Training

### Training Process

The training script (`src/train.py`) provides:

- **Trainer class**: Manages the entire training lifecycle
- **Automatic validation**: Evaluates on validation set each epoch
- **Learning rate scheduling**: Warmup + decay
- **Checkpointing**: Saves best model and periodic checkpoints
- **Visualization**: Plots training curves (loss and perplexity)

### Example Training Output

```
Epoch 1/10
Training: 100%|████████| 10/10 [00:05<00:00,  1.89it/s, loss=4.23]
Validating: 100%|████████| 2/2 [00:00<00:00, 15.42it/s]
Train Loss: 4.2341, Train PPL: 68.9612
Val Loss: 3.8765, Val PPL: 48.2243
Saved best model!
```

### Monitoring Training

Training results are saved in the `results/` directory:

- `training_curves.png`: Training and validation curves
- `best_model.pt`: Best model checkpoint
- `checkpoint.pt`: Latest checkpoint with optimizer state

## Results

After training, you can view:

1. **Training curves**: Loss and perplexity plots in `results/training_curves.png`
2. **Model checkpoint**: Best model saved in `results/best_model.pt`

## Experimental Setup

### Default Hyperparameters

```python
{
    'd_model': 512,
    'num_layers': 6,
    'h': 8,              # number of heads
    'd_ff': 2048,
    'dropout': 0.1,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'optimizer': 'adam',
    'num_epochs': 10,
    'max_len': 128
}
```

### Ablation Studies

To perform ablation studies:

1. **Remove positional encoding**: Comment out positional encoding in `src/model.py`
2. **Vary number of heads**: Test with `h=1, 4, 8`
3. **Adjust model depth**: Try `num_layers=2, 4, 6`

## Development

### Adding New Features

- New modules: Add to `src/modules.py`
- Dataset loading: Implement in `src/dataset.py`
- Training logic: Extend `src/train.py`

### Testing

Run basic tests:

```python
# Test model forward pass
python -c "from src.model import Transformer; import torch; model = Transformer(1000, 1000); x = torch.randint(0, 1000, (2, 10)); y = torch.randint(0, 1000, (2, 10)); out = model(x, y); print(out.shape)"
```

## Implementation Details

### Key Features

- **Proper masking**: Padding mask and causal mask implementation
- **Residual connections**: With layer normalization
- **Xavier initialization**: For all linear layers
- **Gradient clipping**: To prevent gradient explosion
- **Learning rate warmup**: Smooth start for training

### Dimension Flow

```
Input (batch_size, seq_len) 
→ Embedding (batch_size, seq_len, d_model)
→ Positional Encoding (batch_size, seq_len, d_model)
→ Encoder Layers → (batch_size, seq_len, d_model)
→ Decoder Layers → (batch_size, tgt_seq_len, d_model)
→ Output Projection → (batch_size, tgt_seq_len, vocab_size)
```

## Dependencies

**项目管理：**
- 使用 `pyproject.toml` 定义项目配置和依赖
- 支持使用 `uv` 进行快速依赖管理
- 兼容传统的 `requirements.txt` + `pip`

**主要依赖：**
- PyTorch >= 2.0.0
- NumPy >= 1.23.0
- Matplotlib >= 3.7.0
- tqdm >= 4.65.0
- PyYAML >= 6.0
- TensorBoard >= 2.13.0
- SentencePiece >= 0.1.99

### 依赖管理命令

**使用 uv (推荐):**
```bash
# 安装所有依赖
uv sync

# 添加新依赖
uv add package-name

# 移除依赖
uv remove package-name

# 更新依赖
uv sync --upgrade
```

**使用 pip:**
```bash
# 安装依赖
pip install -r requirements.txt

# 生成 requirements.txt
pip freeze > requirements.txt
```

## WSL2 设置

项目包含完整的 WSL2 + uv 设置指南，详见 [WSL2_SETUP.md](WSL2_SETUP.md)。

## License

This project is for educational purposes as part of the BJTU LLM class.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Blog post explaining Transformer architecture

## Author

- Student at BJTU

## Acknowledgments

- Original Transformer paper by Vaswani et al.
- BJTU LLM class instructors
