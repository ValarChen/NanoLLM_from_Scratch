# uv + WSL2 快速开始指南

本指南将帮助你在 WSL2 中快速设置和运行项目。

## 前置准备

### 1. 安装 WSL2（如果尚未安装）

在 Windows PowerShell (管理员保障) 中运行：

```powershell
wsl --install -d Ubuntu-22.04
```

### 2. 进入 WSL2

```bash
wsl
# 或
ubuntu
```

## 安装 uv

在 WSL2 中运行：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加到 PATH（如果未自动添加）
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 验证安装
uv --version
```

## 设置项目

### 选项 1: 在 Linux 文件系统（推荐，性能更好）

```bash
# 克隆到 Linux 文件系统
cd ~
git clone https://github.com/ValarChen/NanoLLM_from_Scratch.git
cd NanoLLM_from_Scratch
```

### 选项 2: 在 Windows 文件系统

```bash
# 进入 Windows 文件系统
cd /mnt/e/NanoLLM_from_Scratch

# 注意：Windows 文件系统性能较差
```

## 安装依赖并运行

```bash
# 1. 创建虚拟环境并安装依赖
uv sync

# 2. 运行训练
uv run python -m src.train
```

就这么简单！uv 会自动：
- 创建虚拟环境（.venv/）
- 安装所有依赖
- 管理 Python 版本

## 常用命令

```bash
# 运行代码
uv run python -m src.train

# 添加新依赖
uv add package-name

# 更新依赖
uv sync --upgrade

# 进入虚拟环境（如需手动激活）
source .venv/bin/activate
```

## VS Code 集成

### 1. 安装 WSL 扩展

在 VS Code 中安装 "Remote - WSL" 扩展

### 2. 打开项目

在 VS Code 中：
1. 按 `Ctrl+Shift+P`
2. 输入 "WSL: Connect to WSL"
3. 选择 Ubuntu
4. 打开项目文件夹

### 3. 选择 Python 解释器

1. 按 `Ctrl+Shift+P`
2. 输入 "Python: Select Interpreter"
3. 选择 `./venv/bin/python`

## 性能优化

在 Windows 用户目录创建 `.wslconfig`：

```ini
# %USERPROFILE%\.wslconfig
[wsl2]
memory=8GB
processors=4
swap=2GB
```

重启 WSL：
```powershell
wsl --shutdown
wsl
```

## 故障排除

### 问题：连接超时

如果在国内，可能需要使用代理：
```bash
export HTTPS_PROXY=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port
```

### 问题：权限错误

```bash
chmod +x scripts/run.sh
```

### 问题：找不到命令

```bash
source ~/.bashrc
```

## 下一步

查看 [WSL2_SETUP.md](WSL2_SETUP.md) 获取更详细的配置信息。

查看 [README.md](README.md) 了解项目架构和使用方法。

