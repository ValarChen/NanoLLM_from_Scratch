# WSL2 + uv 环境设置指南

本指南将帮助你在 WSL2 环境中使用 uv 设置项目环境。

## 前置条件

1. **安装 WSL2**
   ```bash
   # 在 Windows PowerShell (管理员) 中运行
   wsl --install
   
   # 或安装特定发行版
   wsl --install -d Ubuntu-22.04
   ```

2. **更新系统**
   ```bash
   # 在 WSL2 中运行
   sudo apt update && sudo apt upgrade -y
   ```

## 安装 uv

### 方法 1: 使用官方安装脚本（推荐）

```bash
# 在 WSL2 中运行
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 方法 2: 使用 pip

```bash
pip install uv
```

### 方法 3: 使用 Homebrew (如果已安装)

```bash
brew install uv
```

### 验证安装

```bash
uv --version
```

## 设置项目环境

### 1. 克隆项目（如果在 WSL2 中）

```bash
# 如果项目在 Windows 文件系统
cd /mnt/e/NanoLLM_from_Scratch

# 或克隆到 Linux 文件系统（性能更好）
cd ~
git clone https://github.com/ValarChen/NanoLLM_from_Scratch.git
cd NanoLLM_from_Scratch
```

### 2. 创建虚拟环境

uv 会自动创建虚拟环境：

```bash
# 创建虚拟环境并安装依赖
uv sync

# 这会创建 .venv/ 目录并安装所有依赖
```

### 3. 激活环境

```bash
# uv 会自动激活环境
source .venv/bin/activate

# 验证 Python 版本
python --version
```

### 4. 运行项目

```bash
# 方式 1: 使用 uv run（自动激活环境）
uv run python -m src.train

# 方式 2: 手动激活环境后运行
source .venv/bin/activate
python -m src.train
```

## uv 常用命令

### 依赖管理

```bash
# 安装依赖
uv sync

# 添加新依赖
uv add package-name

# 添加开发依赖
uv add --dev pytest

# 移除依赖
uv remove package-name

# 更新依赖
uv lock --upgrade
uv sync --upgrade
```

### 虚拟环境管理

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate

# 指定 Python 版本
uv venv --python python3.11
```

### 运行脚本

```bash
# 运行 Python 脚本（自动使用项目环境）
uv run python script.py

# 运行模块
uv run python -m src.train

# 运行命令（会在项目环境中执行）
uv run pytest
```

## 性能优化建议

### 1. 使用 Linux 文件系统

WSL2 中访问 Windows 文件系统（/mnt/c/ 等）性能较差。建议：

```bash
# 克隆到 Linux 文件系统
cd ~/projects
git clone https://github.com/ValarChen/NanoLLM_from_Scratch.git

# 或使用符号链接
ln -s /mnt/e/NanoLLM_from_Scratch ~/NanoLLM_from_Scratch
```

### 2. 配置 WSL2 内存限制

在 Windows 用户目录创建 `.wslconfig`:

```ini
# %USERPROFILE%\.wslconfig
[wsl2]
memory=8GB
processors=4
swap=2GB
```

然后重启 WSL:

```powershell
wsl --shutdown
```

### 3. 使用 GPU (如果可用)

```bash
# 安装 NVIDIA CUDA Toolkit for WSL2
# 然后验证
nvidia-smi
```

## 故障排除

### 问题 1: uv 命令未找到

```bash
# 添加到 PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 问题 2: 权限错误

```bash
# 修复权限
chmod +x .venv/bin/activate
```

### 问题 3: Python 版本不匹配

```bash
# 列出可用版本
uv python list

# 使用特定版本
uv venv --python 3.11
```

### 问题 4: 依赖安装失败

```bash
# 清理并重新安装
uv sync --clean
```

## 与 VS Code 集成

### 1. 安装 WSL 扩展

在 VS Code 中安装 "Remote - WSL" 扩展

### 2. 连接到 WSL

- 按 `Ctrl+Shift+P`
- 输入 "WSL: Connect to WSL"
- 选择你的发行版

### 3. 选择 Python 解释器

- 按 `Ctrl+Shift+P`
- 输入 "Python: Select Interpreter"
- 选择 `.venv/bin/python`

## 工作流程

### 典型开发流程

```bash
# 1. 进入 WSL2
wsl

# 2. 进入项目目录
cd ~/NanoLLM_from_Scratch

# 3. 激活环境
source .venv/bin/activate

# 4. 开发/运行
uv run python -m src.train

# 5. 提交更改
git add .
git commit -m "your message"
git push
```

### 从 Windows 访问文件

在 Windows 中可以直接访问：

```
\\wsl$\Ubuntu-22.04\home\your_username\NanoLLM_from_Scratch
```

## 参考资源

- [uv 官方文档](https://docs.astral.sh/uv/)
- [WSL2 文档](https://learn.microsoft.com/zh-cn/windows/wsl/)
- [VS Code WSL 教程](https://code.visualstudio.com/docs/remote/wsl)

## 优势总结

使用 uv + WSL2 的好处：

1. ✅ **环境隔离**: 每个项目独立的虚拟环境
2. ✅ **快速安装**: uv 比 pip 快 10-100 倍
3. ✅ **一致的依赖**: 锁文件确保依赖一致性
4. ✅ **WSL2 性能**: 接近原生 Linux 性能
5. ✅ **跨平台**: 在 Windows 和 Linux 上都能使用
6. ✅ **易于复制**: 使用 pyproject.toml 管理依赖

