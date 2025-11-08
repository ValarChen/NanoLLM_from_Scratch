# Windows 安装指南

## 安装 Python 和 uv 到 D 或 E 盘

### 方法1：使用安装脚本（推荐）

运行 `install_python_uv.ps1` 脚本，它会自动：
1. 下载并安装 Python 到指定盘
2. 安装 uv
3. 配置环境变量
4. 安装项目依赖

### 方法2：手动安装

#### 步骤1：安装 Python

1. **下载 Python**
   - 访问：https://www.python.org/downloads/
   - 下载 Python 3.11 或 3.12（推荐 3.11）
   - 选择 Windows installer (64-bit)

2. **安装到 D 或 E 盘**
   - 运行安装程序
   - **重要**：勾选 "Add Python to PATH"
   - 点击 "Customize installation"
   - 在 "Advanced Options" 中：
     - 勾选 "Add Python to environment variables"
     - 修改安装路径为：`D:\Python311` 或 `E:\Python311`
   - 完成安装

3. **验证安装**
   ```powershell
   python --version
   pip --version
   ```

#### 步骤2：安装 uv

**方法A：使用 pip 安装（推荐）**
```powershell
pip install uv
```

**方法B：使用官方安装脚本**
```powershell
# PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

**方法C：使用 pipx（如果已安装）**
```powershell
pipx install uv
```

4. **验证 uv 安装**
```powershell
uv --version
```

#### 步骤3：配置环境变量（如果需要）

如果安装后命令不可用，需要手动添加环境变量：

1. 右键 "此电脑" → "属性" → "高级系统设置" → "环境变量"
2. 在 "系统变量" 的 Path 中添加：
   - `D:\Python311` 或 `E:\Python311`
   - `D:\Python311\Scripts` 或 `E:\Python311\Scripts`
   - `C:\Users\你的用户名\.cargo\bin`（uv的默认位置）

#### 步骤4：安装项目依赖

在项目目录下运行：

**使用 uv（推荐）：**
```powershell
cd E:\NanoLLM_from_Scratch
uv sync
```

**或使用 pip：**
```powershell
cd E:\NanoLLM_from_Scratch
pip install -r requirements.txt
```

### 快速安装脚本

我已经创建了自动安装脚本 `install_python_uv.ps1`，运行它即可自动完成所有步骤。

## 常见问题

### 1. "python 不是内部或外部命令"
- 检查 Python 是否已安装
- 检查环境变量 PATH 是否包含 Python 路径
- 重启命令行窗口或重启电脑

### 2. "pip 不是内部或外部命令"
- 确保安装 Python 时勾选了 "Add Python to PATH"
- 手动添加 Python 和 Scripts 目录到 PATH

### 3. uv 安装失败
- 确保网络连接正常
- 尝试使用管理员权限运行 PowerShell
- 检查防火墙设置

### 4. 权限问题
- 以管理员身份运行 PowerShell
- 或使用用户目录安装（不需要管理员权限）

## 验证安装

运行以下命令验证所有工具已正确安装：

```powershell
python --version
pip --version
uv --version
```

如果所有命令都能正常显示版本号，说明安装成功！

