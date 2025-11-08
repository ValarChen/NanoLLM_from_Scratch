# GitHub 同步说明

## 快速同步

### Windows用户

运行 `sync_to_github.bat` 脚本，它会自动：
1. 检查Git状态
2. 添加所有文件
3. 提交更改
4. 显示下一步操作

### Linux/Mac用户

运行 `sync_to_github.sh` 脚本：
```bash
bash sync_to_github.sh
```

## 手动同步步骤

### 1. 初始化Git仓库（如果还没有）

```bash
git init
```

### 2. 添加所有文件

```bash
git add .
```

### 3. 提交更改

```bash
git commit -m "完成Transformer模型实现和实验

- 实现完整的Transformer模型（Encoder-Decoder架构）
- 添加数据集下载功能（IWSLT2017）
- 实现训练和评估脚本
- 完成5个不同配置的实验对比
- 生成训练曲线图和结果表格
- 完成5页中文实验报告
- 所有实验在GPU上成功运行"
```

### 4. 添加远程仓库

如果还没有GitHub仓库，先在GitHub上创建一个新仓库，然后：

```bash
git remote add origin https://github.com/你的用户名/NanoLLM_from_Scratch.git
```

如果已经存在，检查远程仓库：
```bash
git remote -v
```

### 5. 推送到GitHub

```bash
# 设置主分支为main
git branch -M main

# 推送到GitHub
git push -u origin main
```

## 注意事项

### 被忽略的文件

根据 `.gitignore` 配置，以下文件不会被提交：

- `results/` - 训练结果目录（包含大文件）
- `data/` - 数据集目录
- `*.pt`, `*.pth` - 模型检查点文件
- `__pycache__/` - Python缓存
- `Description_of_the_Assignment.pdf` - 作业PDF
- `gemini讲解.md` - 讲解文档

### 如果需要提交结果文件

如果需要将结果文件也提交到GitHub（不推荐，因为文件较大），可以：

1. 临时修改 `.gitignore`，移除 `results/` 和 `*.pt`
2. 或使用 `git add -f` 强制添加特定文件

### 推荐做法

- 只提交代码和配置文件
- 结果文件（图表、表格）可以单独上传或使用GitHub Releases
- 模型检查点文件太大，建议使用Git LFS或云存储

## 提交的文件列表

主要提交的文件包括：

- ✅ 源代码 (`src/`)
- ✅ 配置文件 (`configs/`)
- ✅ 运行脚本 (`scripts/`)
- ✅ 依赖文件 (`requirements.txt`, `pyproject.toml`)
- ✅ 文档 (`README.md`, `运行指南.md`, `Windows安装指南.md`)
- ✅ 实验报告 (`results/实验报告.md`)
- ✅ 安装脚本 (`快速安装.bat`, `install_python_uv.ps1`)
- ❌ 训练结果（被忽略）
- ❌ 模型检查点（被忽略）
- ❌ 数据集（被忽略）

## 常见问题

### Q: 推送时提示需要认证

A: 使用个人访问令牌（Personal Access Token）：
1. GitHub → Settings → Developer settings → Personal access tokens
2. 生成新token
3. 推送时使用token作为密码

### Q: 推送失败，提示文件太大

A: 某些结果文件可能太大，检查 `.gitignore` 是否正确配置

### Q: 想提交结果文件怎么办？

A: 可以：
1. 只提交小的结果文件（CSV、MD）
2. 使用Git LFS管理大文件
3. 或创建GitHub Release上传

