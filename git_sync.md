# Git 同步到 GitHub 指南

## 步骤

### 1. 初始化 Git 仓库（如果还没有）

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

### 4. 添加远程仓库（如果还没有）

```bash
git remote add origin https://github.com/你的用户名/NanoLLM_from_Scratch.git
```

或者如果已经存在，更新URL：
```bash
git remote set-url origin https://github.com/你的用户名/NanoLLM_from_Scratch.git
```

### 5. 推送到GitHub

```bash
git branch -M main
git push -u origin main
```

## 注意事项

根据 `.gitignore` 配置，以下文件/目录不会被提交：
- `results/` - 训练结果（包含大文件）
- `data/` - 数据集
- `*.pt`, `*.pth` - 模型检查点
- `__pycache__/` - Python缓存
- `Description_of_the_Assignment.pdf` - 作业PDF
- `gemini讲解.md` - 讲解文档

如果需要提交结果文件，可以临时修改 `.gitignore` 或使用 `git add -f` 强制添加。

