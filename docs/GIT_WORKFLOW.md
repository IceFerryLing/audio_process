# Git 工作流

本文档固定本仓库的分支、提交、合并和大文件管理规则。目标是让研究阶段可以复现，同时避免模型、数据和实验产物进入普通 Git。

## 1. 长期引用

| 引用 | 用途 |
| --- | --- |
| `main` | 已通过当前阶段门禁的稳定代码，只通过 Pull Request 更新 |
| `feature/*` | 单一研究或工程任务，合并后删除 |
| `baseline-*` | 已完成基线的不可变里程碑标签 |

当前活动开发分支是 `feature/hubert-phone-ctc`。MFA 基线使用 `baseline-mfa-v1` 标签保存，不再保留长期 `MFA` 分支。

## 2. 创建任务分支

每个任务从最新远端 `main` 创建：

```powershell
git fetch --prune origin
git switch main
git merge --ff-only origin/main
git switch -c feature/<task-name>
```

不得从已经完成但尚未同步的旧功能分支继续分叉。确实存在依赖时，先合并依赖分支或在任务说明中记录依赖关系。

## 3. 提交规则

一个提交只表达一个可验证变化，推荐前缀：

```text
feat: 新增可用能力
fix: 修复行为错误
refactor: 不改变外部行为的结构调整
test: 增加或修正测试
docs: 更新文档
chore: Git、依赖或工程元数据维护
```

提交前至少运行与改动风险相称的测试和 `git diff --check`。训练代码变更还必须验证配置匹配、安全重跑和 checkpoint 恢复。

## 4. 合并与清理

功能分支先同步 `origin/main`，通过测试后创建 Pull Request。只允许对自己尚未合并的功能分支使用：

```powershell
git push --force-with-lease
```

禁止对 `main` 强推。功能分支合并后删除本地和远端引用：

```powershell
git branch -d feature/<task-name>
git push origin --delete feature/<task-name>
```

需要长期引用的实验节点使用 annotated tag，不使用永久功能分支：

```powershell
git tag -a baseline-<name>-v1 <commit> -m "<description>"
git push origin baseline-<name>-v1
```

## 5. 数据与大文件

以下内容不得进入普通 Git：

- LibriSpeech、学习者录音和其他数据集；
- HuBERT/Wav2Vec2/MFA 模型与 checkpoint；
- `artifacts/`、`runs/`、`outputs/`、缓存和训练日志；
- Python 字节码、虚拟环境和编辑器配置。

`.gitignore` 负责目录和文件类型门禁，`.gitattributes` 固定跨平台换行并声明二进制类型。需要发布大文件时使用对象存储、DVC 或经过明确配置的 Git LFS。

提交前检查：

```powershell
git status --short --ignored
git ls-files | rg "(__pycache__|\.pyc$|\.safetensors$|\.pt$|pytorch_model\.bin$)"
```

第二条命令正常情况下没有输出。

## 6. 历史修改与恢复

rebase、远端分支删除和 reflog 清理前必须创建并验证 bundle。bundle 放在仓库外，不提交回仓库：

```powershell
git bundle create <backup-path> --all
git bundle verify <backup-path>
```

恢复时可以从 bundle 克隆或提取指定引用。没有可验证备份时，不执行历史重写、`reflog expire` 或立即 prune。
