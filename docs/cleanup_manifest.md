# GRPO/DAPO 项目精简清单

清理日期：2026-07-10

## 原则

1. 只清理能明确证明与当前 GRPO/DAPO 训练无关的独立目录和脚本。
2. 不拆分 `verl/` 共享运行内核，因为 `main_ppo`、worker 注册、Hydra 配置和模型 monkey patch 存在动态导入。
3. 清理内容先移动到项目外备份区，不做不可恢复删除。
4. 清理前创建 git 保护分支。

## 保留

训练入口：

- `examples/data_preprocess/gsm8k.py`
- `examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`
- `examples/dapo/run_qwen2_5_math_1_5b_gsm8k_dapo_lora_test.sh`

辅助脚本：

- `scripts/install_vllm_wheelhouse.sh`
- `scripts/plot_training_metrics.py`
- `scripts/rollout_viewer.py`
- `scripts/sample_gsm8k_outputs.py`

运行内核和安装：

- `verl/`
- `pyproject.toml`
- `setup.py`
- `requirements.txt`
- `LICENSE`
- `Notice.txt`

数据和测试证据：

- `data/gsm8k/`
- `logs/`
- `outputs/`
- `tensorboard_log/`
- `docs/qwen25_math_1_5b_lora_test_config.md`

## 移出项目

- 其他算法 trainer 示例：PPO、RLOO、ReMax、GSPO、SAPO、GMPO、GPG、DPPO、GDPO、CISPO、OPD 等
- 其他模型示例：DeepSeek、Qwen3/Qwen3.5、视觉模型、Mistral、GLM、Moonlight、Nemotron、Seed 等
- 其他数据预处理脚本：AIME、Geo3K、HellaSwag、OpenR1MM、Pokemon、多轮工具数据等
- SFT、profile、generation、tutorial、tuning 和 rollout correction 示例
- Megatron、VeOmni、NPU、SGLang 安装/转换辅助脚本
- 上游全量 `docs/`
- 上游全量 `tests/`
- `docker/`
- `.github/` 和 `.vscode/`
- `.agent/`、`.claude/`、`.codex/` 和 `.gemini/` 代理配置目录
- NPU/test requirements、pre-commit 和 ReadTheDocs 配置

## 共享内核为何保留

当前两个脚本都执行：

```text
python -m verl.trainer.main_ppo
```

并使用：

- legacy `main_ppo_v0` trainer
- FSDP actor/ref
- vLLM async rollout
- LoRA 动态加载
- GSM8K reward
- naive/DAPO reward manager
- Hydra 配置
- Ray worker 和数据集

这些模块存在动态导入和注册逻辑。即使某些文件名看起来属于其他模型或后端，也可能在模块初始化或 worker 注册时被引用，所以本次没有冒险删除 `verl/` 内部共享包。

## 保护和恢复

清理前提交：

```text
76a2e47
```

保护分支：

```text
backup/pre-grpo-dapo-cleanup-20260710
```

外部备份：

```text
/media/iie/4Tb/zccl/.verl_cleanup_backup/2026-07-10_pre_grpo_dapo
```

恢复单个 git 文件：

```bash
git restore --source backup/pre-grpo-dapo-cleanup-20260710 -- path/to/file
```

恢复整个上游目录时，优先从外部备份复制，或者从保护分支创建新的 worktree 检查后再恢复。
