# Qwen2.5-Math-1.5B GRPO/DAPO LoRA

这是一个针对 `Qwen2.5-Math-1.5B + GSM8K` 精简后的 verl 项目，只保留本次 GRPO-LoRA、DAPO-LoRA 训练需要的入口、数据处理、辅助脚本、运行记录和 verl 共享运行内核。

## 目录

```text
data/gsm8k/                         GSM8K parquet 数据
examples/data_preprocess/gsm8k.py   GSM8K 数据预处理
examples/grpo_trainer/              GRPO-LoRA 训练入口
examples/dapo/                      DAPO-LoRA 训练入口
scripts/                            安装、绘图和输出查看工具
docs/                               配置、结果和清理说明
verl/                               main_ppo、FSDP、vLLM、reward 等共享运行内核
logs/                               训练日志、指标、曲线和 rollout
tensorboard_log/                    TensorBoard 日志
```

## 环境

- Conda：`rl_study`
- Torch：`2.6.0+cu124`
- vLLM：`0.8.5`
- tensordict：`0.10.0`
- xformers：`0.0.29.post2`
- GPU：只使用 `CUDA_VISIBLE_DEVICES=2,6`
- verl：`trainer.n_gpus_per_node=2`

## 数据

```bash
cd /media/iie/4Tb/zccl/verl
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

默认训练数据：

```text
/home/iie/data/gsm8k/train.parquet
/home/iie/data/gsm8k/test.parquet
```

## GRPO-LoRA

```bash
cd /media/iie/4Tb/zccl/verl
bash examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
```

默认模型：

```text
/media/iie/4Tb/model/Qwen2.5-Math-1.5B
```

## DAPO-LoRA

```bash
cd /media/iie/4Tb/zccl/verl
bash examples/dapo/run_qwen2_5_math_1_5b_gsm8k_dapo_lora_test.sh
```

默认模型：

```text
Qwen/Qwen2.5-Math-1.5B
```

## 输出格式

GSM8K 原始格式要求最终答案为：

```text
#### 42
```

Qwen2.5-Math 常输出：

```text
\boxed{42}
```

当前 `verl/utils/reward_score/gsm8k.py` 的 strict reward 同时支持两种格式。

当前 checkout 的 GSM8K reward 还包含格式、推理过程、数字答案和正确性等分项，总分上限为 2.0。之前保留的 2026-07-09 训练日志属于历史 50-step smoke test；以后使用当前 reward 代码重跑时，score 数值尺度可能不同。

## 查看结果

训练脚本会写入：

- `logs/<experiment>.log`
- `logs/<experiment>.metrics.csv`
- `logs/<experiment>.loss.png`
- `logs/rollouts/<experiment>/<step>.jsonl`
- `tensorboard_log/verl_qwen25_math_lora/<experiment>/`

辅助工具：

```bash
python scripts/sample_gsm8k_outputs.py
python scripts/rollout_viewer.py --help
python scripts/plot_training_metrics.py logs/<experiment>.log
```

完整参数和 50-step 测试结果见 [docs/qwen25_math_1_5b_lora_test_config.md](docs/qwen25_math_1_5b_lora_test_config.md)。

## 清理安全

清理前 git 状态为干净状态，基线提交为 `76a2e47`。已建立保护分支：

```text
backup/pre-grpo-dapo-cleanup-20260710
```

被移出的上游示例、文档、测试和容器文件保存在：

```text
/media/iie/4Tb/zccl/.verl_cleanup_backup/2026-07-10_pre_grpo_dapo
```

具体保留范围和恢复方法见 [docs/cleanup_manifest.md](docs/cleanup_manifest.md)。

运行完整的 100-step 区间：
python scripts/summarize_grpo_rollouts.py \
  logs/grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26.metrics.csv \
  logs/rollouts/grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26 \
  --complete-only
默认不加 --complete-only 会同时显示当前未满 100 step 的区间。写出 CSV：
python scripts/summarize_grpo_rollouts.py \
  logs/grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26.metrics.csv \
  logs/rollouts/grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26 \
  --output logs/grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26.summary.csv
