# Qwen2.5-Math-1.5B GSM8K LoRA 50-Step Test

测试目录：`/media/iie/4Tb/zccl/verl`

测试时间：2026-07-09

## 环境

- Conda 环境：`rl_study`
- Python：`/home/iie/miniconda3/envs/rl_study/bin/python`
- Torch：`2.6.0+cu124`
- vLLM：`0.8.5`
- tensordict：`0.10.0`
- xformers：`0.0.29.post2`
- 使用 GPU：`CUDA_VISIBLE_DEVICES=2,6`
- verl 配置：`trainer.n_gpus_per_node=2`

CPU 侧已限制线程，避免额外抢 CPU：

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `TOKENIZERS_PARALLELISM=false`
- `+ray_kwargs.ray_init.num_cpus=12`
- `data.dataloader_num_workers=0`
- `actor_rollout_ref.rollout.agent.num_workers=4`
- `reward.num_workers=2`

## 数据

已执行：

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

生成文件：

- `/home/iie/data/gsm8k/train.parquet`
- `/home/iie/data/gsm8k/test.parquet`

## 输出格式和 Reward 修正

GSM8K 原始 prompt 要求模型最后输出 `#### number`，原始 strict reward 也只从答案末尾匹配这种格式：

```text
#### 42
```

Qwen2.5-Math 模型实际更常输出数学模型格式：

```text
\boxed{42}
```

因此最早 reward 全 0 的主要原因不是 `MAX_PROMPT_LENGTH=256`，而是输出格式和 reward 解析不匹配。`MAX_PROMPT_LENGTH=256` 的训练日志里 `prompt_length/clip_ratio=0`，没有截断 prompt。`MAX_RESPONSE_LENGTH=256` 确实偏小，会截断较多推理过程，所以本次统一改为 `512`。

已修改 `verl/utils/reward_score/gsm8k.py`：

- strict 模式继续支持 `#### number`
- strict 模式新增支持 `\boxed{number}`
- 数字清洗支持去掉 `,` 和 `$`

模型采样输出查看脚本：

- `scripts/sample_gsm8k_outputs.py`
- 输出文件：`logs/gsm8k_sample_outputs.jsonl`

## GRPO-LoRA 脚本

路径：`examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`

主要参数：

- `actor_rollout_ref.model.path=/media/iie/4Tb/model/Qwen2.5-Math-1.5B`
- `algorithm.adv_estimator=grpo`
- `algorithm.use_kl_in_reward=False`
- `trainer.n_gpus_per_node=2`
- `data.train_batch_size=8`
- `actor_rollout_ref.actor.ppo_mini_batch_size=8`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
- `actor_rollout_ref.rollout.n=4`
- `actor_rollout_ref.rollout.temperature=0.7`
- `actor_rollout_ref.rollout.top_p=0.95`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.5`
- `data.max_prompt_length=256`
- `data.max_response_length=512`
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192`
- `actor_rollout_ref.model.lora_rank=16`
- `actor_rollout_ref.model.lora_alpha=16`
- `actor_rollout_ref.model.target_modules=all-linear`
- `actor_rollout_ref.actor.optim.lr=3e-6`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`
- `trainer.total_training_steps=50`
- `trainer.logger=["console","tensorboard"]`
- `trainer.rollout_data_dir=logs/rollouts/grpo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step`

50-step 测试结果：

- 退出码：0
- 日志：`logs/grpo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step.log`
- 曲线图：`logs/grpo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step.loss.png`
- 指标 CSV：`logs/grpo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step.metrics.csv`
- Rollout：`logs/rollouts/grpo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step/`
- 训练步数：50
- Rollout 文件数：50
- 最后一步 `critic/score/mean=0.8125`
- 最后一步 `critic/rewards/mean=0.8125`
- 最后一步 `actor/loss=-0.018458649516105652`
- 最后一步 `response_length/clip_ratio=0.0`
- 最后一步 `prompt_length/clip_ratio=0.0`
- 50 步平均 `critic/score/mean=0.635`
- 50 步平均 `response_length/clip_ratio=0.115625`

## DAPO-LoRA 脚本

路径：`recipe/dapo/run_qwen2_5_math_1_5b_gsm8k_dapo_lora_test.sh`

尽量与 GRPO 保持一致：

- 同样使用 `CUDA_VISIBLE_DEVICES=2,6`
- 同样使用 `trainer.n_gpus_per_node=2`
- 同样使用 GSM8K train/test parquet
- 同样 `data.train_batch_size=8`
- 同样 `actor_rollout_ref.actor.ppo_mini_batch_size=8`
- 同样 `actor_rollout_ref.rollout.n=4`
- 同样 `lora_rank=16`
- 同样 `lora_alpha=16`
- 同样 `max_prompt_length=256`
- 同样 `max_response_length=512`
- 同样 `ppo_max_token_len_per_gpu=8192`
- 同样跑 50 个 global training steps

DAPO 特有参数：

- `actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B`
- `actor_rollout_ref.rollout.temperature=1.0`
- `actor_rollout_ref.rollout.top_p=1.0`
- `actor_rollout_ref.actor.use_kl_loss=False`
- `actor_rollout_ref.actor.kl_loss_coef=0.0`
- `actor_rollout_ref.actor.clip_ratio_low=0.2`
- `actor_rollout_ref.actor.clip_ratio_high=0.28`
- `actor_rollout_ref.actor.clip_ratio_c=10.0`
- `actor_rollout_ref.actor.loss_agg_mode=token-mean`
- `reward.reward_manager.name=dapo`
- `reward.reward_kwargs.overlong_buffer_cfg.enable=True`
- `reward.reward_kwargs.overlong_buffer_cfg.len=64`
- `reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0`
- `reward.reward_kwargs.max_resp_len=512`
- `algorithm.filter_groups.enable=False`
- `algorithm.filter_groups.metric=acc`
- `algorithm.filter_groups.max_num_gen_batches=10`
- `actor_rollout_ref.rollout.max_num_batched_tokens=8192`

说明：DAPO 的 overlong buffer 会对过长/截断样本做惩罚，所以 `critic/score/min=-1` 和最后一步分数波动是正常现象。小 batch、50 step 的 smoke test 不代表最终训练效果。

50-step 测试结果：

- 退出码：0
- 日志：`logs/dapo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step.log`
- 曲线图：`logs/dapo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step.loss.png`
- 指标 CSV：`logs/dapo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step.metrics.csv`
- Rollout：`logs/rollouts/dapo_qwen25_math_1_5b_gsm8k_lora_rewardfix_2gpu26_50step/`
- 训练步数：50
- Rollout 文件数：50
- 最后一步 `critic/score/mean=-0.0576171875`
- 最后一步 `critic/rewards/mean=-0.0576171875`
- 最后一步 `actor/loss=0.1326463669538498`
- 最后一步 `response_length/clip_ratio=0.3125`
- 最后一步 `prompt_length/clip_ratio=0.0`
- 50 步平均 `critic/score/mean=0.110146484375`
- 50 步平均 `response_length/clip_ratio=0.24`

## 可视化

新增/更新脚本：

- `scripts/plot_training_metrics.py`

两份训练脚本会在训练时启动 watcher，训练结束后再重画一次：

- `logs/<experiment>.loss.png`
- `logs/<experiment>.metrics.csv`
- `logs/<experiment>.plot.log`

绘图脚本已支持解析 verl 日志里的 `np.float64(...)` 指标，并默认绘制 `actor/loss`、`actor/pg_loss`、`critic/rewards/mean`、`critic/score/mean`、`response_length/clip_ratio` 等指标。

## 兼容性补丁

为适配当前 `rl_study` 环境和 vLLM 0.8.5，做了这些小补丁：

- `verl/utils/tokenizer/continuous_token_wiring.py`：兼容 Python 3.10 没有标准库 `StrEnum` 的情况
- `verl/utils/attention_utils.py`：没有 `flash_attn` 时回退，避免直接导入失败
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`：兼容 vLLM 0.8.5 的 async server、CLI 参数、`TokensPrompt`、`priority`、sleep/reset 相关接口差异
- `scripts/install_vllm_wheelhouse.sh`：保留离线 wheelhouse 安装辅助脚本

## 验证命令

已通过：

```bash
bash -n examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
bash -n recipe/dapo/run_qwen2_5_math_1_5b_gsm8k_dapo_lora_test.sh
/home/iie/miniconda3/envs/rl_study/bin/python -m py_compile scripts/plot_training_metrics.py scripts/sample_gsm8k_outputs.py
```

训练结束后未发现残留的 Ray/vLLM 训练进程。
