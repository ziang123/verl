#!/usr/bin/env python3
"""Two-GPU LoRA SFT for strict GSM8K response formatting."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from verl.utils.reward_score.gsm8k import compute_score, extract_solution, hard_format_reward

ROOT_DIR = Path(__file__).resolve().parents[1]


DEFAULT_MODEL_PATH = "/media/iie/4Tb/model/Qwen2.5-3B-Instruct"


@dataclass(frozen=True)
class Example:
    messages: list[dict[str, str]]
    question: str
    response: str
    ground_truth: str


class JsonlDataset(Dataset[Example]):
    def __init__(self, path: Path, max_samples: int = 0):
        self.examples: list[Example] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.examples.append(
                    Example(
                        messages=row["messages"],
                        question=row["question"],
                        response=row["response"],
                        ground_truth=str(row["ground_truth"]),
                    )
                )
                if max_samples > 0 and len(self.examples) >= max_samples:
                    break
        if not self.examples:
            raise ValueError(f"No examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


class CompletionOnlyCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[Example]) -> dict[str, torch.Tensor]:
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                example.messages[:-1], tokenize=False, add_generation_prompt=True
            )
            for example in examples
        ]
        full_texts = [
            self.tokenizer.apply_chat_template(example.messages, tokenize=False, add_generation_prompt=False)
            for example in examples
        ]
        model_inputs = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.tokenizer(
            prompt_texts,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        labels = model_inputs["input_ids"].clone()
        labels[model_inputs["attention_mask"] == 0] = -100
        for row_index, prompt_ids in enumerate(prompt_inputs["input_ids"]):
            labels[row_index, : len(prompt_ids)] = -100
            if not torch.any(labels[row_index] != -100):
                raise ValueError("A sample has no assistant tokens after truncation; increase --max-length")
        model_inputs["labels"] = labels
        return model_inputs


class RunLogger:
    def __init__(self, output_dir: Path, enabled: bool):
        self.output_dir = output_dir
        self.enabled = enabled
        self.metrics: list[dict[str, float | int]] = []
        self.writer: SummaryWriter | None = None
        self.metrics_handle = None
        self.samples_handle = None
        if enabled:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(output_dir / "tensorboard")
            self.metrics_handle = (output_dir / "metrics.jsonl").open("w", encoding="utf-8")
            self.samples_handle = (output_dir / "generations.jsonl").open("w", encoding="utf-8")

    def log_metrics(self, row: dict[str, float | int]) -> None:
        if not self.enabled:
            return
        self.metrics.append(row)
        self.metrics_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.metrics_handle.flush()
        step = int(row["step"])
        for key, value in row.items():
            if key != "step":
                self.writer.add_scalar(key, value, step)
        self.writer.flush()
        self._write_plot()

    def log_generation(self, row: dict[str, object]) -> None:
        if not self.enabled:
            return
        self.samples_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.samples_handle.flush()
        self.writer.add_text(
            f"generation/{row['sample_index']}",
            str(row["generated"]).replace("\n", "  \n"),
            int(row["step"]),
        )
        self.writer.flush()

    def _write_plot(self) -> None:
        if not self.metrics:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for key in ("train/loss", "eval/loss"):
            points = [(int(row["step"]), float(row[key])) for row in self.metrics if key in row]
            if points:
                ax.plot([point[0] for point in points], [point[1] for point in points], label=key)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss.png", dpi=160)
        plt.close(fig)

    def close(self) -> None:
        if not self.enabled:
            return
        self.metrics_handle.close()
        self.samples_handle.close()
        self.writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = ROOT_DIR / "data" / "sft-data"
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-file", type=Path, default=data_dir / "train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=data_dir / "test.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs all configured epochs")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--num-generation-samples", type=int, default=1)
    parser.add_argument("--generation-max-new-tokens", type=int, default=192)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> PeftModel:
    model_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator) -> float:
    model.eval()
    gathered_losses = []
    for batch in loader:
        outputs = model(**batch)
        batch_size = batch["input_ids"].shape[0]
        gathered_losses.append(accelerator.gather_for_metrics(outputs.loss.detach().repeat(batch_size)))
    model.train()
    return torch.cat(gathered_losses).float().mean().item()


@torch.no_grad()
def log_generations(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    accelerator: Accelerator,
    run_logger: RunLogger,
    step: int,
    stage: str,
    max_new_tokens: int,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        was_training = unwrapped.training
        old_use_cache = unwrapped.config.use_cache
        unwrapped.eval()
        unwrapped.config.use_cache = True
        for sample_index, example in enumerate(examples):
            prompt = tokenizer.apply_chat_template(
                example.messages[:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
            output_ids = unwrapped.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            generated = generated.strip()
            extracted = extract_solution(generated, method="strict")
            format_pass = hard_format_reward(generated) == 0.3
            reward = compute_score(generated, example.ground_truth, method="strict")
            row = {
                "stage": stage,
                "step": step,
                "sample_index": sample_index,
                "question": example.question,
                "expected": example.response,
                "ground_truth": example.ground_truth,
                "generated": generated,
                "strict_format": format_pass,
                "extracted_answer": extracted,
                "reward": reward,
            }
            run_logger.log_generation(row)
            print(
                f"generation stage={stage} step={step} sample={sample_index} "
                f"strict={format_pass} reward={reward:.3f}\n{generated}\n",
                flush=True,
            )
        unwrapped.config.use_cache = old_use_cache
        if was_training:
            unwrapped.train()
    accelerator.wait_for_everyone()


def save_adapter(
    model: torch.nn.Module,
    tokenizer: Any,
    accelerator: Accelerator,
    output_dir: Path,
) -> None:
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=True,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if accelerator.num_processes != 2:
        raise RuntimeError(f"This run requires exactly two GPU processes, got {accelerator.num_processes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = JsonlDataset(args.train_file, args.max_train_samples)
    eval_dataset = JsonlDataset(args.eval_file, args.max_eval_samples)
    collator = CompletionOnlyCollator(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(args)
    trainable, total = count_parameters(model)
    accelerator.print(f"Trainable parameters: {trainable:,}/{total:,} ({trainable / total:.2%})")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    batches_per_process = math.ceil(len(train_loader) / accelerator.num_processes)
    updates_per_epoch = math.ceil(batches_per_process / args.gradient_accumulation_steps)
    full_train_steps = max(1, math.ceil(args.num_train_epochs * updates_per_epoch))
    max_train_steps = args.max_steps if args.max_steps > 0 else full_train_steps
    num_epochs = max(1, math.ceil(max_train_steps / updates_per_epoch))
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * args.warmup_ratio),
        num_training_steps=max_train_steps,
    )
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    run_logger = RunLogger(args.output_dir, accelerator.is_main_process)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
        config.update(
            {
                "world_size": accelerator.num_processes,
                "effective_batch_size": args.per_device_train_batch_size
                * accelerator.num_processes
                * args.gradient_accumulation_steps,
                "resolved_max_train_steps": max_train_steps,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
            }
        )
        with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")

    accelerator.print(
        f"Starting LoRA SFT: train={len(train_dataset)}, eval={len(eval_dataset)}, "
        f"steps={max_train_steps}, epochs={num_epochs}, world_size={accelerator.num_processes}"
    )
    generation_examples = eval_dataset.examples[: args.num_generation_samples]
    log_generations(
        model,
        tokenizer,
        generation_examples,
        accelerator,
        run_logger,
        step=0,
        stage="before_training",
        max_new_tokens=args.generation_max_new_tokens,
    )

    global_step = 0
    loss_sum = 0.0
    loss_count = 0
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            reduced_loss = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            loss_sum += reduced_loss
            loss_count += 1
            if not accelerator.sync_gradients:
                continue

            scheduler.step()
            global_step += 1
            if global_step % args.logging_steps == 0:
                train_loss = loss_sum / max(loss_count, 1)
                row = {
                    "step": global_step,
                    "train/loss": train_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": min(global_step / updates_per_epoch, num_epochs),
                }
                run_logger.log_metrics(row)
                accelerator.print(
                    f"step={global_step}/{max_train_steps} loss={train_loss:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )
                loss_sum = 0.0
                loss_count = 0

            should_eval = args.eval_steps > 0 and global_step % args.eval_steps == 0
            if should_eval or global_step == max_train_steps:
                eval_loss = evaluate(model, eval_loader, accelerator)
                run_logger.log_metrics({"step": global_step, "eval/loss": eval_loss})
                accelerator.print(
                    f"eval step={global_step} loss={eval_loss:.6f} "
                    f"ppl={math.exp(min(eval_loss, 20)):.3f}"
                )

            should_sample = args.sample_steps > 0 and global_step % args.sample_steps == 0
            if should_sample or global_step == max_train_steps:
                log_generations(
                    model,
                    tokenizer,
                    generation_examples,
                    accelerator,
                    run_logger,
                    step=global_step,
                    stage="during_training" if global_step < max_train_steps else "after_training",
                    max_new_tokens=args.generation_max_new_tokens,
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0 and global_step < max_train_steps:
                save_adapter(model, tokenizer, accelerator, args.output_dir / f"checkpoint-{global_step}")

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    save_adapter(model, tokenizer, accelerator, args.output_dir / "final")
    run_logger.close()
    accelerator.print(f"Saved final LoRA adapter to {args.output_dir / 'final'}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
