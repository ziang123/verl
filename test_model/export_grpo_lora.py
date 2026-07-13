#!/usr/bin/env python3
"""Export only the PEFT adapter from a verl FSDP LoRA checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(args.checkpoint_dir),
        target_dir=str(args.output_dir),
        hf_model_config_path=str(args.checkpoint_dir / "huggingface"),
    )
    merger = FSDPModelMerger(config)
    world_size = merger._get_world_size()
    rank_zero_state = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero_state, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    del rank_zero_state
    state_dict = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)
    adapter_path = merger.save_lora_adapter(state_dict)
    if adapter_path is None:
        raise RuntimeError(f"No LoRA parameters found in {args.checkpoint_dir}")
    weights_path = Path(adapter_path) / "adapter_model.safetensors"
    print(f"Exported {weights_path} ({os.path.getsize(weights_path):,} bytes)")


if __name__ == "__main__":
    main()
