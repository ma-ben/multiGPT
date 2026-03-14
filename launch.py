from omegaconf import OmegaConf
import os
import argparse

CONFIG_FILE = "configs/gpt2.yaml"


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=CONFIG_FILE)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--port_debug", type=int, default=5678)
args = parser.parse_args()

launch_mode = f"debugpy-run -m torch.distributed.run -p {args.port_debug} --" \
                if args.debug else \
                "torchrun"

cfg = OmegaConf.load(args.config)

word_size = (
    cfg.distributed.tp_size
    * cfg.distributed.cp_size
    * cfg.distributed.pp_size
    * cfg.distributed.dp_size
)
assert cfg.model.vocab_size % cfg.distributed.tp_size == 0, "vocab_size must be divisible by tp_size for Tensor Parallelism"
assert cfg.model.block_size % cfg.distributed.cp_size == 0, "block_size must be divisible by cp_size for Context Parallelism"
assert cfg.model.embed_dim % cfg.distributed.pp_size == 0, "embed_dim must be divisible by pp_size for Pipeline Parallelism"
assert cfg.model.num_heads % cfg.distributed.dp_size == 0, "num_heads must be divisible by dp_size for Data Parallelism"

print(f"TP: {cfg.distributed.tp_size}, CP: {cfg.distributed.cp_size}, PP: {cfg.distributed.pp_size}, DP: {cfg.distributed.dp_size}")
cmd = f"""
CUDA_DEVICE_MAX_CONNECTIONS=1 \\
{launch_mode} \\
    --nproc_per_node {word_size} \\
    train.py \\
        --config {CONFIG_FILE}
"""

print(cmd)
os.system(cmd)
