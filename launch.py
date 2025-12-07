from omegaconf import OmegaConf
import os


CONFIG_FILE = "configs/gpt2.yaml"

cfg = OmegaConf.load(CONFIG_FILE)

word_size = (
    cfg.distributed.tp_size
    * cfg.distributed.cp_size
    * cfg.distributed.pp_size
    * cfg.distributed.dp_size
)


cmd = f"""
CUDA_DEVICE_MAX_CONNECTIONS=1 \\
torchrun \\
    --nproc_per_node {word_size} \\
    train.py \\
        --config {CONFIG_FILE}
"""

print(cmd)
os.system(cmd)
