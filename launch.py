import argparse
import shlex
import subprocess
import sys

from omegaconf import OmegaConf


DEFAULT_CONFIG_FILE = "configs/gpt2.yaml"


def validate_topology(cfg):
    """
    在真正启动分布式入口之前，先把最容易出错的拓扑约束挡在入口。

    这里故意只做“和当前代码实现强相关”的检查，不做过度设计：
    1. TP 需要 head / vocab 能整除；
    2. CP 需要序列长度能整除；
    3. PP 当前是按 layer 数切分，所以检查 num_layers 是否足够。
    """
    distributed = cfg.distributed
    model = cfg.model

    assert model.vocab_size % distributed.tp_size == 0, "vocab_size 必须能被 tp_size 整除"
    assert model.num_heads % distributed.tp_size == 0, "num_heads 必须能被 tp_size 整除"
    assert model.block_size % distributed.cp_size == 0, "block_size 必须能被 cp_size 整除"
    assert model.num_layers >= distributed.pp_size, "当前 PP 方案要求 num_layers >= pp_size"


def build_command(args, cfg):
    """
    构造启动命令。

    这里不再把 config 路径写死成默认值，而是严格使用命令行传入的路径。
    这是训练系统最基础的“单命令可复现”要求之一。
    """
    world_size = (
        cfg.distributed.tp_size
        * cfg.distributed.cp_size
        * cfg.distributed.pp_size
        * cfg.distributed.dp_size
    )

    launch_mode = [
        "debugpy-run",
        "-m",
        "torch.distributed.run",
        "-p",
        str(args.port_debug),
        "--",
    ] if args.debug else [sys.executable, "-m", "torch.distributed.run"]

    return [
        "env",
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        *launch_mode,
        "--master_port",
        str(args.master_port),
        "--nproc_per_node",
        str(world_size),
        "train.py",
        "--config",
        args.config,
    ]


def validate_runtime_environment(cfg, dry_run: bool):
    """
    当前启动链路只支持单机 CUDA 训练。

    dry-run 只打印命令，因此不强制要求本机此刻有可用 GPU。
    真正启动前则尽早校验，避免进到 torchrun 里才由各 rank 分别报错。
    """
    if dry_run:
        return

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前启动解释器未安装 PyTorch，无法执行 CUDA 训练。") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("当前训练入口只支持 CUDA，未检测到可用 GPU。")

    world_size = (
        cfg.distributed.tp_size
        * cfg.distributed.cp_size
        * cfg.distributed.pp_size
        * cfg.distributed.dp_size
    )
    device_count = torch.cuda.device_count()
    if device_count < world_size:
        raise RuntimeError(f"当前配置需要 {world_size} 张 GPU，但本机只检测到 {device_count} 张。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port_debug", type=int, default=5678)
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    validate_topology(cfg)
    validate_runtime_environment(cfg, dry_run=args.dry_run)

    print(
        "TP: {tp}, CP: {cp}, PP: {pp}, DP: {dp}".format(
            tp=cfg.distributed.tp_size,
            cp=cfg.distributed.cp_size,
            pp=cfg.distributed.pp_size,
            dp=cfg.distributed.dp_size,
        )
    )

    command = build_command(args, cfg)
    printable_command = " ".join(shlex.quote(part) for part in command)
    print(printable_command)

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
