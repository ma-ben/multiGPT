"""
训练入口。

这个版本刻意围绕 DO.md / TARGET 里的主干路线做了几件“少量但关键”的优化：
1. 修正启动与训练闭环上的基础正确性问题；
2. 把 train step 拆成可观察的阶段，并给 profiler 打标签；
3. 接入轻量级 checkpoint save/load，保证主干能恢复；
4. 保留代码总量克制，不把仓库直接推成重框架。
"""

import argparse
import contextlib
import datetime
import inspect
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.profiler import record_function

import picotron.process_group_manager as pgm
from picotron.checkpoint import CheckpointManager, init_model_with_dematerialized_weights, init_model_with_materialized_weights
from picotron.context_parallel.context_parallel import apply_context_parallel
from picotron.data import MicroBatchDataLoader, RandomMicroBatchDataLoader
from picotron.data_parallel.data_parallel import DataParallelBucket
from picotron.model.gpt2 import GPT
from picotron.monitor import StepTimer, profiling_context
from picotron.pipeline_parallel.pipeline_parallel import PipelineParallel, train_step_pipeline_1f1b, train_step_pipeline_afab
from picotron.process_group_manager import setup_process_group_manager
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from picotron.utils import (
    average_loss_across_dp_cp_ranks,
    get_memory_usage_gb,
    print,
    set_all_seed,
    to_readable_format,
)


def get_autocast_context(model):
    """
    统一构造 autocast 上下文。

    这里把是否开启 autocast、用什么 dtype 的决定权交给模型运行时属性，
    这样 PP / 非 PP 路径都可以复用同一套判断逻辑。
    """
    use_autocast = getattr(model, "runtime_use_autocast", False)
    autocast_dtype = getattr(model, "runtime_autocast_dtype", None)
    if not use_autocast:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


def train_step(model, data_loader, device):
    """
    非 PP 路径的单 step 计算。

    这里把 micro-batch 内部过程切成 data / forward / loss / backward 四段，
    是为了后面用 profiler 时能直接看到：
    1. 数据准备是不是瓶颈；
    2. 前向和反向各占多少时间；
    3. gradient accumulation 期间同步到底发生在哪个 micro-batch。
    """
    accumulated_loss = 0.0
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1

    for micro_batch_idx in range(data_loader.grad_acc_steps):
        with record_function("train/load_micro_batch"):
            batch = next(data_loader)
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

        # 只有最后一个 micro-batch 才允许做 DP/CP 梯度同步。
        # 这样能避免 gradient accumulation 时反复 all-reduce。
        if requires_grad_sync:
            model.require_backward_grad_sync = (micro_batch_idx == data_loader.grad_acc_steps - 1)

        with get_autocast_context(model):
            with record_function("train/forward"):
                outputs = model(input_ids=input_ids, position_ids=batch["position_ids"].to(device))

            with record_function("train/loss"):
                batch_size, seq_len = input_ids.shape
                target_ids = target_ids.reshape(-1)
                outputs = outputs.view(seq_len * batch_size, -1)
                loss = F.cross_entropy(outputs, target_ids, reduction="mean") / data_loader.grad_acc_steps

        with record_function("train/backward"):
            loss.backward()

        accumulated_loss += loss.item()

    return accumulated_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2.yaml", help="Path to config file")
    return parser.parse_args()


def resolve_cuda_device_and_dtype(local_rank):
    """
    训练入口固定为 CUDA-only。

    这里集中做三件事：
    1. 确认当前进程看得到 CUDA；
    2. 把 local rank 绑定到对应 GPU；
    3. 统一决定训练 dtype。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("当前训练入口只支持 CUDA，未检测到可用 GPU。")

    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} 超出当前可见 GPU 数量 {device_count}。")

    backend = "nccl"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return backend, device, dtype


def build_data_loader(config, device, is_print_rank):
    """
    数据入口同时支持两种模式：
    1. random: 纯随机 token，适合并行机制和吞吐基线；
    2. hf: 真实 HuggingFace dataset，适合做训练闭环验证。

    为了控制代码量，这里不另外引入复杂的数据管线，只把已有的两个 loader 接到统一入口。
    """
    loader_mode = OmegaConf.select(config, "dataset.loader", default="random")
    print(f"初始化数据加载器，模式={loader_mode}", is_print_rank=is_print_rank)

    if loader_mode == "hf":
        return MicroBatchDataLoader(
            micro_batch_size=config.training.micro_batch_size,
            seq_length=config.model.block_size,
            dataset_name=config.dataset.name,
            tokenizer_name=config.model.tokenizer,
            num_workers=config.dataset.num_workers,
            num_proc=config.dataset.num_proc,
            grad_acc_steps=config.training.gradient_accumulation_steps,
            device=device,
            subset_name=config.dataset.subset_name,
            num_samples=config.training.num_samples,
        )

    return RandomMicroBatchDataLoader(
        micro_batch_size=config.training.micro_batch_size,
        seq_length=config.model.block_size,
        grad_acc_steps=config.training.gradient_accumulation_steps,
        device=device,
        vocab_size=config.model.vocab_size,
    )


def build_model(config, dtype, device, is_print_rank):
    """
    模型初始化顺序刻意保持为：
    meta init -> 并行包装 -> materialize -> move to device/dtype -> DP 包装

    这是训练系统里很常见的内存友好初始化顺序，尤其在模型变大以后更重要。
    """
    print("开始初始化模型（meta -> parallel wrapper -> materialize）", is_print_rank=is_print_rank)
    model_config = config.model
    attention_backend = model_config.get("attention_backend", "eager") if hasattr(model_config, "get") else "eager"
    activation_checkpointing = bool(model_config.get("activation_checkpointing", False)) if hasattr(model_config, "get") else False
    with init_model_with_dematerialized_weights():
        model = GPT(
            vocab_size=model_config.vocab_size,
            block_size=model_config.block_size,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            attention_backend=attention_backend,
            activation_checkpointing=activation_checkpointing,
        )

        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)
        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    model = init_model_with_materialized_weights(model, model_config)

    if pgm.process_group_manager.cp_world_size > 1:
        model = apply_context_parallel(model)

    model = model.to(device=device, dtype=dtype)

    # 小规模“最佳实践”里，一个很重要的分叉是：
    # - 并行包装复杂时，先保证语义稳定，不强上 compile；
    # - 单卡 dense 路径下，再去追 `torch.compile` 的收益。
    if bool(config.training.get("use_torch_compile", False)) if hasattr(config.training, "get") else False:
        compile_is_safe = (
            pgm.process_group_manager.tp_world_size == 1
            and pgm.process_group_manager.cp_world_size == 1
            and pgm.process_group_manager.pp_world_size == 1
            and pgm.process_group_manager.dp_world_size == 1
            and hasattr(torch, "compile")
        )
        if compile_is_safe:
            compile_mode = config.training.get("torch_compile_mode", "reduce-overhead")
            print(f"启用 torch.compile，mode={compile_mode}", is_print_rank=is_print_rank)
            model = torch.compile(model, mode=compile_mode)
        else:
            print("跳过 torch.compile：当前仅建议在单卡 CUDA dense 路径开启", is_print_rank=is_print_rank)

    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)

    runtime_use_autocast = bool(config.training.get("use_autocast", False)) if hasattr(config.training, "get") else False
    # 运行时标志挂在模型对象上，便于 train_step / PP 调度直接读取。
    model.runtime_use_autocast = runtime_use_autocast
    model.runtime_autocast_dtype = dtype

    return model


def build_optimizer(config, model):
    """
    fused AdamW 只在实现支持时打开。
    """
    extra_args = {}
    if config.training.use_fused_adam:
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        if fused_available:
            extra_args["fused"] = True
    return AdamW(model.parameters(), lr=config.training.learning_rate, **extra_args)


def resume_training_from_checkpoint(checkpoint_manager, model, optimizer, load_path, is_print_rank):
    """
    尽量用最少代码把恢复主干接起来：
    - 同拓扑 save/load；
    - 记录 trained_steps 和 trained_tokens。

    这一步不是完整 DCP，但足够让仓库从“只能从头跑”升级到“能恢复”。
    """
    print(f"从 checkpoint 恢复：{load_path}", is_print_rank=is_print_rank)
    trained_steps, trained_tokens = checkpoint_manager.load_checkpoint(model, optimizer, load_path)
    return trained_steps, trained_tokens


def log_rank_mapping():
    """
    每个 rank 打一行自身的并行坐标，方便排查 topology 问题。
    这是 T0 阶段非常实用的最小观测手段。
    """
    manager = pgm.process_group_manager
    print(
        f"rank-mapping | global={manager.global_rank} local={manager.local_rank} "
        f"dp={manager.dp_rank} pp={manager.pp_rank} cp={manager.cp_rank} tp={manager.tp_rank}"
    )


def init_wandb_run(config, data_loader):
    loader_mode = OmegaConf.select(config, "dataset.loader", default="random")
    dataset_name = config.dataset.name if loader_mode == "hf" else "random_tokens"
    wandb.init(
        project=config.logging.project_name,
        name=f"{config.logging.run_name}_{to_readable_format(data_loader.global_batch_size * config.model.block_size)}_{pgm.process_group_manager}",
        config={
            "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
            "context_parallel_size": pgm.process_group_manager.cp_world_size,
            "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
            "data_parallel_size": pgm.process_group_manager.dp_world_size,
            "model": config.model.name,
            "dataset": dataset_name,
            "max_tokens": config.training.max_tokens,
            "learning_rate": config.training.learning_rate,
            "seed": config.training.seed,
            "micro_batch_size": data_loader.micro_batch_size,
            "global_batch_size": data_loader.global_batch_size,
            "gradient_accumulation": data_loader.grad_acc_steps,
        },
    )


def run_one_training_step(config, model, data_loader, tensor_shapes, device, dtype):
    """
    按当前并行模式调度一个 step。
    把 PP 和非 PP 分支收口在这里，主循环就能更聚焦于“训练状态机”本身。
    """
    if pgm.process_group_manager.pp_world_size > 1:
        if config.distributed.pp_engine == "afab":
            return train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype)
        if config.distributed.pp_engine == "1f1b":
            return train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype)
        raise ValueError(f"Invalid pipeline parallel engine: {config.distributed.pp_engine}")
    return train_step(model, data_loader, device)


def save_checkpoint(checkpoint_manager, model, optimizer, save_dir, step, trained_tokens, is_print_rank):
    print(f"保存 checkpoint 到 {save_dir}", is_print_rank=is_print_rank)
    checkpoint_manager.save_checkpoint(model, optimizer, step, trained_tokens, save_dir)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    set_all_seed(config.training.seed)
    os.environ["OMP_NUM_THREADS"] = str(config.environment.OMP_NUM_THREADS)
    os.environ["TOKENIZERS_PARALLELISM"] = str(config.environment.TOKENIZERS_PARALLELISM)

    backend, device, dtype = resolve_cuda_device_and_dtype(local_rank)
    dist.init_process_group(
        rank=global_rank,
        world_size=world_size,
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(minutes=3),
    )

    setup_process_group_manager(
        tp_size=config.distributed.tp_size,
        cp_size=config.distributed.cp_size,
        pp_size=config.distributed.pp_size,
        dp_size=config.distributed.dp_size,
    )

    log_rank_mapping()

    is_logging_rank = (
        pgm.process_group_manager.tp_rank == 0
        and pgm.process_group_manager.dp_rank == 0
        and pgm.process_group_manager.cp_rank == 0
        and pgm.process_group_manager.pp_is_last_stage
    )
    is_profile_rank = pgm.process_group_manager.global_rank == 0

    data_loader = build_data_loader(config, device, is_logging_rank)
    dist.barrier()

    tokens_per_step = data_loader.global_batch_size * config.model.block_size
    if pgm.process_group_manager.global_rank == 0:
        print(f"Tokens per step: {to_readable_format(tokens_per_step)}")

    if is_logging_rank and config.logging.use_wandb:
        init_wandb_run(config, data_loader)

    model = build_model(config, dtype, device, is_logging_rank)
    optimizer = build_optimizer(config, model)
    checkpoint_manager = CheckpointManager()
    checkpoint_load_path = config.checkpoint.load_path
    if checkpoint_load_path:
        trained_steps, trained_tokens = resume_training_from_checkpoint(
            checkpoint_manager=checkpoint_manager,
            model=model,
            optimizer=optimizer,
            load_path=checkpoint_load_path,
            is_print_rank=is_logging_rank,
        )
    else:
        trained_steps, trained_tokens = 0, 0

    checkpoint_save_frequency = int(config.checkpoint.save_frequency)
    should_save_checkpoints = checkpoint_save_frequency > 0

    model.train()
    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, config.model.embed_dim)

    with profiling_context(config, is_profile_rank) as profiler:
        while config.training.max_tokens is None or trained_tokens < config.training.max_tokens:
            step_timer = StepTimer()
            step_start_time = time.time()

            with step_timer.track("step/zero_grad"):
                optimizer.zero_grad(set_to_none=True)

            with step_timer.track("step/compute"):
                loss = run_one_training_step(config, model, data_loader, tensor_shapes, device, dtype)

            with step_timer.track("step/loss_sync"):
                loss = average_loss_across_dp_cp_ranks(loss, device)

            with step_timer.track("step/optimizer_step"):
                optimizer.step()

            trained_tokens += tokens_per_step
            trained_steps += 1

            if hasattr(model, "reset"):
                with step_timer.track("step/runtime_reset"):
                    model.reset()

            if should_save_checkpoints and trained_steps % checkpoint_save_frequency == 0:
                save_dir = os.path.join(config.checkpoint.save_dir, f"step_{trained_steps}")
                save_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    model=model,
                    optimizer=optimizer,
                    save_dir=save_dir,
                    step=trained_steps,
                    trained_tokens=trained_tokens,
                    is_print_rank=is_logging_rank,
                )

            if profiler is not None:
                profiler.step()

            step_duration = time.time() - step_start_time
            tokens_per_second = tokens_per_step / step_duration
            tokens_per_second_per_gpu = tokens_per_second / world_size
            memory_usage_gb = get_memory_usage_gb(device)

            if is_logging_rank:
                print(
                    f"[rank {pgm.process_group_manager.global_rank}] "
                    f"Step: {trained_steps:<5d} | "
                    f"Loss: {loss:6.4f} | "
                    f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                    f"Tokens/s: {to_readable_format(tokens_per_second):>7s} | "
                    f"Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                    f"Tokens: {to_readable_format(trained_tokens):>7s}"
                    f"{('/' + to_readable_format(config.training.max_tokens)) if config.training.max_tokens else ''} | "
                    f"Memory usage: {memory_usage_gb:6.2f}GB | "
                    f"Timeline: {step_timer.format_summary()}",
                    is_print_rank=is_logging_rank,
                )

                if config.logging.use_wandb:
                    log_data = {
                        "loss": loss,
                        "tokens_per_step": tokens_per_step,
                        "tokens_per_second": tokens_per_second,
                        "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                        "memory_usage_gb": memory_usage_gb,
                        "trained_tokens": trained_tokens,
                        "trained_steps": trained_steps,
                    }
                    wandb.log(log_data)

            if trained_steps >= config.training.total_train_steps:
                break

    if is_logging_rank and config.logging.use_wandb:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
