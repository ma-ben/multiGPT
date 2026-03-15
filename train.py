"""Training script for LLaMA model.
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --config tmp/fast_benchmark/120M_model_tiny_stories_dp=4.json
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --config tmp/dummy/llama2_7b_benchmark.json
"""
import os
import inspect
from contextlib import nullcontext
from omegaconf import OmegaConf
import time
import datetime
import argparse
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch.optim import AdamW

from picotron.context_parallel.context_parallel import apply_context_parallel
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
import picotron.process_group_manager as pgm
from picotron.utils import average_loss_across_dp_cp_ranks, set_all_seed, print, to_readable_format
from picotron.checkpoint import init_model_with_dematerialized_weights, init_model_with_materialized_weights
from picotron.data import RandomMicroBatchDataLoader
from picotron.process_group_manager import setup_process_group_manager
from picotron.pipeline_parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from picotron.data_parallel.data_parallel import DataParallelBucket
from picotron.model.gpt2 import GPT
from picotron.profiling import profile_range
import wandb


def build_torch_profiler(args, device, global_rank):
    if not args.enable_torch_profiler:
        return None
    if args.profile_ranks == "rank0" and global_rank != 0:
        return None

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    rank_dir = os.path.join(args.profiler_dir, f"rank{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=args.profiler_wait,
            warmup=args.profiler_warmup,
            active=args.profiler_active,
            repeat=args.profiler_repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            rank_dir,
            worker_name=f"rank{global_rank}",
        ),
        record_shapes=args.profiler_record_shapes,
        profile_memory=args.profiler_profile_memory,
        with_stack=args.profiler_with_stack,
        with_flops=args.profiler_with_flops,
    )


def maybe_toggle_cuda_profiler(args, device, step, started):
    if not args.enable_cuda_profiler_api or device.type != "cuda":
        return started

    if not started and step == args.cuda_profiler_start_step:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        return True

    if started and step == args.cuda_profiler_stop_step:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
        return False

    return started


def train_step(model, data_loader, device):
    acc_loss = 0.0

    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    for i in range(data_loader.grad_acc_steps):
        with profile_range("train/microbatch"):
            batch = next(data_loader)
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            if requires_grad_sync:
                model.require_backward_grad_sync = (i == data_loader.grad_acc_steps - 1)

            with profile_range("train/forward"):
                outputs = model(input_ids=input_ids)

            batch_size, seq_len = input_ids.shape
            target_ids = target_ids.reshape(-1)
            outputs = outputs.view(seq_len * batch_size, -1)

            with profile_range("train/loss"):
                loss = F.cross_entropy(outputs, target_ids, reduction='mean') / data_loader.grad_acc_steps

            with profile_range("train/backward"):
                loss.backward()

            acc_loss += loss.item()

    return acc_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    parser.add_argument("--total_train_steps_override", type=int, default=None, help="Override total_train_steps from config")

    parser.add_argument("--enable_torch_profiler", action="store_true", help="Enable torch profiler")
    parser.add_argument("--profiler_dir", type=str, default="profiles/torch", help="Torch profiler output dir")
    parser.add_argument("--profiler_wait", type=int, default=1, help="Profiler schedule wait steps")
    parser.add_argument("--profiler_warmup", type=int, default=1, help="Profiler schedule warmup steps")
    parser.add_argument("--profiler_active", type=int, default=4, help="Profiler schedule active steps")
    parser.add_argument("--profiler_repeat", type=int, default=1, help="Profiler schedule repeat count")
    parser.add_argument("--profiler_record_shapes", action="store_true", help="Enable record_shapes in torch profiler")
    parser.add_argument("--profiler_profile_memory", action="store_true", help="Enable profile_memory in torch profiler")
    parser.add_argument("--profiler_with_stack", action="store_true", help="Enable with_stack in torch profiler")
    parser.add_argument("--profiler_with_flops", action="store_true", help="Enable with_flops in torch profiler")
    parser.add_argument("--profile_ranks", type=str, choices=["rank0", "all"], default="rank0", help="Which ranks run torch profiler")

    parser.add_argument("--enable_cuda_profiler_api", action="store_true", help="Enable cudaProfilerStart/Stop markers for Nsight Systems")
    parser.add_argument("--cuda_profiler_start_step", type=int, default=2, help="Step index to start CUDA profiler capture")
    parser.add_argument("--cuda_profiler_stop_step", type=int, default=8, help="Step index to stop CUDA profiler capture")

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.total_train_steps_override is not None:
        config.training.total_train_steps = args.total_train_steps_override

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    set_all_seed(config.training.seed)
    os.environ["OMP_NUM_THREADS"] = config.environment.OMP_NUM_THREADS
    os.environ["TOKENIZERS_PARALLELISM"] = config.environment.TOKENIZERS_PARALLELISM

    if config.distributed.use_cpu:
        backend = "gloo"
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        backend = "nccl"
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

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

    is_wandb_rank = (
        pgm.process_group_manager.tp_rank == 0
        and pgm.process_group_manager.dp_rank == 0
        and pgm.process_group_manager.cp_rank == 0
        and pgm.process_group_manager.pp_is_last_stage
    )

    print("initializing data loader...", is_print_rank=is_wandb_rank)
    data_loader = RandomMicroBatchDataLoader(
        micro_batch_size=config.training.micro_batch_size,
        seq_length=config.model.block_size,
        grad_acc_steps=config.training.gradient_accumulation_steps,
        device=device,
        vocab_size=config.model.vocab_size,
    )
    dist.barrier()

    tokens_per_step = data_loader.global_batch_size * config.model.block_size
    if pgm.process_group_manager.global_rank == 0:
        print("Tokens per step:", to_readable_format(tokens_per_step), is_print_rank=is_wandb_rank)

    if is_wandb_rank and config.logging.use_wandb:
        wandb.init(
            project="picotron",
            name=f"{config.logging.run_name}_{to_readable_format(tokens_per_step)}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                "context_parallel_size": pgm.process_group_manager.cp_world_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
                "data_parallel_size": pgm.process_group_manager.dp_world_size,
                "model": config.model.name,
                "dataset": "random_tokens",
                "max_tokens": config.training.max_tokens,
                "learning_rate": config.training.learning_rate,
                "seed": config.training.seed,
                "micro_batch_size": data_loader.micro_batch_size,
                "global_batch_size": data_loader.global_batch_size,
                "gradient_accumulation": data_loader.grad_acc_steps,
            },
        )

    if pgm.process_group_manager.global_rank == 0:
        print(f"rank {pgm.process_group_manager.global_rank}: Creating model config")
        model_config = config.model
        objects = [model_config]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0, device=device)
    model_config = objects[0]
    print(
        f"rank {pgm.process_group_manager.global_rank}: Broadcasting model_config to all ranks",
        is_print_rank=pgm.process_group_manager.global_rank == 0,
    )
    dist.barrier()

    print(f"rank {pgm.process_group_manager.global_rank}: Initializing model meta device", is_print_rank=is_wandb_rank)
    with init_model_with_dematerialized_weights():
        model = GPT(
            vocab_size=model_config.vocab_size,
            block_size=model_config.block_size,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
        )

        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)
        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    model = init_model_with_materialized_weights(model, model_config)
    if pgm.process_group_manager.cp_world_size > 1:
        model = apply_context_parallel(model)
    model.to(dtype).to(device)
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)

    model.train()
    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, model_config.embed_dim)
    trained_tokens, step = 0, 0

    if config.training.use_fused_adam:
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == "cuda"
        extra_args = dict(fused=use_fused)
    else:
        extra_args = dict()

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, **extra_args)
    dist.barrier()

    torch_profiler = build_torch_profiler(args, device, global_rank)
    profiler_context = torch_profiler if torch_profiler is not None else nullcontext()
    cuda_profiler_started = False

    with profiler_context as profiler:
        while config.training.max_tokens is None or trained_tokens < config.training.max_tokens:
            cuda_profiler_started = maybe_toggle_cuda_profiler(args, device, step, cuda_profiler_started)

            step_start_time = time.time()
            with profile_range("train/zero_grad"):
                optimizer.zero_grad()

            with profile_range("train/forward_backward"):
                if pgm.process_group_manager.pp_world_size > 1:
                    if config.distributed.pp_engine == "afab":
                        loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype)
                    elif config.distributed.pp_engine == "1f1b":
                        loss = train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype)
                    else:
                        raise ValueError(f"Invalid pipeline parallel engine: {config.distributed.pp_engine}")
                else:
                    loss = train_step(model, data_loader, device)

            with profile_range("train/loss_reduce"):
                loss = average_loss_across_dp_cp_ranks(loss, device)

            with profile_range("train/optimizer_step"):
                optimizer.step()

            trained_tokens += tokens_per_step
            step += 1

            if hasattr(model, "reset"):
                model.reset()

            step_duration = time.time() - step_start_time
            tokens_per_second = tokens_per_step / step_duration
            tokens_per_second_per_gpu = tokens_per_second / world_size
            memory_gb = torch.cuda.memory_reserved() / 1e9 if device.type == "cuda" else 0.0

            if is_wandb_rank:
                print(
                    f"[rank {pgm.process_group_manager.global_rank}] "
                    f"Step: {step:<5d} | "
                    f"Loss: {loss:6.4f} | "
                    f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                    f"Tokens/s: {to_readable_format(tokens_per_second):>7s} | "
                    f"Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                    f"Tokens: {to_readable_format(trained_tokens):>7s}{('/' + to_readable_format(config.training.max_tokens)) if config.training.max_tokens else ''} | "
                    f"Memory usage: {memory_gb:6.2f}GB",
                    is_print_rank=is_wandb_rank,
                )

                if config.logging.use_wandb:
                    wandb.log(
                        {
                            "loss": loss,
                            "tokens_per_step": tokens_per_step,
                            "tokens_per_second": tokens_per_step / step_duration,
                            "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                            "memory_usage": memory_gb,
                            "trained_tokens": trained_tokens,
                        }
                    )

            if profiler is not None:
                profiler.step()

            if step >= config.training.total_train_steps:
                break

    if cuda_profiler_started and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()

    if is_wandb_rank and config.logging.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
