import torch.distributed as dist
import torch
import picotron.process_group_manager as pgm
import torch.nn.functional as F

from typing import Tuple

from picotron.profiling import profile_range


def merge_first_two_dims(grad_output: torch.Tensor, input_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge the first two dimensions of tensors."""
    return grad_output.contiguous().view(-1, *grad_output.shape[2:]), input_.contiguous().view(-1, *input_.shape[2:])


def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension into num_partitions chunks."""
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)


class CopyToModelParallelRegion(torch.autograd.Function):
    """
    Copy in forward pass, all-reduce in backward pass.
    This is the `f` function in the paper: https://arxiv.org/abs/1909.08053
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output

        with profile_range("tp_comm/copy_to_mp_backward_all_reduce"):
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output


class ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    All-reduce in forward pass, identity in backward pass.
    This is the `g` function in the paper: https://arxiv.org/abs/1909.08053
    """

    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x

        with profile_range("tp_comm/reduce_from_mp_forward_all_reduce"):
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather in forward pass, split in backward pass."""

    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x

        last_dim = x.dim() - 1
        x = x.contiguous()

        tensor_list = [torch.empty_like(x) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = x

        with profile_range("tp_comm/gather_from_mp_all_gather"):
            dist.all_gather(tensor_list, x, group=pgm.process_group_manager.tp_group)

        with profile_range("tp_compute/gather_from_mp_concat"):
            output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output

        with profile_range("tp_compute/gather_from_mp_backward_split"):
            chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()


class LinearWithAsyncAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        with profile_range("tp_compute/linear_forward"):
            output = input_ @ weight.t() + bias if bias is not None else input_ @ weight.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        The key difference with "linear_with_all_reduce" is that the all reduce of input_ gradeint is before
        the calculation of the gradient of weights and bias, instead of after. So we can overlap the computation and communication.

        Before: grad_output -> grad_input, grad_weight, grad_bias  -> grad_input all reduce
        Now:    grad_output -> grad_input -> grad_input all reduce -> grad_weight, grad_bias
        """
        input_, weight = ctx.saved_tensors

        with profile_range("tp_compute/linear_backward_grad_input"):
            grad_input = grad_output @ weight

        with profile_range("tp_comm/linear_backward_grad_input_all_reduce_async"):
            input_gradient_all_reduce_handle = dist.all_reduce(
                grad_input,
                group=pgm.process_group_manager.tp_group,
                async_op=True,
            )

        with profile_range("tp_compute/linear_backward_prepare"):
            grad_output, input_ = merge_first_two_dims(grad_output, input_)

        with profile_range("tp_compute/linear_backward_grad_weight"):
            grad_weight = grad_output.t() @ input_

        with profile_range("tp_compute/linear_backward_grad_bias"):
            grad_bias = grad_output.sum(0) if ctx.use_bias else None

        with profile_range("tp_comm/linear_backward_grad_input_all_reduce_wait"):
            input_gradient_all_reduce_handle.wait()

        return grad_input, grad_weight, grad_bias


def linear_with_all_reduce(x, weight, bias):
    input_parallel = CopyToModelParallelRegion.apply(x)
    with profile_range("tp_compute/linear_with_all_reduce_forward"):
        output = F.linear(input_parallel, weight, bias)
    return output


def linear_with_async_all_reduce(x, weight, bias):
    return LinearWithAsyncAllReduce.apply(x, weight, bias)
