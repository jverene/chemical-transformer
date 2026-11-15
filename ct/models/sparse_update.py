"""Dense-forward FFN with configurable active-row-only backward matmuls."""
import torch
import torch.nn.functional as F


def _dropout_mask(shape, probability, training, device, dtype):
    if not training or probability == 0:
        return torch.ones(shape, device=device, dtype=dtype)
    return (torch.rand(shape, device=device) >= probability).to(dtype) / (1 - probability)


class SparseUpdateFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, active, first_weight, first_bias,
                second_weight, second_bias, dropout, training,
                dense_input_gradient, gradient_scale):
        first_input = x
        first_output = F.linear(first_input, first_weight, first_bias)
        hidden = F.gelu(first_output)
        first_dropout = _dropout_mask(
            hidden.shape, dropout, training, hidden.device, hidden.dtype)
        dropped_hidden = hidden * first_dropout
        second_output = F.linear(dropped_hidden, second_weight, second_bias)
        second_dropout = _dropout_mask(
            second_output.shape, dropout, training,
            second_output.device, second_output.dtype)
        output = second_output * second_dropout
        ctx.save_for_backward(
            first_input, first_output, dropped_hidden, first_dropout,
            second_dropout, active, first_weight, second_weight)
        ctx.dense_input_gradient = dense_input_gradient
        ctx.gradient_scale = gradient_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (first_input, first_output, dropped_hidden, first_dropout,
         second_dropout, active, first_weight, second_weight) = ctx.saved_tensors
        indices = active.nonzero(as_tuple=False).squeeze(-1)
        grad_output = grad_output * second_dropout
        active_output = grad_output.index_select(0, indices)
        active_hidden = dropped_hidden.index_select(0, indices)

        second_weight_grad = active_output.t() @ active_hidden
        second_bias_grad = active_output.sum(dim=0)
        variance = 0.5
        if ctx.dense_input_gradient:
            hidden_grad = (grad_output @ second_weight) * first_dropout
            cdf = 0.5 * (1 + torch.erf(first_output * (variance ** 0.5)))
            pdf = torch.exp(-0.5 * first_output.square()) * ((2 * torch.pi) ** -0.5)
            first_grad = hidden_grad * (cdf + first_output * pdf)
            active_first_grad = first_grad.index_select(0, indices)
            active_input = first_input.index_select(0, indices)
            first_weight_grad = active_first_grad.t() @ active_input
            first_bias_grad = active_first_grad.sum(dim=0)
            input_grad = first_grad @ first_weight
        else:
            hidden_grad = active_output @ second_weight
            hidden_grad = hidden_grad * first_dropout.index_select(0, indices)
            active_first = first_output.index_select(0, indices)
            cdf = 0.5 * (1 + torch.erf(active_first * (variance ** 0.5)))
            pdf = torch.exp(-0.5 * active_first.square()) * ((2 * torch.pi) ** -0.5)
            first_grad = hidden_grad * (cdf + active_first * pdf)
            active_input = first_input.index_select(0, indices)
            first_weight_grad = first_grad.t() @ active_input
            first_bias_grad = first_grad.sum(dim=0)
            input_grad = torch.zeros_like(first_input)
            input_grad[indices] = first_grad @ first_weight
        scale = ctx.gradient_scale
        return (input_grad * scale, None, first_weight_grad * scale,
                first_bias_grad * scale, second_weight_grad * scale,
                second_bias_grad * scale, None, None, None, None)


def sparse_update_ffn(x, active, ffn, training, dense_input_gradient=False,
                      gradient_scale=1.0):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    output = SparseUpdateFFN.apply(
        flat, active.reshape(-1), ffn[0].weight, ffn[0].bias,
        ffn[3].weight, ffn[3].bias, ffn[2].p, training,
        dense_input_gradient, gradient_scale)
    return output.view(shape)


class LateMaskSparseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, mask_state):
        ctx.save_for_backward(x, weight)
        ctx.mask_state = mask_state
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        active = ctx.mask_state.get("active")
        if active is None:
            raise RuntimeError("late sparse-linear mask was not set before backward")
        indices = active.nonzero(as_tuple=False).squeeze(-1)
        active_output = grad_output.index_select(0, indices)
        active_input = x.index_select(0, indices)
        weight_grad = active_output.t() @ active_input
        bias_grad = active_output.sum(dim=0)
        input_grad = torch.zeros_like(x)
        input_grad[indices] = active_output @ weight
        return input_grad, weight_grad, bias_grad, None


def sparse_update_linear(x, weight, bias, mask_state):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    output = LateMaskSparseLinear.apply(flat, weight, bias, mask_state)
    return output.view(*shape[:-1], weight.shape[0])
