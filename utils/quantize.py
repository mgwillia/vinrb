import torch
from torch import nn
import copy

from .helpers import unwrap_model
from .grid import SparseGrid, FeatureGrid


_quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, SparseGrid, FeatureGrid)

def _is_quant_target(model, name, module):
    return name.startswith(model.bitstream_prefix) and not name.startswith(model.no_quant_prefix) and isinstance(module, _quant_target_cls)


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            weight_q = x
        else:
            weight = torch.tanh(x)
            weight_q = qfn.apply(weight, self.wbit)
        return weight_q
    

def _ste(x):
    """
    Straight-through estimator.
    """
    return (x.round() - x).detach() + x


def _quantize_ste(x, n, axis=None):
    """
    Per-channel & symmetric quantization with STE.
    """
    quant_range = 2. ** n - 1.
    x_max = abs(x).max(dim=axis, keepdim=True)[0] if axis is not None else abs(x).max()
    x_scale = 2 * x_max / quant_range + 1e-6
    x_q = _ste(x / x_scale).clamp(-2**(n - 1), 2**(n - 1) - 1)
    return x_q, x_scale


class QuantNoise(nn.Module):
    """
    Quant-Noise with optional STE.
    """
    def __init__(self, bitwidth, noise_ratio, ste, axis):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))
        self.register_buffer('noise_ratio', torch.tensor(noise_ratio, dtype=torch.float32))
        self.ste = ste
        self.axis = axis

    def extra_repr(self):
        s = 'ste={ste}, axis={axis}'
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:
            x_q, x_scale = _quantize_ste(x, self.bitwidth, self.axis)
            x_q = x_q if self.ste else x_q.detach()
            x_qr = x_q.to(x.dtype) * x_scale
            mask = (torch.rand_like(x) > self.noise_ratio).to(x.dtype)
            return x * mask + x_qr * (1. - mask)
        else:
            return x


def set_quantization(model, quant_level, quant_noise, quant_ste):
    """
    Set quantization for the model.
    """
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, QuantNoise):
                v.bitwidth.copy_(quant_level)
                v.noise_ratio.copy_(quant_noise)
                v.ste = quant_ste


def compute_best_quant_axis(x, thres=0.05):
    """
    Compute the best quantization axis for a tensor. 
    Similar to the one used in HNeRV quantization: https://github.com/haochen-rye/HNeRV/blob/main/hnerv_utils.py#L26
    """
    best_axis = None
    best_axis_dim = 0
    for axis in range(x.ndim):
        dim = x.shape[axis]
        if x.numel() / dim >= x.numel() * thres:
            continue
        if dim > best_axis_dim:
            best_axis = axis
            best_axis_dim = dim
    return best_axis


def init_quantization(model):
    """
    Initialize quantization for the model.
    """
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if _is_quant_target(model, k, v):
                quant_layer = QuantNoise(bitwidth=8, noise_ratio=0., ste=False, axis=compute_best_quant_axis(v.weight))
                quant_layer.to(list(v.parameters())[0].device)
                torch.nn.utils.parametrize.register_parametrization(v, 'weight', quant_layer)


def _quant_tensor(x, quant_level):
    """
    Quantize a tensor.
    """
    axis = compute_best_quant_axis(x)
    with torch.no_grad():
        x_q, x_scale = _quantize_ste(x, quant_level, axis)
        x_q = x_q.to(torch.int32)
        x_qr = x_q.to(x.dtype) * x_scale

    # Return the quantisied tensors (both int/float) and config
    meta = {
        'axis': axis,
        'scale': x_scale.half()
    }

    return x_q, x_qr, meta


def quant_model(model, quant_level):
    """
    Quantize the full model.
    """
    model = unwrap_model(model)

    # Get the quantisation targets
    excluded_keys = set()

    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix):
            excluded_keys.add(k)

    # Quantize
    qr_state_dict = copy.deepcopy(model.state_dict())
    q_state_dict = {}
    q_config = {'quant_level': quant_level}
    if quant_level == 32:
        pass
    elif quant_level <= 16:
        for k, v in model.state_dict().items():
            if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                q_state_dict[k], qr_state_dict[k], q_config[k] = \
                    _quant_tensor(v, quant_level)
    else:
        raise ValueError

    # Load quantisied state dict
    model.load_state_dict(qr_state_dict)

    # Ruturn configs
    return q_state_dict, q_config
