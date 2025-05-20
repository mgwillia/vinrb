import torch.nn as nn
import torch.nn.functional as F

CONV3D_FLOPS = 0

def get_flops_post_iter():
    global CONV3D_FLOPS
    tmp = CONV3D_FLOPS
    CONV3D_FLOPS = 0
    return tmp

def conv3d_flops_hook(module, input, output):
    input_tensor = input[0]
    batch_sz, in_chans, in_d, in_h, in_w = input_tensor.shape
    out_chans, out_d, out_h, out_w = output.shape[1:]

    out_chans //= module.groups
    
    ks = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
    
    # *2, for mul and add
    flops_per_group = 2 * ks * in_chans * out_chans * out_d * out_h * out_w
    flops = flops_per_group * batch_sz

    if module.bias is not None:
        bias_flops = out_chans * out_d * out_h * out_w * batch_sz
        flops += bias_flops

    global CONV3D_FLOPS 
    CONV3D_FLOPS += flops

def conv3d_flops_backward_hook(module, grad_input, grad_output):
    input_tensor = grad_input[0]
    out_tensor = grad_output[0]

    if input_tensor is not None and out_tensor is not None:
        batch_sz, in_chans, in_d, in_h, in_w = input_tensor.shape
        out_chans, out_d, out_h, out_w = out_tensor.shape[1:]

        out_chans //= module.groups
        
        ks = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]

        # gradient wrt output
        flops_input = 2 * ks * in_chans * out_chans * out_d * out_h * out_w
        flops_input *= batch_sz

        # gradient wrt weight
        flops_weight = 2 * ks * in_chans * out_chans * in_d * in_h * in_w
        flops_weight *= batch_sz

        # flops to apply weight
        flops_apply = out_chans * in_chans * ks

        flops_bias = 0
        if module.bias is not None:
            # *2 for gradient wrt bias and applying bias
            flops_bias = 2 * out_chans * out_d * out_h * out_w * batch_sz

        global CONV3D_FLOPS
        CONV3D_FLOPS += flops_input + flops_weight + flops_apply + flops_bias

def register_conv3d_flops_hook(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv3d):
            layer.register_forward_hook(conv3d_flops_hook)
            layer.register_full_backward_hook(conv3d_flops_backward_hook)