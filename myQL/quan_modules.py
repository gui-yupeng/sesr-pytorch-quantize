import torch
from torch import nn
import torch.nn.functional as F
from myQL.quan_func import 	quantize_model_weight, \
							quantize_asymmetrical_by_tensor, \
							reshape_input_for_hardware_pe, \
							PEs_and_bias_adder,\
							requan_conv2d_output
from define import  WEIGHT_W_FLG, QUAN_BIT, WEIGHT_W_HIST_PNG, INPUT_W_HIST_PNG, \
                    OUTPUT_PE_W_FLG, OUTPUT_PE_ADD_W_FLG, INPUT_W_FLG, BIAS_W_FLG,\
                    REQUAN_BIT, REQUAN_N_MAX, BIAS_QUAN_W_FLG, REQUAN_FACTOR_W_FLG,\
                    PE, BIAS_BIT, PE_ACC_BIT, PE_ADD_BIT

class myMinMaxObserver(nn.Module):
    """To record the max/min value as register in network"""
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))

# 自写卷积层
def conv_forward_naive(x, w, stride, pad_num, pad_value):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N,C,H,W = x.shape
    F,_,HH,WW = w.shape
    S = stride
    P = pad_num
    P_val = pad_value
    Ho = 1 + (H + 2 * P - HH) / S
    Wo = 1 + (W + 2 * P - WW) / S
    Ho = int(Ho)
    Wo = int(Wo)
    x_pad = torch.full((N,C,H+2*P,W+2*P), fill_value= P_val).cuda()
    x_pad[:,:,P:P+H,P:P+W]=x
    #x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')
    out = torch.zeros((N,F,Ho,Wo)).cuda()
 
    for f in range(F):
      for i in range(Ho):
        for j in range(Wo):
          # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
          out[:,f,i,j] = torch.sum(x_pad[:, :, i*S : i*S+HH, j*S : j*S+WW] * w[f, :, :, :], dim=(1, 2, 3)) 
 
    #   out[:,f,:,:]+=b[f]
#     cache = (x, w, b, conv_param)
#     return out, cache
    return out

def sesr_forward_sim(input):
    """
    模拟SESR网络的前向过程
    sesr(
    (conv_first): CollapsibleLinearBlock(
    (conv_expand): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (conv_squeeze): Identity()
    (activation): ReLU()
    )
    (residual_block): Sequential(
    (0): ResidualCollapsibleLinearBlock(
      (conv_expand): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_squeeze): Identity()
      (activation): ReLU()
    )
    (1): ResidualCollapsibleLinearBlock(
      (conv_expand): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_squeeze): Identity()
      (activation): ReLU()
    )
    (2): ResidualCollapsibleLinearBlock(
      (conv_expand): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_squeeze): Identity()
      (activation): ReLU()
    )
    )
    (add_residual): AddOp()
    (conv_last): CollapsibleLinearBlock(
    (conv_expand): Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (conv_squeeze): Identity()
    (activation): Identity()
    )
    (depth_to_space): PixelShuffle(upscale_factor=4)
    )
    """
    exe_mode = 1

    func_id = 0
    input_0 = input
    input_0 = quantize_asymmetrical_by_tensor(tensor_input=input_0, width=QUAN_BIT, exe_mode=exe_mode, func_id=func_id)
    weight_0 = torch.load("output_pt/weight/conv.weight.{}.pt".format(func_id))
    bias_0 = torch.load("output_pt/bias/conv.bias.{}.pt".format(func_id)).cpu().numpy().tolist()
    input_0 = reshape_input_for_hardware_pe(input_0)
    output_0 = conv_forward_naive(x=input_0,w=weight_0, stride=1, pad_num=2, pad_value=0)
    output_0 = PEs_and_bias_adder(input_tensor=output_0, bias=bias_0, pe_acc_width=PE_ACC_BIT, pe_add_width= PE_ADD_BIT, 
                                       bias_width= BIAS_BIT, func_id=func_id, pe_num=4, exe_mode=exe_mode )
    output_0 = requan_conv2d_output(input_tensor=output_0, func_id=func_id,exe_mode=exe_mode)
    output_0 = F.relu(output_0)

    func_id = 1
    input_1 = output_0
    input_1 = quantize_asymmetrical_by_tensor(tensor_input=input_1, width=QUAN_BIT, exe_mode=exe_mode, func_id=func_id)
    weight_1 = torch.load("output_pt/weight/conv.weight.{}.pt".format(func_id))
    bias_1 = torch.load("output_pt/bias/conv.bias.{}.pt".format(func_id)).cpu().numpy().tolist()
    input_1 = reshape_input_for_hardware_pe(input_1)
    output_1 = conv_forward_naive(x=input_1,w=weight_1, stride=1, pad_num=1, pad_value=0)
    output_1 = PEs_and_bias_adder(input_tensor=output_1, bias=bias_1, pe_acc_width=PE_ACC_BIT, pe_add_width= PE_ADD_BIT, 
                                       bias_width= BIAS_BIT, func_id=func_id, pe_num=4, exe_mode=exe_mode )
    output_1 = requan_conv2d_output(input_tensor=output_1, func_id=func_id,exe_mode=exe_mode)
    output_1 = F.relu(output_1)

    func_id = 2
    input_2 = output_1
    input_2 = quantize_asymmetrical_by_tensor(tensor_input=input_2, width=QUAN_BIT, exe_mode=exe_mode, func_id=func_id)
    weight_2 = torch.load("output_pt/weight/conv.weight.{}.pt".format(func_id))
    bias_2 = torch.load("output_pt/bias/conv.bias.{}.pt".format(func_id)).cpu().numpy().tolist()
    input_2 = reshape_input_for_hardware_pe(input_2)
    output_2 = conv_forward_naive(x=input_2,w=weight_2, stride=1, pad_num=1, pad_value=0)
    output_2 = PEs_and_bias_adder(input_tensor=output_2, bias=bias_2, pe_acc_width=PE_ACC_BIT, pe_add_width= PE_ADD_BIT, 
                                       bias_width= BIAS_BIT, func_id=func_id, pe_num=4, exe_mode=exe_mode )
    output_2 = requan_conv2d_output(input_tensor=output_2, func_id=func_id,exe_mode=exe_mode)
    output_2 = F.relu(output_2)

    func_id = 3
    input_3 = output_2
    input_3 = quantize_asymmetrical_by_tensor(tensor_input=input_3, width=QUAN_BIT, exe_mode=exe_mode, func_id=func_id)
    weight_3 = torch.load("output_pt/weight/conv.weight.{}.pt".format(func_id))
    bias_3 = torch.load("output_pt/bias/conv.bias.{}.pt".format(func_id)).cpu().numpy().tolist()
    input_3 = reshape_input_for_hardware_pe(input_3)
    output_3 = conv_forward_naive(x=input_3,w=weight_3, stride=1, pad_num=1, pad_value=0)
    output_3 = PEs_and_bias_adder(input_tensor=output_3, bias=bias_3, pe_acc_width=PE_ACC_BIT, pe_add_width= PE_ADD_BIT, 
                                       bias_width= BIAS_BIT, func_id=func_id, pe_num=4, exe_mode=exe_mode )
    output_3 = requan_conv2d_output(input_tensor=output_3, func_id=func_id,exe_mode=exe_mode)
    output_3 = F.relu(output_3)

    func_id = 4
    input_4 = output_3
    input_4 = quantize_asymmetrical_by_tensor(tensor_input=input_4, width=QUAN_BIT, exe_mode=exe_mode, func_id=func_id)
    weight_4 = torch.load("output_pt/weight/conv.weight.{}.pt".format(func_id))
    bias_4 = torch.load("output_pt/bias/conv.bias.{}.pt".format(func_id)).cpu().numpy().tolist()
    input_4 = reshape_input_for_hardware_pe(input_4)
    output_4 = conv_forward_naive(x=input_4,w=weight_4, stride=1, pad_num=2, pad_value=0)
    output_4 = PEs_and_bias_adder(input_tensor=output_4, bias=bias_4, pe_acc_width=PE_ACC_BIT, pe_add_width= PE_ADD_BIT, 
                                       bias_width= BIAS_BIT, func_id=func_id, pe_num=4, exe_mode=exe_mode )
    output_4 = requan_conv2d_output(input_tensor=output_4, func_id=func_id,exe_mode=exe_mode)
    output_4 = F.relu(output_4)

    depth_to_space = nn.PixelShuffle(4)

    result = depth_to_space(output_4)
    return result
