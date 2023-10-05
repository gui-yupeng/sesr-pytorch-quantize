import copy
import os
import math
import torch
from torch import Tensor
from torch import nn 
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from define import  WEIGHT_W_FLG, QUAN_BIT, WEIGHT_W_HIST_PNG, INPUT_W_HIST_PNG, \
                    OUTPUT_PE_W_FLG, OUTPUT_PE_ADD_W_FLG, INPUT_W_FLG, BIAS_W_FLG,\
                    REQUAN_BIT, REQUAN_N_MAX, BIAS_QUAN_W_FLG, REQUAN_FACTOR_W_FLG


def remove_suffix(string):
    parts = string.split(".")
    result = ".".join(parts[:-1])
    return result

def float_to_hex(item, bit_width):
    """example: float(127) to signed int hex(7f) string"""
    bit_hex = math.ceil(bit_width / 4)
    max = 2**bit_width
    item_int = int(item)
    
    if item_int < 0:
        int_res = max + item_int
    else:
        int_res = item_int

    if bit_hex == 3 :
        str_hex = '{:03x}'.format(int_res)
    elif bit_hex == 4 :
        str_hex = '{:04x}'.format(int_res)
    elif bit_hex == 5 :
        str_hex = '{:05x}'.format(int_res)
    else:
        str_hex = '{:02x}'.format(int_res)
    
    return str_hex

def quantize_symmetrical_by_tensor(tensor_input: torch.Tensor, width: int, exe_mode :int, func_id: int = None, filename : str = None) -> torch.Tensor:
    """
    exe mode 0: return tensor staged
    exe mode 1: return tensor quantized
    """
    #get input tensor hist
    if WEIGHT_W_HIST_PNG:
        store_path = "output_png/weight/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        plt.cla()
        plt.hist(tensor_input.reshape(-1).cpu().numpy(),bins=300)
        plt.savefig("output_png/weight/conv.weight.{}.png".format(func_id))

    max_val = torch.max(tensor_input).item()
    min_val = torch.min(tensor_input).item()

    tensor_abs_max = max(abs(max_val), abs(min_val))

    assert tensor_abs_max > 0 , "Conv2d weight tensor is all zero"

    inp_max = tensor_abs_max
    inp_min = 0 - tensor_abs_max
    quan_max = 2 ** (width - 1) - 1
    quan_min = 0 - 2 ** (width - 1)

    quan_scale = (inp_max - inp_min) / (quan_max - quan_min)
    tensor_quan = torch.clamp(torch.round(tensor_input / quan_scale), min=quan_min, max=quan_max)

    #store the pt file
    store_path = "output_pt/weight/"
    if  not os.path.exists(store_path):#如果路径不存在
        os.makedirs(store_path)
    torch.save(quan_scale,"output_pt/weight/conv.weight.{}.scale.pt".format(func_id))
    torch.save(tensor_quan, "output_pt/weight/conv.weight.{}.pt".format(func_id))
    # torch.save(1,"output_pt/weight/conv.weight.{}.scale.pt".format(func_id))
    # torch.save(tensor_quan* quan_scale,"output_pt/weight/conv.weight.{}.pt".format(func_id))

    if WEIGHT_W_FLG is True :
        store_path = "output_txt/weight/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        store_txt_path = "output_txt/weight/conv.weight.{}.txt".format(func_id)
        oc_real = tensor_quan.size(0)
        ic_real = tensor_quan.size(1)
        kh_real = tensor_quan.size(2)
        kw_real = tensor_quan.size(3)

        oc = math.ceil(oc_real / 4) * 4
        ic = math.ceil(ic_real / 4) * 4
        kh = kh_real
        kw = kw_real

        tensor_store = torch.zeros(oc, ic, kh, kw)
        tensor_store[0:oc_real, 0:ic_real, 0:kh_real, 0:kw_real] = tensor_quan[0:oc_real, 0:ic_real, 0:kh_real, 0:kw_real]
        wr_line_num = oc * ic * kh * kw / 16

        with open(store_txt_path,"w") as f:
            f.write('{}\n'.format("{:02x}".format(int(wr_line_num))))
            for block_oc in range(0,oc,4):
                for block_ic in range(0,ic,4):
                    for kh_i in range(kh):
                        for kw_i in range(kw):
                            for ic_i in range(4):
                                for oc_i in range(4):
                                    data_item = tensor_store[block_oc+oc_i, block_ic+ic_i, kh_i, kw_i].item()
                                    f.write(float_to_hex(data_item, QUAN_BIT))
                            f.write('\n')
    
    if exe_mode == 0 :
        quantized_tensor = tensor_quan * quan_scale
    elif exe_mode == 1 :
        quantized_tensor = tensor_quan

    if WEIGHT_W_HIST_PNG:
        store_path = "output_png/weight_quan/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        plt.cla()
        plt.hist(quantized_tensor.reshape(-1).cpu().numpy(),bins=300)
        plt.savefig("output_png/weight_quan/conv.weightquan.{}.png".format(func_id))

    return quantized_tensor

def quantize_model_weight(model_input: nn.Module, weight_width: int, exe_mode: int) :
    """quantize the weight parameters in model"""
    model = copy.deepcopy(model_input)
    model_parameters = model.state_dict()
    model_modules_dict = dict(model.named_modules())

    module_type_to_quan = [nn.Conv2d]
    """weight quantizing only supports nn.Conv2d now"""
    conv_func_id = 0

    for param in model_parameters:
        # print(param,model_parameters[param].shape)
        module_name = remove_suffix(param)
        module_type = type(model_modules_dict[module_name])
        if module_type in module_type_to_quan:
            if 'weight' in param:
                if module_type in [nn.Conv2d]:
                    model_parameters[param] = quantize_symmetrical_by_tensor(
                        model_parameters[param], weight_width, exe_mode ,func_id=conv_func_id
                    )
                    conv_func_id = conv_func_id + 1
                else:
                    """if other module type in module_type_to_quan also has weight paramrters"""
                    pass
            elif 'bias' in param:
                """no operating for bias here"""
                pass
            else:
                raise KeyError('Unsupported state dict type found. (%s)' % param)
    #print(model_parameters.keys())
    model.load_state_dict(model_parameters)
    return model

def quantize_asymmetrical_by_tensor(tensor_input: torch.Tensor, width: int, exe_mode :int, func_id: int = None) -> torch.Tensor:
    """
    exe mode 0: return tensor staged
    exe mode 1: return tensor quantized
    """
    #get input tensor hist
    if INPUT_W_HIST_PNG:
        store_path = "output_png/input/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        plt.cla()
        plt.hist(tensor_input.reshape(-1).cpu().numpy(),bins=100)
        plt.savefig("output_png/input/input.{}.png".format(func_id))

    if exe_mode == 0 :
        #校准
        max_val = torch.max(tensor_input).item()
        min_val = torch.min(tensor_input).item()

        #MinMax观察器
        store_path = "output_pt/input/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        if  not os.path.isfile("output_pt/input/input.{}.max_val.pt".format(func_id)):
            torch.save(max_val,"output_pt/input/input.{}.max_val.pt".format(func_id))
            torch.save(min_val,"output_pt/input/input.{}.min_val.pt".format(func_id))
        else:
            last_max_val = torch.load("output_pt/input/input.{}.max_val.pt".format(func_id))
            last_min_val = torch.load("output_pt/input/input.{}.min_val.pt".format(func_id))
            if last_max_val < max_val :
                torch.save(max_val,"output_pt/input/input.{}.max_val.pt".format(func_id))
                last_max_val = max_val
            if last_min_val > min_val :
                torch.save(min_val,"output_pt/input/input.{}.min_val.pt".format(func_id))
                last_min_val = min_val

        assert max_val != min_val , "Input tensor is all equal"

        inp_max = max_val
        inp_min = min_val
        # 融入一些校准误差
        # inp_max = last_max_val
        # inp_min = last_min_val
        quan_max = 2 ** (width - 1) - 1
        quan_min = 0 - 2 ** (width - 1)

        quan_scale = (inp_max - inp_min) / (quan_max - quan_min)
        quan_zero = quan_min - round(inp_min/quan_scale)

        tensor_quan = torch.clamp(torch.round(tensor_input / quan_scale + quan_zero), min=quan_min, max=quan_max)

        #store the pt file

        torch.save(quan_scale,"output_pt/input/input.{}.scale.pt".format(func_id))
        torch.save(quan_zero,"output_pt/input/input.{}.zero.pt".format(func_id))


        quantized_tensor = (tensor_quan - quan_zero) * quan_scale

    elif exe_mode == 1 :
        # if func_id == 4:
        #     residual = torch.load("output_pt/residual/shortcut_tensor.pt")
        #     tensor_add = torch.round(residual) + torch.round(tensor_input)
        #     quan_max = 2 ** (width - 1) - 1
        #     quan_min = 0 - 2 ** (width - 1)

        #     scale_1 = torch.load("output_pt/input/input.{}.scale.pt".format(1))

        #     res_add_requan = scale_1 / scale
        #     res_add_requan_16bit, res_add_requan_n = quan_layer_between_const(res_add_requan, REQUAN_BIT, REQUAN_N_MAX)
        #     tensor_add = tensor_add * res_add_requan_16bit * (2**(0-res_add_requan_n))
        #     quantized_tensor = torch.clamp(torch.round(tensor_add + zero), min=quan_min, max=quan_max)
        # else:
        #     max_val = torch.max(tensor_input).item()
        #     min_val = torch.min(tensor_input).item()

        #     assert max_val != min_val , "Input tensor is all equal"

        #     inp_max = max_val
        #     inp_min = min_val
        #     quan_max = 2 ** (width - 1) - 1
        #     quan_min = 0 - 2 ** (width - 1)

        #     quan_scale = (inp_max - inp_min) / (quan_max - quan_min)
        #     quan_zero = quan_min - round(inp_min/quan_scale)

        #     tensor_quan = torch.clamp(torch.round(tensor_input / quan_scale + quan_zero), min=quan_min, max=quan_max)

        #     #store the pt file

        #     torch.save(quan_scale,"output_pt/input/input.{}.scale.pt".format(func_id))
        #     torch.save(quan_zero,"output_pt/input/input.{}.zero.pt".format(func_id))

        if func_id == 0:
            #只有第一层需要量化，其余层做requantize
            scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id))
            zero = torch.load("output_pt/input/input.{}.zero.pt".format(func_id))
            quan_max = 2 ** (width - 1) - 1
            quan_min = 0 - 2 ** (width - 1)
            quantized_tensor = torch.clamp(torch.round(tensor_input / scale + zero), min=quan_min, max=quan_max)

        elif func_id == 4:
            #残差
            residual = torch.load("output_pt/residual/shortcut_tensor.pt")
            tensor_add = torch.round(residual) + torch.round(tensor_input)

            quan_max = 2 ** (width - 1) - 1
            quan_min = 0 - 2 ** (width - 1)
            scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id))
            scale_1 = torch.load("output_pt/input/input.{}.scale.pt".format(1))
            zero = torch.load("output_pt/input/input.{}.zero.pt".format(func_id))
            res_add_requan = scale_1 / scale
            res_add_requan_16bit, res_add_requan_n = quan_layer_between_const(res_add_requan, REQUAN_BIT, REQUAN_N_MAX)

            if REQUAN_FACTOR_W_FLG :
                store_path = "output_pt/requan_factor/"
                if  not os.path.exists(store_path):#如果路径不存在
                    os.makedirs(store_path)
                torch.save(res_add_requan_16bit,"output_pt/requan_factor/requan_res.pt")
                torch.save(res_add_requan_n,"output_pt/requan_factor/n_res.pt")

            tensor_add = tensor_add * res_add_requan_16bit * (2**(0-res_add_requan_n))
            quantized_tensor = torch.clamp(torch.round(tensor_add + zero), min=quan_min, max=quan_max)

        else:
            #重量化后，加上zero
            quan_max = 2 ** (width - 1) - 1
            quan_min = 0 - 2 ** (width - 1)
            scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id))
            zero = torch.load("output_pt/input/input.{}.zero.pt".format(func_id))
            quantized_tensor = torch.clamp(torch.round(tensor_input + zero), min=quan_min, max=quan_max)


    if INPUT_W_FLG:
        torch.save(quantized_tensor,"output_pt/input/input.{}.pt".format(func_id))

    if exe_mode == 0:
        return quantized_tensor
    elif exe_mode == 1:
        if zero < -128:
            return quantized_tensor - (-128)
        else:
            return quantized_tensor - zero
        
    


def reshape_input_for_hardware_pe(input_tensor, pe_num: int = 4):
    """divide input into 4 batches"""
    input_dimension = len(input_tensor.shape)
    assert input_dimension == 4, 'Expect input tensor dimension: 4, but get %d' % input_dimension

    batch, channel, height, width = input_tensor.shape
    expanded_channel = channel
    if channel % pe_num != 0:
        expanded_channel = ((channel // pe_num) + 1) * pe_num

    tensor_buffer = torch.zeros([batch, expanded_channel, height, width]).to(input_tensor.device)
    tensor_buffer[:, 0: channel, :, :] = input_tensor
    output_tensor = torch.zeros(batch * pe_num, channel, height, width).to(input_tensor.device)

    for current_batch in range(batch):
        for current_pe in range(pe_num):
            batch_idx = current_batch * pe_num + current_pe
            for current_ch in range(current_pe , channel , 4):
                output_tensor[batch_idx, current_ch , :, :] = tensor_buffer[current_batch, current_ch, :, :]
    
    return output_tensor

def reshape_ouput_for_hardware_pe(input_tensor: Tensor, conv_weight, width_accum, input_scale, input_zero, weight_scale, exe_mode, func_id, pe_num: int = 4):
    input_dimension = len(input_tensor.shape)
    assert input_dimension == 4, 'Expect input tensor dimension: 4, but get %d' % input_dimension

    batch, channel, height, width = input_tensor.shape
    assert batch % pe_num == 0, 'Input batch size %d is not multiples of PE number %d' % (batch, pe_num)

    overflow_max = 2**(width_accum-1) - 1
    overflow_min = 0 - 2**(width_accum - 1)

    if exe_mode == 0:
        float_max = (overflow_max - input_zero) * input_scale * weight_scale
        float_min = (overflow_min - input_zero) * input_scale * weight_scale
        input_tensor_overflowed = torch.clamp(input_tensor, min=float_min, max=float_max)

    if exe_mode == 1:
        # 因为送入卷积层的输入是input - zero，在这里进行还原，计算输入为input的结果
        # 按输入的相同形式分batch
        conv_weight = reshape_input_for_hardware_pe(conv_weight, pe_num = 4)
        conv_weight = torch.sum(conv_weight, dim=(1,2,3))
        length = conv_weight.shape[0]
        conv_weight_pe0 = conv_weight[0:length:4]
        conv_weight_pe1 = conv_weight[1:length:4]
        conv_weight_pe2 = conv_weight[2:length:4]
        conv_weight_pe3 = conv_weight[3:length:4]
        conv_weight_pe0 = conv_weight_pe0[: , None, None]
        conv_weight_pe1 = conv_weight_pe1[: , None, None]
        conv_weight_pe2 = conv_weight_pe2[: , None, None]
        conv_weight_pe3 = conv_weight_pe3[: , None, None]
        # 目前仅支持输入batch = 1 
        input_add_zero = torch.zeros_like(input_tensor)
        if input_zero < -128:
            input_zero = -128
        input_add_zero[0,:,:,:] = input_tensor[0,:,:,:] + conv_weight_pe0 * input_zero
        input_add_zero[1,:,:,:] = input_tensor[1,:,:,:] + conv_weight_pe1 * input_zero
        input_add_zero[2,:,:,:] = input_tensor[2,:,:,:] + conv_weight_pe2 * input_zero
        input_add_zero[3,:,:,:] = input_tensor[3,:,:,:] + conv_weight_pe3 * input_zero
        # 测试硬件的阶段溢出
        # if input_add_zero.max()>overflow_max:
        #     print('max_overflow')
        # if input_add_zero.min()<overflow_min:
        #     print('min_overflow')
        # input_pos = input_add_zero % (2**(width_accum-1))
        # input_neg = (input_add_zero*(-1) % (2**(width_accum-1))) * (-1)
        # input_tensor_overflowed = torch.where(input_add_zero >= 0 ,input_pos, input_neg)
        input_tensor_overflowed = torch.clamp(input_add_zero, min=overflow_min, max=overflow_max)

    if(OUTPUT_PE_W_FLG):
        assert batch == 4 , "only support input batch = 1"
        store_path = "output_pt/pe_out/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        for pe_idx in range(pe_num):
            torch.save(input_tensor_overflowed[pe_idx,:,:,:],"output_pt/pe_out/pe_output{}_{}.pt".format(func_id, pe_idx))

    output_batch = int(batch / pe_num)
    output_tensor = torch.zeros(output_batch, channel, height, width).to(input_tensor.device)
    
    for i in range(output_batch):
        batch_start = i * pe_num
        batch_end = (i + 1) * pe_num
        output_tensor[i, :, :, :] = torch.sum(input_tensor_overflowed[batch_start: batch_end, :, :, :], dim=0)

    
    return output_tensor

def quantize_bias_with_scale(tensor_input: torch.Tensor, bias_scale, bias_width: int, exe_mode:int , func_id) -> torch.Tensor:
    assert isinstance(tensor_input, torch.Tensor)

    max_val = torch.max(tensor_input).item()
    min_val = torch.min(tensor_input).item()

    assert max_val != min_val , "bias elements are all equal"

    quan_max = 2 ** (bias_width - 1) - 1
    quan_min = 0 - 2 ** (bias_width - 1)

    bias_quan = torch.clamp(torch.round(tensor_input / bias_scale), min=quan_min, max=quan_max)

    if BIAS_W_FLG :
        store_path = "output_pt/bias/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        torch.save(bias_quan,"output_pt/bias/conv.bias.{}.pt".format(func_id))
        torch.save(bias_scale,"output_pt/bias/conv.bias.{}.scale.pt".format(func_id))

    if exe_mode == 0:
        quantized_tensor = bias_quan * bias_scale
    if exe_mode == 1:
        quantized_tensor = bias_quan
        
    return quantized_tensor

def PEs_and_bias_adder(input_tensor, bias, pe_add_width, pe_acc_width, bias_width, func_id, pe_num, exe_mode):
    input_scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id))
    input_zero = torch.load("output_pt/input/input.{}.zero.pt".format(func_id))
    weight_scale = torch.load("output_pt/weight/conv.weight.{}.scale.pt".format(func_id))
    conv_weight = torch.load("output_pt/weight/conv.weight.{}.pt".format(func_id))

    # Reshape the batched tensor to original shape
    tensor_buffer = reshape_ouput_for_hardware_pe(input_tensor, conv_weight, pe_acc_width, input_scale, input_zero, weight_scale, exe_mode, func_id, pe_num)
    #tensor_buffer = input_tensor

    overflow_max = 2**(pe_add_width-1) - 1
    overflow_min = 0 - 2**(pe_add_width - 1)

    if exe_mode == 0:
        float_max = (overflow_max - input_zero) * input_scale * weight_scale
        float_min = (overflow_min - input_zero) * input_scale * weight_scale
        input_tensor_overflowed = torch.clamp(tensor_buffer, min=float_min, max=float_max)

    if exe_mode == 1:
        input_tensor_overflowed = torch.clamp(tensor_buffer, min=overflow_min, max=overflow_max)

        if(OUTPUT_PE_ADD_W_FLG):
            store_path = "output_pt/pe_add/"
            if  not os.path.exists(store_path):#如果路径不存在
                os.makedirs(store_path)
            torch.save(input_tensor_overflowed,"output_pt/pe_add/pe_add_output{}.pt".format(func_id))

    # Convert bias list to tensor
    bias_tensor = Tensor(bias).to(input_tensor.device)
    # Broadcast 1D tensor to 4D
    bias_scale = input_scale * weight_scale
    quantized_bias = quantize_bias_with_scale(bias_tensor, bias_scale, bias_width, exe_mode, func_id)
    quantized_bias_broadcast = quantized_bias[None, :, None, None]
    store_path = "output_pt/bias/"
    if  not os.path.exists(store_path):#如果路径不存在
        os.makedirs(store_path)
    torch.save(bias_scale, "output_pt/bias/conv.bias.{}.scale.pt".format(func_id))
    torch.save(bias_tensor, "output_pt/bias/conv.bias.{}.pt".format(func_id))

    if exe_mode == 0:
        # Add the bias
        output_tensor = input_tensor_overflowed + quantized_bias_broadcast
    elif exe_mode == 1:
        conv_weight = torch.sum(conv_weight,dim=(1,2,3))
        conv_append = conv_weight * input_zero
        #conv_append = conv_weight * input_zero * input_scale
        conv_append_broadcast = conv_append[None, :, None, None]
        add_const = quantized_bias_broadcast - conv_append_broadcast
        quan_add_const = torch.clamp(add_const, min=-2**(bias_width-1),max=2**(bias_width-1)-1)

        if BIAS_QUAN_W_FLG :
            torch.save(quan_add_const,"output_pt/bias/conv.bias.quan{}.pt".format(func_id))
        #quan_add_const_hex_str = float_to_hex(quan_add_const, pe_acc_width)
        output_tensor = input_tensor_overflowed + quan_add_const

    return output_tensor

def quan_layer_between_const(input, data_bit = 16, shift_max = 32):
    assert data_bit < shift_max, "requan data bit must be less than shift_max"
    if int(input) != 0:
        #大于1
        before_point_bit = math.ceil(math.log2(int(input)+1))
        shift_n = data_bit - before_point_bit
        data_16bit = int(input*(2**shift_n))
        n = shift_n
    else:
        data = input * 2
        times = 0
        while int(data) == 0:
            times = times + 1
            data = data * 2
        shift_n = times + data_bit
        if shift_n > shift_max:
            shift_n = shift_max
        data_16bit = int(input*(2**shift_n))
        n = shift_n

    return data_16bit, n

def requan_conv2d_output(input_tensor, func_id, exe_mode):
    if exe_mode == 0: 
        return input_tensor
    elif exe_mode == 1:
        this_input_scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id))
        this_weight_scale = torch.load("output_pt/weight/conv.weight.{}.scale.pt".format(func_id))
        if func_id == 0:
            #第一层，残差单独处理
            next_input_scale = torch.load("output_pt/input/input.1.scale.pt")

            requan_const = this_input_scale / next_input_scale * this_weight_scale 
            requan_const = this_input_scale / next_input_scale * this_weight_scale 
            requan_const_16bit, requan_const_n = quan_layer_between_const(requan_const, REQUAN_BIT, REQUAN_N_MAX)
            output_tensor = input_tensor * requan_const_16bit * 2**(0-requan_const_n)
            output_shortcut = F.relu(output_tensor)

            if REQUAN_FACTOR_W_FLG :
                store_path = "output_pt/requan_factor/"
                if  not os.path.exists(store_path):#如果路径不存在
                    os.makedirs(store_path)
                torch.save(requan_const_16bit,"output_pt/requan_factor/requan_0_1.pt")
                torch.save(requan_const_n,"output_pt/requan_factor/n_0_1.pt")

            #残差输送给id为4的层，但是仍然用第二层scale的量化
            # next_input_scale_res = torch.load("output_pt/input/input.4.scale.pt")
            # requan_const_res = this_input_scale / next_input_scale_res * this_weight_scale 
            # requan_const_16bit_res, requan_const_n_res = quan_layer_between_const(requan_const_res, REQUAN_BIT, REQUAN_N_MAX)
            # shortcut_tensor = input_tensor * requan_const_16bit_res * 2**(0-requan_const_n_res)
            # # shortcut_tensor = torch.round(input_tensor / next_input_scale_res)
            # shortcut_tensor = F.relu(shortcut_tensor)
            store_path = "output_pt/residual/"
            if  not os.path.exists(store_path):#如果路径不存在
                os.makedirs(store_path)
            torch.save(output_shortcut,"output_pt/residual/shortcut_tensor.pt")

            # if REQUAN_FACTOR_W_FLG :
            #     torch.save(requan_const_16bit_res,"output_pt/requan_factor/requan_res_0_4.pt")
            #     torch.save(requan_const_n_res,"output_pt/requan_factor/n_res_0_4.pt")
        elif func_id == 3:
            #最后一层，直接反量化为浮点
            next_input_scale = torch.load("output_pt/input/input.{}.scale.pt".format(1))
            requan_const = this_input_scale / next_input_scale * this_weight_scale 
            requan_const_16bit, requan_const_n = quan_layer_between_const(requan_const, REQUAN_BIT, REQUAN_N_MAX)
            # output_tensor = input_tensor * requan_const
            output_tensor = input_tensor * requan_const_16bit * 2**(0-requan_const_n)
            # output_tensor = input_tensor

            if REQUAN_FACTOR_W_FLG :
                torch.save(requan_const_16bit,"output_pt/requan_factor/requan_3_4.pt")
                torch.save(requan_const_n,"output_pt/requan_factor/n_3_4.pt")

        elif func_id == 4:
            #最后一层，直接反量化为浮点
            requan_const = this_input_scale * this_weight_scale
            requan_const_16bit, requan_const_n = quan_layer_between_const(requan_const, REQUAN_BIT, REQUAN_N_MAX)
            # output_tensor = input_tensor * requan_const
            output_tensor = input_tensor * requan_const_16bit * 2**(0-requan_const_n)
            # output_tensor = input_tensor

            if REQUAN_FACTOR_W_FLG :
                torch.save(requan_const_16bit,"output_pt/requan_factor/requan_4_5.pt")
                torch.save(requan_const_n,"output_pt/requan_factor/n_4_5.pt")
        else:
            #其他层，将输出反量化
            next_input_scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id + 1))

            requan_const = this_input_scale / next_input_scale * this_weight_scale 
            requan_const_16bit, requan_const_n = quan_layer_between_const(requan_const, REQUAN_BIT, REQUAN_N_MAX)
            # output_tensor = input_tensor * requan_const
            output_tensor = input_tensor * requan_const_16bit * 2**(0-requan_const_n)
            # output_tensor = torch.round(input_tensor / next_input_scale)

            if REQUAN_FACTOR_W_FLG :
                torch.save(requan_const_16bit,"output_pt/requan_factor/requan_{}_{}.pt".format(func_id, func_id + 1))
                torch.save(requan_const_n,"output_pt/requan_factor/n_{}_{}.pt".format(func_id, func_id + 1))

        return output_tensor

