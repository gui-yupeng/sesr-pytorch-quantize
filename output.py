import torch
import math
import os
import math
from define import PE_ACC_BIT, PE_ADD_BIT, QUAN_BIT, REQUAN_BIT, REQUAN_N_MAX, BIAS_BIT

target = "input"




tile_width = 32

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
    elif bit_hex == 6 :
        str_hex = '{:06x}'.format(int_res)
    elif bit_hex == 7 :
        str_hex = '{:07x}'.format(int_res)
    elif bit_hex == 8 :
        str_hex = '{:08x}'.format(int_res)
    else:
        str_hex = '{:02x}'.format(int_res)
    
    return str_hex

if target == "input":
    conv_kernel_size = [0,5,3,3,3,5]
    current_height_overlap = tile_width
    current_width_overlap = tile_width
    for layer_id in range(6):
        tensor_data = torch.load("output_pt/input/input.{}.pt".format(layer_id))
        batch, channel, height, width = tensor_data.shape
        expanded_height = height
        expanded_width = width
        # if height % tile_width != 0:
        #     expanded_height = ((height // tile_width) + 1) * tile_width
        # if width % tile_width != 0:
        #     expanded_width = ((width // tile_width) + 1) * tile_width
        # pad2
        expanded_height = ((height // tile_width) + 1) * tile_width
        expanded_width = ((width // tile_width) + 1) * tile_width

        tensor_expand = torch.zeros(batch, channel, height, expanded_width)
        tensor_expand[:, :, 0:height, 0:width] = tensor_data[:, :, :, :]
        # height = height + 1
        width_block_num = int(expanded_width / tile_width)
        height_block_num = int(expanded_height / tile_width)

        current_height_overlap = current_height_overlap - conv_kernel_size[layer_id]//2
        current_width_overlap = current_width_overlap - conv_kernel_size[layer_id]//2

        store_path = "output_txt/input/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        with open("output_txt/input/input.{}.txt".format(layer_id),"w") as f:
            block_h_start = 0
            for height_block_idx in range(height_block_num):
                block_w_start = 0
                is_first_height_block = (height_block_idx == 0)
                if is_first_height_block:
                    current_height = current_height_overlap
                else:
                    current_height = tile_width
                for width_block_idx in range(width_block_num):
                    is_first_width_block = (width_block_idx == 0)
                    if is_first_width_block:
                        current_width = current_width_overlap
                    else:
                        current_width = tile_width
                    
                    is_last_height_block = (height_block_idx == (height_block_num-1))
                    if is_last_height_block:
                        current_height = height - block_h_start
                    else:
                        current_height = current_height
                    
                    f.write('{}\n'.format("{:02x}".format(int(current_height))))
                    f.write('{}\n'.format("{:02x}".format(int(channel))))

                    for c_idx in range(channel):
                        f.write('{}\n'.format("{:02x}".format(int(c_idx))))
                        for h_idx in range(current_height):
                            # if is_first_width_block:
                            #     for w_idx in range(32-current_width):
                            #         f.write(float_to_hex(0, QUAN_BIT))
                            #     for w_idx in range(current_width):
                            #         real_hi = block_h_start + h_idx
                            #         real_wi = block_w_start + w_idx
                            #         data_item = tensor_expand[0, c_idx, real_hi, real_wi].item()
                            #         f.write(float_to_hex(data_item, QUAN_BIT))
                            #     f.write('\n')
                            # else:
                                for w_idx in range(current_width):
                                    real_hi = block_h_start + h_idx
                                    real_wi = block_w_start + w_idx
                                    data_item = tensor_expand[0, c_idx, real_hi, real_wi].item()
                                    f.write(float_to_hex(data_item, QUAN_BIT))
                                for w_idx in range(32-current_width):
                                    f.write(float_to_hex(0, QUAN_BIT))
                                f.write('\n')
                    
                    # 更新每一次分块的起始地址
                    block_w_start = block_w_start + current_width
                block_h_start = block_h_start + current_height

target = "bias"
if target == "bias":
    store_path = "output_txt/bias/"
    if  not os.path.exists(store_path):#如果路径不存在
        os.makedirs(store_path)
    with open("output_txt/bias/param_buf.txt", "w") as f:
        f.write(float_to_hex(5,8))
        f.write('\n')
        requan_res = torch.load("output_pt/requan_factor/requan_res.pt")
        requan_res = float_to_hex(requan_res, REQUAN_BIT)
        for layer_id in range(5):
            bias_quan = torch.load("output_pt/bias/conv.bias.quan{}.pt".format(layer_id))
            requan_factor = torch.load("output_pt/requan_factor/requan_{}_{}.pt".format(layer_id, layer_id + 1))
            _, chnl, _, _ = bias_quan.shape
            f.write(float_to_hex(chnl,8))
            f.write('\n')
            for ci in range(chnl):
                f.write(float_to_hex(bias_quan[0,ci,0,0],BIAS_BIT))
                f.write(float_to_hex(requan_factor,REQUAN_BIT))
                f.write(requan_res)
                f.write('\n')

target = "pe_out"
if target == "pe_out":
    for layer_id in range(5):
        for pe_id in range(4):
            tensor_data = torch.load("output_pt/pe_out/pe_output{}_{}.pt".format(layer_id, pe_id))
            #保存的时候是三维
            tensor_data = tensor_data[None,:,:,:]
            batch, channel, height, width = tensor_data.shape
            expanded_height = height
            expanded_width = width
            if height % tile_width != 0:
                expanded_height = ((height // tile_width) + 1) * tile_width
            if width % tile_width != 0:
                expanded_width = ((width // tile_width) + 1) * tile_width

            tensor_expand = torch.zeros(batch, channel, height, expanded_width)
            tensor_expand[:, :, :, 0:width] = tensor_data[:, :, :, :]

            width_block_num = int(expanded_width / tile_width)
            height_block_num = int(expanded_height / tile_width)

            store_path = "output_txt/pe_out/"
            if  not os.path.exists(store_path):#如果路径不存在
                os.makedirs(store_path)
            with open("output_txt/pe_out/pe_output{}_{}.txt".format(layer_id, pe_id),"w") as f:
                for height_block_idx in range(height_block_num):
                    for width_block_idx in range(width_block_num):
                        block_h_start = height_block_idx * tile_width
                        block_w_start = width_block_idx * tile_width
                        is_last_height_block = (height_block_idx == (height_block_num-1))
                        if is_last_height_block:
                             wr_line_num = height - block_h_start
                        else:
                            wr_line_num = 32
                        f.write('{}\n'.format("{:02x}".format(int(wr_line_num))))
                        f.write('{}\n'.format("{:02x}".format(int(channel))))

                        for c_idx in range(channel):
                            f.write('{}\n'.format("{:02x}".format(int(c_idx))))
                            for h_idx in range(32):
                                for w_idx in range(32):
                                    real_hi = block_h_start + h_idx
                                    real_wi = block_w_start + w_idx
                                    data_item = tensor_expand[0, c_idx, real_hi, real_wi].item()
                                    f.write(float_to_hex(data_item, PE_ACC_BIT))
                                f.write('\n')
                                if block_h_start + h_idx == height - 1:
                                    break

target = "pe_add"
if target == "pe_add":
    for layer_id in range(5):
        tensor_data = torch.load("output_pt/pe_add/pe_add_output{}.pt".format(layer_id))
        batch, channel, height, width = tensor_data.shape
        expanded_height = height
        expanded_width = width
        if height % tile_width != 0:
            expanded_height = ((height // tile_width) + 1) * tile_width
        if width % tile_width != 0:
            expanded_width = ((width // tile_width) + 1) * tile_width
        tensor_expand = torch.zeros(batch, channel, height, expanded_width)
        tensor_expand[:, :, :, 0:width] = tensor_data[:, :, :, :]
        width_block_num = int(expanded_width / tile_width)
        height_block_num = int(expanded_height / tile_width)
        store_path = "output_txt/pe_add/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        with open("output_txt/pe_add/pe_add_output{}.txt".format(layer_id),"w") as f:
            for height_block_idx in range(height_block_num):
                for width_block_idx in range(width_block_num):
                    block_h_start = height_block_idx * tile_width
                    block_w_start = width_block_idx * tile_width
                    is_last_height_block = (height_block_idx == (height_block_num-1))
                    if is_last_height_block:
                         wr_line_num = height - block_h_start
                    else:
                        wr_line_num = 32
                    f.write('{}\n'.format("{:02x}".format(int(wr_line_num))))
                    f.write('{}\n'.format("{:02x}".format(int(channel))))
                    for c_idx in range(channel):
                        f.write('{}\n'.format("{:02x}".format(int(c_idx))))
                        for h_idx in range(32):
                            for w_idx in range(32):
                                real_hi = block_h_start + h_idx
                                real_wi = block_w_start + w_idx
                                data_item = tensor_expand[0, c_idx, real_hi, real_wi].item()
                                f.write(float_to_hex(data_item, PE_ADD_BIT))           
                            f.write('\n')
                            if block_h_start + h_idx == height - 1:
                                break

target = "requan_shift_n"
if target == "requan_shift_n":
    store_path = "output_txt/requan_shift_n/"
    if  not os.path.exists(store_path):#如果路径不存在
        os.makedirs(store_path)
    with open("output_txt/requan_shift_n/requan_shift_n.txt","w") as f:
        for layer_id in range(5):
            tensor_data = torch.load("output_pt/requan_factor/n_{}_{}.pt".format(layer_id, layer_id + 1))
            f.write(float_to_hex(tensor_data, math.log2(REQUAN_N_MAX)))
            f.write('\n')
        tensor_data = torch.load("output_pt/requan_factor/n_res.pt")
        f.write(float_to_hex(tensor_data, math.log2(REQUAN_N_MAX)))



