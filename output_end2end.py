import torch
import math
import os

from define import PE_ACC_BIT, PE_ADD_BIT, QUAN_BIT, REQUAN_BIT, REQUAN_N_MAX, BIAS_BIT

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

target = "start"
if target == "start":
    conv_kernel_size = [0,5,3,3,3,5]
    current_height_overlap = tile_width

    layer_id=0
    tensor_data = torch.load("output_pt/input/input.{}.pt".format(layer_id))
    batch, channel, height, width = tensor_data.shape
    expanded_height = height
    if height % tile_width != 0:
        expanded_height = ((height // tile_width) + 1) * tile_width
    tensor_expand = torch.zeros(batch, channel, expanded_height, width)
    tensor_expand[:, :, 0:height, 0:width] = tensor_data[:, :, :, :]

    height_block_num = int(expanded_height / tile_width)
    store_path = "output_txt/input/"
    if  not os.path.exists(store_path):#如果路径不存在
        os.makedirs(store_path)
    with open("output_txt/input/input.{}.txt".format(layer_id),"w") as f:
        for height_block_idx in range(height_block_num):
            f.write('{}\n'.format("{:02x}".format(int(height_block_idx))))
            for channel_idx in range(channel):
                f.write('{}\n'.format("{:02x}".format(int(channel_idx))))
                line_num = 0
                for block_height_idx in range(tile_width):
                    for width_idx in range(width):
                        data_item = tensor_expand[0, channel_idx, height_block_idx+block_height_idx, width_idx].item()
                        f.write(float_to_hex(data_item, QUAN_BIT))
                        line_num = line_num + 1
                        if line_num == 4:
                            f.write('\n')
                            line_num = 0
                    if line_num != 0:
                        f.write('\n')


    layer_id=5
    tensor_data = torch.load("output_pt/input/input.{}.pt".format(layer_id))
    batch, channel, height, width = tensor_data.shape
    expanded_height = height
    if height % tile_width != 0:
        expanded_height = ((height // tile_width) + 1) * tile_width
    tensor_expand = torch.zeros(batch, channel, expanded_height, width)
    tensor_expand[:, :, 0:height, 0:width] = tensor_data[:, :, :, :]

    height_block_num = int(expanded_height / tile_width)
    store_path = "output_txt/input/"
    if  not os.path.exists(store_path):#如果路径不存在
        os.makedirs(store_path)
    with open("output_txt/input/input.{}.txt".format(layer_id),"w") as f:
        for height_block_idx in range(height_block_num):
            f.write('{}\n'.format("{:02x}".format(int(height_block_idx))))
            for channel_idx in range(channel):
                f.write('{}\n'.format("{:02x}".format(int(channel_idx))))
                line_num = 0
                for block_height_idx in range(tile_width):
                    for width_idx in range(width):
                        data_item = tensor_expand[0, channel_idx, height_block_idx+block_height_idx, width_idx].item()
                        f.write(float_to_hex(data_item, QUAN_BIT))
                        line_num = line_num + 1
                        if line_num == 4:
                            f.write('\n')
                            line_num = 0
                    if line_num != 0:
                        f.write('\n')