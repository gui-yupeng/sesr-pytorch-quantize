import torch
import math
import os

target = "input"
func_id = 1
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
    else:
        str_hex = '{:02x}'.format(int_res)
    
    return str_hex

if target == "input":
    tensor_data = torch.load("output_pt/input/input.{}.pt".format(func_id))
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

    store_path = "output_txt/input/"
    if  not os.path.exists(store_path):#如果路径不存在
        os.makedirs(store_path)
    with open("output_txt/input/input.{}.txt".format(func_id),"w") as f:
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
                            f.write(float_to_hex(data_item, 8))
                        f.write('\n')
                        if block_h_start + h_idx == height - 1:
                            break
