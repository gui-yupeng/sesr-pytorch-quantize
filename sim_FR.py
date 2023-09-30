#寻找真相
import torch
from models import dm
from models import sesr_sim,sesr
from models import nr
from models import nrdm_3
from models import nrdm_6
from self_dataset import TestDataset
from torch import Tensor
import os
from torch import nn
from models import quantize_utils_cuda as quantize
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models import quantize_utils_cuda as quantize

from define import QUAN_BIT, PE, BIAS_BIT, PE_ACC_BIT, PE_ADD_BIT

from myQL.quan_func import 	quantize_model_weight, \
							quantize_asymmetrical_by_tensor, \
							reshape_input_for_hardware_pe, \
							PEs_and_bias_adder,\
							requan_conv2d_output
from myQL.quan_classes import NodeInsertMapping, FunctionPackage, NodeInsertMappingElement
from myQL.graph_modify import insert_before, insert_bias_bypass, insert_after
from torch.nn import functional as F
import matplotlib.pyplot as plt

mflag = 5
#qatf = "qat_"
qatf = ""
if mflag == 1:
    model = nr.nr()
    traindata = TestDataset(1)
    checkpointp = './model_params/nr_' + qatf
elif mflag == 2:
    model = dm.dm()
    traindata = TestDataset(2)
    checkpointp = './model_params/dm_' + qatf
elif mflag == 3:
    model = nrdm_3.nr()
    traindata = TestDataset(3)
    checkpointp = './model_params/nrdm_3_' + qatf
elif mflag == 4:
    model = nrdm_6.nr()
    traindata = TestDataset(4)
    checkpointp = './model_params/nrdm_6_' + qatf
elif mflag == 5:
    model = sesr.sesr()
    traindata = TestDataset(5)
    checkpointp = './model_params/sr_' + qatf

model = model.cuda()
loader_train = torch.utils.data.DataLoader(traindata, batch_size=1, num_workers=4,
										   shuffle=False, pin_memory=False)


# Training
model.train()
if qatf == "qat_":
	quantize.prepare(model, inplace=True, a_bits=8, w_bits=8, q_type=0, q_level="C")
state_temp_dict = torch.load(checkpointp +'G.pth')
model.load_state_dict(state_temp_dict)

# infer
model.collapse()
print(model)
"""------------------------------------quantize start---------------------------------------"""

def PEs_and_bias_adder_fr(input_tensor, bias, pe_add_width, pe_acc_width, bias_width, func_id, pe_num, exe_mode):
    input_scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id))
    input_zero = torch.load("output_pt/input/input.{}.zero.pt".format(func_id))
    weight_scale = torch.load("output_pt/weight/conv.weight.{}.scale.pt".format(func_id))

    overflow_max = 2**(pe_add_width-1) - 1
    overflow_min = 0 - 2**(pe_add_width - 1)
    

    input_tensor_overflowed = torch.clamp(input_tensor, min=overflow_min, max=overflow_max) 

    # Convert bias list to tensor
    bias_tensor = Tensor(bias).to(input_tensor.device)
    # Broadcast 1D tensor to 4D
    bias_tensor_broadcast = bias_tensor[None, :, None, None]

    # store_path = "output_png/bias/"
    # if  not os.path.exists(store_path):#如果路径不存在
    #     os.makedirs(store_path)
    # plt.cla()
    # plt.hist(bias_tensor_broadcast.reshape(-1).cpu().numpy(),bins=300)
    # plt.savefig("output_png/bias/conv.bias.{}.png".format(func_id))

    bias_scale = input_scale * weight_scale
    quantized_bias_broadcast = torch.clamp(torch.round(bias_tensor_broadcast / bias_scale), min=0 - 2 ** (bias_width - 1), max=2 ** (bias_width - 1) - 1)

    if exe_mode == 0:
        output_tensor = input_tensor
        if func_id == 0:
            np_tensor = np.array(output_tensor.reshape(-1).cpu())
            np.savetxt('npresult0.txt',np_tensor)
    elif exe_mode == 1:
        #问题所在
        conv_weight = torch.load("output_pt/weight/conv.weight.{}.pt".format(func_id))
        conv_weight = torch.sum(conv_weight,dim=(1,2,3))
        # conv_append = conv_weight * input_zero
        print(conv_weight)
        print(input_zero)
        conv_append = conv_weight * input_zero * input_scale * weight_scale * 0
        conv_append_broadcast = conv_append[None, :, None, None]
        # add_const = quantized_bias_broadcast - conv_append_broadcast
        add_const = quantized_bias_broadcast * bias_scale - conv_append_broadcast
        quan_add_const = torch.clamp(add_const, min=-2**(bias_width-1),max=2**(bias_width-1)-1)

        store_path = "output_png/bias/"
        if  not os.path.exists(store_path):#如果路径不存在
            os.makedirs(store_path)
        plt.cla()
        plt.hist(quan_add_const.reshape(-1).cpu().numpy(),bins=10)
        plt.savefig("output_png/bias/conv.bias.{}.png".format(func_id))

        output_tensor = input_tensor_overflowed + quan_add_const
        # output_tensor = output_tensor * input_scale * weight_scale
        
        if func_id == 0:
            np_tensor = np.array(output_tensor.reshape(-1).cpu())
            np.savetxt('npresult1.txt',np_tensor)

    return output_tensor

def quantize_asymmetrical_by_tensor_fr(tensor_input: torch.Tensor, width: int, exe_mode :int, func_id: int = None, filename : str = None) -> torch.Tensor:
        
    scale = torch.load("output_pt/input/input.{}.scale.pt".format(func_id))
    zero = torch.load("output_pt/input/input.{}.zero.pt".format(func_id))
    quan_max = 2 ** (width - 1) - 1
    quan_min = 0 - 2 ** (width - 1)
    quantized_tensor = torch.clamp(torch.round(tensor_input / scale + zero), min=quan_min, max=quan_max)
    if exe_mode == 1:
        # return torch.tensor(quantized_tensor, dtype=torch.int32)
        # return quantized_tensor
        return (quantized_tensor - zero ) * scale
    elif exe_mode == 0:
        return (quantized_tensor - zero) * scale


qmode = 1

# quantize weight
model = quantize_model_weight(model, QUAN_BIT, qmode)

mapping = NodeInsertMapping()
quan_FP = FunctionPackage(quantize_asymmetrical_by_tensor_fr, {'width': QUAN_BIT, 'exe_mode':qmode})
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, quan_FP)
mapping.add_config(conv2d_config)
model = insert_before(model_input=model, insert_mapping=mapping, has_func_id=True)

bypass_mapping = NodeInsertMapping()
bypass_FP = FunctionPackage(PEs_and_bias_adder_fr, {'pe_add_width':PE_ADD_BIT, 'pe_acc_width': PE_ACC_BIT, 'bias_width': BIAS_BIT, 'pe_num': PE, 'exe_mode':qmode})
bypass_config = NodeInsertMappingElement(torch.nn.Conv2d, bypass_FP)
bypass_mapping.add_config(bypass_config)
model = insert_bias_bypass(model_input=model, insert_mapping=bypass_mapping)
"""------------------------------------quantize end-----------------------------------------"""

def three2one(in_np):
	outs = np.zeros(( in_np.shape[0], in_np.shape[1]))
	outs[0::2, 0::2] = in_np[0::2, 0::2, 0]
	outs[1::2, 0::2] = in_np[1::2, 0::2, 1]
	outs[0::2, 1::2] = in_np[0::2, 1::2, 1]
	outs[1::2, 1::2] = in_np[1::2, 1::2, 2]
	return outs

totalpsnr = 0
totalssim = 0
totalnum = 0
for i, data in enumerate(loader_train):
	if i == 1:
		inps,gts,_ = data[:]
		inps = inps.cuda()
		gts = gts.detach().numpy()[0, :, :, :].transpose(1, 2, 0)
		with torch.no_grad():
			gfake = model(inps)

		# compute psnr and ssim
		gfake = gfake.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
		gfake = np.clip(gfake, 0, 1)
		if mflag == 1: #nr
			gfake = three2one(gfake)
			gts = three2one(gts)
		if mflag == 5:
			gfake = gfake[:,:,0]
			gts = gts[:,:,0]

		isppsnr = compare_psnr(gts, gfake, data_range=1.0)
		if  mflag == 1 or  mflag == 5:
			ispssim = compare_ssim(gts, gfake, data_range=1.0, multichannel=False)
		else:
			ispssim = compare_ssim(gts, gfake, data_range=1.0, multichannel=True)
		print(isppsnr)
		totalpsnr+= isppsnr
		totalssim += ispssim
		totalnum += 1
tasks = ['nr','dm','nrdm_small','nrdm_big','sr']
print(tasks[mflag-1] + ' mean psnr is: ' ,totalpsnr/totalnum,' ssim is: ',totalssim/totalnum)

# inps = torch.rand(1,1,40,40).cuda()
# gfake = model(inps)

print("bit:",QUAN_BIT)