import torch
from models import dm
from models import sesr
from models import nr
from models import nrdm_3
from models import nrdm_6
from self_dataset import TestDataset
import os
from torch import nn
from models import quantize_utils_cuda as quantize
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models import quantize_utils_cuda as quantize

from define import QUAN_BIT, PE, BIAS_BIT, PE_ACC_BIT, PE_ADD_BIT, MFLAG, TGT_BINS_NUM, BINS_NUM

from myQL.quan_func import quantize_model_weight, quantize_asymmetrical_by_tensor, reshape_input_for_hardware_pe, PEs_and_bias_adder, requan_conv2d_output
from myQL.quan_classes import NodeInsertMapping, FunctionPackage, NodeInsertMappingElement
from myQL.graph_modify import insert_before, insert_bias_bypass, insert_after

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mflag = MFLAG
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

# print(model)
"""------------------------------------quantize start---------------------------------------"""
qmode = 0

# quantize weight
model = quantize_model_weight(model, QUAN_BIT, qmode)

#quantize input
# 1: quantize
mapping = NodeInsertMapping()
quan_FP = FunctionPackage(quantize_asymmetrical_by_tensor, {'width': QUAN_BIT, 'exe_mode':qmode})
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, quan_FP)
mapping.add_config(conv2d_config)
conv2d_config_pxlshfl = NodeInsertMappingElement(torch.nn.PixelShuffle, quan_FP)
mapping.add_config(conv2d_config_pxlshfl)
model = insert_before(model_input=model, insert_mapping=mapping, has_func_id=True)

# 2: divide input into 4 batches for 4 PEs
input_reshape_mapping = NodeInsertMapping()
reshape_function_package = FunctionPackage(reshape_input_for_hardware_pe, {'pe_num': PE})
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
input_reshape_mapping.add_config(conv2d_config)
model = insert_before(model_input=model, insert_mapping=input_reshape_mapping)

# 3: add batches of 4 PEs 
bypass_mapping = NodeInsertMapping()
reshape_function_package = FunctionPackage(PEs_and_bias_adder, {'pe_add_width':PE_ADD_BIT, 'pe_acc_width': PE_ACC_BIT, 'bias_width': BIAS_BIT, 'pe_num': PE, 'exe_mode':qmode})
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
bypass_mapping.add_config(conv2d_config)
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
		ispssim = compare_ssim(gts, gfake, data_range=1.0, channel_axis=2)
	print(isppsnr)
	totalpsnr+= isppsnr
	totalssim += ispssim
	totalnum += 1
	# if(i==0):
	# 	break
tasks = ['nr','dm','nrdm_small','nrdm_big','sr']
print(tasks[mflag-1] + ' mean psnr is: ' ,totalpsnr/totalnum,' ssim is: ',totalssim/totalnum)

for id in range(6):
	inp_hist = torch.load("output_pt/input/input.{}.hist.pt".format(id))
	print(inp_hist)
	for threshold in range(TGT_BINS_NUM,BINS_NUM):
		# get P
		reference_distribution_P = inp_hist[0:threshold]
		outliers_count = torch.sum(inp_hist[threshold:BINS_NUM])
		reference_distribution_P[threshold-1] += outliers_count
		P = reference_distribution_P / torch.sum(reference_distribution_P)

		# get Q
		target_bin = TGT_BINS_NUM
		num_per_bin = threshold / target_bin
		quantize_distribution = np.zeros((target_bin,))
		for i in range(target_bin):
			start = i * num_per_bin
			end = start + num_per_bin
			left_upper = int(np.ceil(start))
			if left_upper > start:
				left_scale = left_upper - start
				quantize_distribution[i] += left_scale * distribution[left_upper - 1]
			right_lower = int(np.floor(end))
			if right_lower < end:
				right_scale = end - right_lower
				quantize_distribution[i] += right_scale * distribution[right_lower]
			for j in range(left_upper,right_lower):
				quantize_distribution[i] += distribution[j]
		# get Q
		expand_distribution=np.zeros_like(t_distribution)
		for i in range(target_bin):
		    start = i * num_per_bin
		    end = start + num_per_bin
		    count = 0
		    left_upper = int(np.ceil(start))
		    left_scale = 0
		    if left_upper > start:
		        left_scale = left_upper - start
		        if t_distribution[left_upper - 1] != 0:
		            count += left_scale
		    right_lower = int(np.floor(end))
		    right_scale = 0
		    if right_lower < end:
		        right_scale = end - right_lower
		        if t_distribution[right_lower] != 0:
		            count += right_scale
		    for j in range(left_upper,right_lower):
		        if t_distribution[j] != 0:
		            count+=1
		    expand_value = quantize_distribution[i] / count
		    if left_upper > start:
		        if t_distribution[left_upper - 1] != 0:
		            expand_distribution[left_upper - 1] += expand_value * left_scale
		    if right_lower < end:
		        if t_distribution[right_lower] != 0:
		            expand_distribution[right_lower] += expand_value * right_scale
		    for j in range(left_upper,right_lower):
		        if t_distribution[j] != 0:
		            expand_distribution[j] += expand_value






