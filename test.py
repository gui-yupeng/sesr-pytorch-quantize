import torch
from models import dm
from models import sesr
from models import nr
from models import nrdm_3
from models import nrdm_6
from models import sesr_arch
from self_dataset import TestDataset
import os
from torch import nn
from models import quantize_utils_cuda as quantize
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models import quantize_utils_cuda as quantize

from define import QUAN_BIT, PE, BIAS_BIT, PE_ACC_BIT, PE_ADD_BIT, MFLAG

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
    checkpointp = './model_params/nrdm_3_raw_' + qatf
elif mflag == 4:
    model = nrdm_6.nr()
    traindata = TestDataset(4)
    checkpointp = './model_params/nrdm_6_' + qatf
elif mflag == 5:
    model = sesr.sesr()
    traindata = TestDataset(5)
    checkpointp = './model_params/sr_' + qatf
elif mflag == 6:
    model = sesr_arch.sesr()
    traindata = TestDataset(6)
    checkpointp = './model_params/sr_x2_' + qatf

model = model.cuda()
loader_train = torch.utils.data.DataLoader(traindata, batch_size=1, num_workers=4,
										   shuffle=False, pin_memory=False)


# Training
model.train()
if qatf == "qat_":
	quantize.prepare(model, inplace=True, a_bits=8, w_bits=8, q_type=0, q_level="C")

if mflag == 6:
	state_temp_dict = torch.load('./model_params/x2sesr.pth.tar')['state_dict']
elif mflag == 5:
	state_temp_dict = torch.load('./model_params/x4sesr.pth')
else:
	state_temp_dict = torch.load(checkpointp +'G.pth')

model = model.float()
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

for id in range(6):
	if os.path.isfile("output_pt/input/input.{}.max_val.pt".format(id)):
		max_val = -1000
		min_val = 1000
		torch.save(max_val,"output_pt/input/input.{}.max_val.pt".format(id))
		torch.save(min_val,"output_pt/input/input.{}.min_val.pt".format(id))



"""------------------------------------quantize end-----------------------------------------"""



def three2one(in_np):
	outs = np.zeros(( in_np.shape[0], in_np.shape[1]))
	outs[0::2, 0::2] = in_np[0::2, 0::2, 0]
	outs[1::2, 0::2] = in_np[1::2, 0::2, 1]
	outs[0::2, 1::2] = in_np[0::2, 1::2, 1]
	outs[1::2, 1::2] = in_np[1::2, 1::2, 2]
	return outs

def compute_psnr(img_pred, img_true, data_range=255., eps=1e-8):
    err = (img_pred - img_true) ** 2
    err = np.mean(err)
    return 10. * np.log10((data_range ** 2) / (err + eps))
def rgb_to_yuv(img):# range in 0-1，out0-255
    rgb_weights = np.array([65.481, 128.553, 24.966])
    img = np.matmul(img, rgb_weights) + 16.
    return np.clip(img,0,255.)

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
	inp_size= inps.size()
	inps_x2 = torch.zeros(inp_size[0],inp_size[1],inp_size[2]*2,inp_size[3]*2).cuda()
	inps_x2[:,:,0::2,0::2] = inps[:,:,:,:]
	inps_x2[:,:,0::2,1::2] = inps[:,:,:,:]
	inps_x2[:,:,1::2,0::2] = inps[:,:,:,:]
	inps_x2[:,:,1::2,1::2] = inps[:,:,:,:]
	if mflag == 6:
		gfake = gfake + inps_x2
	gfake = gfake.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
	gfake = np.clip(gfake, 0, 1)
	if mflag == 1: #nr
		gfake = three2one(gfake)
		gts = three2one(gts)
	if mflag == 5:
		gfake = gfake[:,:,0]
		gts = gts[:,:,0]
	
	if mflag == 5:
		isppsnr = compute_psnr(gts*255., gfake*255.)
	elif mflag == 6:
		isppsnr = compute_psnr(rgb_to_yuv(gts), rgb_to_yuv(gfake))
	else:
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
tasks = ['nr','dm','nrdm_small','nrdm_big','srx4','srx2']
print(tasks[mflag-1] + ' mean psnr is: ' ,totalpsnr/totalnum,' ssim is: ',totalssim/totalnum)

#用于计算scale和zero，对于最后一个输入(id=5)，防止zero小于-128，硬件无法实现，所以扩大scale，强行令zero=-128 
print("calibrate start")
for id in range(5):
	inp_max = torch.load("output_pt/input/input.{}.max_val.pt".format(id))
	inp_min = torch.load("output_pt/input/input.{}.min_val.pt".format(id))
	quan_max = 2 ** (QUAN_BIT - 1) - 1
	quan_min = 0 - 2 ** (QUAN_BIT - 1)
	quan_scale = (inp_max - inp_min) / (quan_max - quan_min)
	quan_zero = quan_min - round(inp_min/quan_scale)
	print('scale:',quan_scale)
	print('zero:',quan_zero)
	# quan_scale = 1
	# quan_zero = 0
	torch.save(quan_scale,"output_pt/input/input.{}.scale.pt".format(id))
	torch.save(quan_zero,"output_pt/input/input.{}.zero.pt".format(id))


# pixelshuffle输入的量化单独做
inp_max = torch.load("output_pt/input/input.5.max_val.pt")
inp_min = torch.load("output_pt/input/input.5.min_val.pt")
# 最后一层的输出，我们更希望所有输出都是正值，所以将小于零的最小输出强行定为0
# 由于硬件8bit不方便存储小于-128的数据，所以将大于零的最小输入数据强行定为0
inp_min = 0
quan_max = 2 ** (QUAN_BIT - 1) - 1
quan_min = 0 - 2 ** (QUAN_BIT - 1)
quan_scale = (inp_max - inp_min) / (quan_max - quan_min)
quan_zero = quan_min - round(inp_min/quan_scale)
print('scale:',quan_scale)
print('zero:',quan_zero)
# quan_scale = 1
# quan_zero = 0
torch.save(quan_scale,"output_pt/input/input.{}.scale.pt".format(id+1))
torch.save(quan_zero,"output_pt/input/input.{}.zero.pt".format(id+1))


# inps = torch.rand(1,1,40,40).cuda()
# gfake = model(inps)
print("calibrate end")
print("bit:",QUAN_BIT)




