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
#from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

mflag = 3
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

model = model.cpu()
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
	# print(state_temp_dict)
model = model.float()
model.load_state_dict(state_temp_dict,strict=False)
# print(model.state_dict())
# print(model)
# infer
model.collapse()
# model.eval()
# print(model)
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
	inps,gts = data[:]
	inps = inps.cpu()
	gts = gts.detach().numpy()[0, :, :, :].transpose(1, 2, 0)
	with torch.no_grad():
		gfake = model(inps)
	# compute psnr and ssim
	inp_size= inps.size()
	inps_x2 = torch.zeros(inp_size[0],inp_size[1],inp_size[2]*2,inp_size[3]*2)
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

	# isppsnr = compare_psnr(gts, gfake, data_range=1.0)
	if mflag == 5:
		isppsnr = compute_psnr(gts*255., gfake*255.)
	elif mflag == 6:
		isppsnr = compute_psnr(rgb_to_yuv(gts), rgb_to_yuv(gfake))
	else:
		isppsnr = compare_psnr(gts, gfake, data_range=1.0)
	if  mflag == 1 or  mflag == 5:
		#ispssim = compare_ssim(gts, gfake, data_range=1.0, multichannel=False)
		ispssim = compare_ssim(gts, gfake, data_range=1.0)
		gts_out = torch.from_numpy(gts)
		gfake_out = torch.from_numpy(gfake)
		inps_out = inps.cpu()[0,0,:,:]
		tt = torch.cat([gts_out,gfake_out],1).detach().cpu()[:,:]
		if i<10:
			cv2.imwrite(str(mflag)+'_'+str(i)+'temp.png',np.uint8( tt*255) )
	else:
		# print(gts.shape,gfake.shape,inps.size())
		gts_out = torch.from_numpy(gts.transpose(2,0,1))
		gfake_out = torch.from_numpy(gfake.transpose(2,0,1))
		inps_out = inps.cpu()[0,:,:,:]
		# tt = torch.cat([gts_out,gfake_out,inps_out],2).detach().cpu()[0,:,:]
		# if i<10:
		# 	cv2.imwrite(str(mflag)+'_'+str(i)+'temp.png',np.uint8( tt*255) )
		ispssim = compare_ssim(gts, gfake, data_range=1.0, channel_axis=2)
	print(isppsnr)
	totalpsnr+= isppsnr
	totalssim += ispssim
	totalnum += 1
tasks = ['nr','dm','nrdm_small','nrdm_big','srx4','srx2']
print(tasks[mflag-1] + ' mean psnr is: ' ,totalpsnr/totalnum,' ssim is: ',totalssim/totalnum)





