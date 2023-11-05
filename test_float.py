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
#from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
# state_temp_dict = torch.load(checkpointp +'G.pth')
state_temp_dict = torch.load(checkpointp +'G_raw.pth')
model.load_state_dict(state_temp_dict)
# infer
model.collapse()
# print(model)
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
		#ispssim = compare_ssim(gts, gfake, data_range=1.0, multichannel=False)
		ispssim = compare_ssim(gts, gfake, data_range=1.0)
	else:
		# print(gts.shape,gfake.shape,inps.size())
		gts_out = torch.from_numpy(gts.transpose(2,0,1))
		gfake_out = torch.from_numpy(gfake.transpose(2,0,1))
		inps_out = inps.cpu()[0,:,:,:]
		tt = torch.cat([gts_out,gfake_out,inps_out],2).detach().cpu()[0,:,:]
		if i<10:
			cv2.imwrite(str(mflag)+'_'+str(i)+'temp.png',np.uint8( tt*255) )
		ispssim = compare_ssim(gts, gfake, data_range=1.0, channel_axis=2)
	print(isppsnr)
	totalpsnr+= isppsnr
	totalssim += ispssim
	totalnum += 1
tasks = ['nr','dm','nrdm_small','nrdm_big','sr']
print(tasks[mflag-1] + ' mean psnr is: ' ,totalpsnr/totalnum,' ssim is: ',totalssim/totalnum)





