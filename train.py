import torch
from models import dm
from models import sesr
from models import nr
from models import nrdm_3
from models import nrdm_6
from self_dataset import TrainDataset
import os
from torch import nn
from models import quantize_utils_cuda as quantize
import cv2
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mflag = 3
# qatf = "qat_"
qatf = ""
if mflag == 1:
    model = nr.nr()
    traindata = TrainDataset(1)
    checkpointp = './model_params/nr_' + qatf
elif mflag == 2:
    model = dm.dm()
    traindata = TrainDataset(2)
    checkpointp = './model_params/dm_' + qatf
elif mflag == 3:
    model = nrdm_3.nr()
    traindata = TrainDataset(3)
    checkpointp = './model_params/nrdm_3_' + qatf
elif mflag == 4:
    model = nrdm_6.nr()
    traindata = TrainDataset(4)
    checkpointp = './model_params/nrdm_6_' + qatf
elif mflag == 5:
    model = sesr.sesr()
    traindata = TrainDataset(5)
    checkpointp = './model_params/sr_' + qatf

model = model.cuda()
loader_train = torch.utils.data.DataLoader(traindata, batch_size=32, num_workers=4,
										   shuffle=True, pin_memory=True)
torch.backends.cudnn.benchmark = True # CUDNN optimization

# Define loss
mseloss = nn.MSELoss( )#nn.MSELoss()
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001 )

# Training
model.train()

state_temp_dict = torch.load(checkpointp+ qatf +'G.pth')
model.load_state_dict(state_temp_dict)

# # qat stage
# if qatf == "qat_":
# 	quantize.prepare(model, inplace=True, a_bits=8, w_bits=8,q_type = 0,q_level="C")
#a_bits act // w_bits weight // q_type 0 symquant //q_level = L /C
for epoch in range(0, 2000):
	g_loss = 0
	if epoch > 10000:
		quantize.prepare(model, inplace=True, a_bits=8, w_bits=8, q_type=0, q_level="C")
		qatf = "qat_"
	model_optimizer.zero_grad()
	for i, data in enumerate(loader_train):
		inps,gts,_ = data[:]
		inps = inps.cuda()
		gts = gts.cuda()

		gfake = model(inps)

		gen_loss = mseloss(gfake, gts)

		g_loss += gen_loss.item()
		gen_loss.backward()
		# nn.utils.clip_grad_norm(parameters=G.parameters(), max_norm=5, norm_type=2)
		model_optimizer.step()
		model_optimizer.zero_grad()

		if epoch % 10 == 0 and i == 0:
			if mflag < 5:
				tt = torch.cat([gts,gfake,inps],3).detach().cpu()[0,0,0::2,0::2]
			else:
				tt = torch.cat([gts, gfake], 3).detach().cpu()[0, 0, 0::2, 0::2]
			torch.save(model.state_dict(), checkpointp+ qatf +'G.pth')
			cv2.imwrite(str(mflag)+'temp.png',np.uint8( tt*255) )
	print(g_loss)



