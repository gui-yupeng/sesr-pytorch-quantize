import os
import cv2
from torch.utils.data import Dataset
import time
import torch
import glob
import scipy.io as scio
import numpy as np
import random
import h5py
from scipy.io import loadmat
import torch.distributions as tdist

def aug_img_np(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.ascontiguousarray(np.flipud(img))
    elif mode == 2:
        return np.ascontiguousarray(np.rot90(img))
    elif mode == 3:
        return np.ascontiguousarray(np.flipud(np.rot90(img)))
    elif mode == 4:
        return np.ascontiguousarray(np.rot90(img, k=2))
    elif mode == 5:
        return np.ascontiguousarray(np.flipud(np.rot90(img, k=2)))
    elif mode == 6:
        return np.ascontiguousarray(np.rot90(img, k=3))
    elif mode == 7:
        return np.ascontiguousarray(np.flipud(np.rot90(img, k=3)))

def crop_img_np(rgb, patch_size, center_crop=True):
    # crop
    w, h, _ = rgb.shape
    if not (w, h) == (patch_size, patch_size):
        if not center_crop:
            i = random.randint(0, h - patch_size)
            j = random.randint(0, w - patch_size)
        else:
            i = h//2 - patch_size//2
            j = w//2 - patch_size//2
        rgb = rgb[i:i+patch_size, j:j+patch_size, :]
    return rgb

def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    red = image[0, 0::2, 0::2]
    green_red = image[1, 0::2, 1::2]
    green_blue = image[1, 1::2, 0::2]
    blue = image[2, 1::2, 1::2]
    out = torch.stack((red, green_red, green_blue, blue), dim=0)
    return out

def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
    shot_noise = torch.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    n = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26]))
    log_read_noise = line(log_shot_noise) + n.sample()
    read_noise = torch.exp(log_read_noise)
    return shot_noise, read_noise

def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
    noise = n.sample()
    out = image + noise
    return out

def metadata2tensor(metadata):
    xyz2cam = torch.FloatTensor(metadata['colormatrix'])
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    cam2rgb = torch.inverse(rgb2cam)

    red_gain = torch.FloatTensor(metadata['red_gain'])
    blue_gain = torch.FloatTensor(metadata['blue_gain'])

    return cam2rgb.squeeze(), red_gain.squeeze().unsqueeze(0), blue_gain.squeeze().unsqueeze(0)

def four2three( four_tensor):
    cc,ww,hh =four_tensor.size()
    three_tensor = torch.zeros((3,ww*2,hh*2))
    three_tensor[0, 0::2, 0::2] = four_tensor[0, :, :]
    three_tensor[1, 1::2, 0::2] = four_tensor[1, :, :]
    three_tensor[1, 0::2, 1::2] = four_tensor[2, :, :]
    three_tensor[2, 1::2, 1::2] = four_tensor[3, :, :]
    return three_tensor
'''*******************************
        define load data
 *******************************'''
class TrainDataset(Dataset):
    def __init__(self, mflag=1):
        self.proot ='/home/guiyp/Work/sesr/train/train_rggb/'
        self.rggb = []
        self.ps = 128
        self.mflag = mflag
        for tt in glob.glob(self.proot+'*.mat'):
            # img = scio.loadmat(tt)['mat_crop'] #rggb,int16
            # img = np.int16( np.array(img) )
            # img = np.clip(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 0, 1)
            self.rggb.append(tt)
    def __len__(self):
        return len(self.rggb)
    def __getitem__(self, index):
        tt = self.rggb[index]
        img = scio.loadmat(tt)['mat_crop']  # rggb,int16
        img = np.array(img) / (2**14-1.) #14bit

        WW, HH = img.shape[:2]
        bii = np.random.randint(0, WW - self.ps)
        bjj = np.random.randint(0, HH - self.ps)
        img_patch = img[bii:bii + self.ps, bjj:bjj + self.ps, :]
        linrgb = np.stack((img_patch[:, :, 0], \
               np.mean(img_patch[:, :, 1:3], axis=-1), img_patch[:, :, 3]), axis=2)
        linrgb = np.clip( aug_img_np(linrgb, random.randint(0, 7)) ,0,1)
        if self.mflag == 5: #sr
            linrgb = linrgb ** (1/2.2)
            gt = 0.299 * linrgb[:, :, 0] + 0.587 * linrgb[:, :, 1] + 0.114 * linrgb[:, :, 2]
            inp = cv2.resize(gt,(0,0),fx=1/4.,fy=1/4., interpolation=cv2.INTER_CUBIC)
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            inp = torch.from_numpy(inp).float().unsqueeze(0)
            variance = 0
        else:
            linrgb = torch.from_numpy(linrgb.transpose(2, 0, 1).copy()).float()
            linrgb = torch.clamp(linrgb, 0., 1.)
            inp = mosaic(linrgb)
            shot_noise, read_noise = random_noise_levels()
            if self.mflag == 2: #dm
                gt = linrgb
                inp = four2three(inp)
                variance = 0
            if self.mflag == 1: #nr
                gt = four2three(inp)
                inp = add_noise(inp, shot_noise, read_noise)
                variance = shot_noise * inp + read_noise
                inp = four2three(inp)
            if self.mflag == 3 or self.mflag == 4:
                gt = linrgb
                inp = add_noise(inp, shot_noise, read_noise)
                variance = shot_noise * inp + read_noise
                inp = four2three(inp)
        return torch.clamp(inp,0,1), torch.clamp(gt,0,1), variance
# self.proot = '/home/data/yiqing.zh/test/'
class TestDataset(Dataset):
    def __init__(self, mflag=1):
        # self.proot ='/home/guiyp/Work/sesr/DataSet/DIV2K_valid_HR/'
        self.proot ='/home/guiyp/Work/sesr/DataSet/Set5/GTmod12/'
        self.rggb = []
        self.ps = 128
        self.mflag = mflag
        for tt in glob.glob(self.proot+'*.png'):
            # img = scio.loadmat(tt)['mat_crop'] #rggb,int16
            # img = np.int16( np.array(img) )
            # img = np.clip(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 0, 1)
            self.rggb.append(tt)
    def __len__(self):
        return len(self.rggb)
    def __getitem__(self, index):
        tt = self.rggb[index]

        # with h5py.File(tt) as matfile:
        #     rggb = np.asarray(matfile['raw']).astype(np.float32) / (2 ** 14- 1.)
        #     img = np.transpose(rggb, (2, 1, 0))
        #     # matainfo = matfile['metadata']
        #     # matainfo = {'colormatrix': np.transpose(matainfo['colormatrix']),
        #     #                 'red_gain': matainfo['red_gain'],
        #     #                 'blue_gain': matainfo['blue_gain'] }
        #     # ccm, red_g, blue_g = process.metadata2tensor(matainfo)
        #     # metadata = {'ccm': ccm, 'red_gain': red_g, 'blue_gain': blue_g}

        # # WW, HH = img.shape[:2]
        # # bii = np.random.randint(0, WW - self.ps)
        # # bjj = np.random.randint(0, HH - self.ps)
        # # img_patch = img[bii:bii + self.ps, bjj:bjj + self.ps, :]
        # linrgb = np.stack((img[:, :, 0], \
        #        np.mean(img[:, :, 1:3], axis=-1), img[:, :, 3]), axis=2)

        # DNDM需要-1参数    img = cv2.imread(tt,-1)
        img = np.array(cv2.imread(tt)[:,:,::-1])
        # print(img.shape)
        # linrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = img/255.0
        if self.mflag == 5: #sr_x4
            # 仅Y in Y out
            inp = np.array(cv2.imread(tt.replace('GTmod12','LRbicx4') )[:,:,::-1] )/255.
            gt = (65.481 * gt[:, :, 0] + 128.553 * gt[:, :, 1] + 24.966 * gt[:, :, 2] + 16.)/255.0
            gt = np.clip(gt,0,1)
            inp = (65.481 * inp[:, :, 0] + 128.553 * inp[:, :, 1] + 24.966 * inp[:, :, 2] + 16.)/255.0
            inp = np.clip(inp,0,1)
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            inp = torch.from_numpy(inp).float().unsqueeze(0)
            variance = 0
            # print(gt.shape,inp.shape)
        elif self.mflag == 6: #sr_x2
            # 只有pixelshift200需要这步
            # linrgb = linrgb ** (1/2.2)
            # gt = 0.299 * linrgb[:, :, 0] + 0.587 * linrgb[:, :, 1] + 0.114 * linrgb[:, :, 2]
            # gt = (65.481 * linrgb[:, :, 0] + 128.533 * linrgb[:, :, 1] + 24.966 * linrgb[:, :, 2] + 16.)/255
            # RGB in RGB out
            gt = np.clip(gt,0,1)
            inp = np.array(cv2.imread(tt.replace('GTmod12','LRbicx2') )[:,:,::-1] )/255.
            # inp = cv2.resize(gt,(0,0),fx=1/2.,fy=1/2., interpolation=cv2.INTER_CUBIC)
            gt = torch.from_numpy(gt).float().permute(2,0,1)
            inp = torch.from_numpy(inp).float().permute(2,0,1)
            variance = 0
        else:
            linrgb = torch.from_numpy(linrgb.transpose(2, 0, 1).copy()).float()
            linrgb = torch.clamp(linrgb, 0., 1.)
            inp = mosaic(linrgb)
            shot_noise, read_noise = random_noise_levels()
            if self.mflag == 2: #dm
                gt = linrgb
                inp = four2three(inp)
                variance = 0
            if self.mflag == 1: #nr
                gt = four2three(inp)
                inp = add_noise(inp, shot_noise, read_noise)
                variance = shot_noise * inp + read_noise
                inp = four2three(inp)
            if self.mflag == 3 or self.mflag == 4:
                gt = linrgb
                inp = add_noise(inp, shot_noise, read_noise)
                variance = shot_noise * inp + read_noise
                inp = four2three(inp)

        return torch.clamp(inp,0,1), torch.clamp(gt,0,1), variance