#!/usr/bin/env python
# coding: utf-8

# In[3]:


import h5py
import os
os.environ['HF_HOME'] = '/scratch/cache'
os.environ["TORCH_HOME"] = '/scratch/cache'
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
from transformers import AutoModel, CLIPImageProcessor
import inspect
from pprint import pprint
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import RandomResizedCrop
from torchvision.transforms.v2.functional import resized_crop
import torch
import matplotlib.pyplot as plt
import random
import sys
# sys.path.append("/home/afluo/aug_avg/SCLIP")
# sys.path.append("/home/afluo/aug_avg/clip")
# import clipa
import gc
import numpy as np
import os
import torch
from torch import nn
from sklearn.decomposition import PCA
sys.path.insert(0,"/home/afluo/aug_avg/NACLIP")
import clip
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import tqdm

from time import time
import argparse


# In[2]:


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-s','--split', help='Description for foo argument', required=False,default=1)
args = parser.parse_args()
print("Selecting split", args.split)

cur_split = int(args.split)


# In[36]:


import urllib

import io
import numpy as np
from PIL import Image
from math import floor
desired_size = 518

def shuffle_shift(input_image, offset_x,offset_y, infill = -100):
    orig_shape = input_image.shape
    temp = input_image[:,:, max(0,offset_x):min(orig_shape[2], orig_shape[2]+offset_x), max(0,offset_y):min(orig_shape[3], orig_shape[3]+offset_y)]
    temp = torch.nn.functional.pad(temp, (max(0, -offset_y),max(0,offset_y), max(0, -offset_x), max(0,offset_x)), value=infill)
    return temp

OPENAI_CLIP_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None]
OPENAI_CLIP_STD = np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None]

def normalize_image_deterministic(image_resized):
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))/(255.0)
    # print(scaled_image.shape, "shape")
    return (scaled_image - OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD
# print((1-OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD)

# _ = normalize_image_deterministic(current_image)
# print(OPENAI_CLIP_MEAN.shape, OPENAI_CLIP_STD.shape)
# del _


# In[76]:


torch.manual_seed(42)
torch.cuda.manual_seed(42)
def farthest_point_sample(xy, npoint):
    """
    Input:
        xy: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xy.device
    B, N, C = xy.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xy[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xy - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


img_size = desired_size
counts = 25
patch_size = 14
num_patches = img_size//patch_size
print(num_patches)
shift_frac = 0.15
rng = np.random.default_rng(seed=42)

shift_patch = np.rint(shift_frac*num_patches).astype(np.int32)
assert img_size%patch_size == 0

rand_x = rng.integers(-shift_patch, shift_patch+1, 5000)
rand_y = rng.integers(-shift_patch, shift_patch+1, 5000)
all_offset = np.stack([rand_x, rand_y]).T[None]
selected = farthest_point_sample(torch.from_numpy(all_offset).float(), counts)
selected_offset = np.rint(all_offset[0][selected[0].numpy()]).astype(np.int32)
selected = selected_offset

selected[0][:] = 0
patch_idx_offset = np.arange(0, desired_size//patch_size)
good_y_patch, good_x_patch = np.meshgrid(patch_idx_offset, patch_idx_offset)
indices_offset = torch.from_numpy(np.stack([good_x_patch, good_y_patch])[None])
multiplied_chunk = selected*patch_size
flip = rng.integers(0,2,counts)
flip[:1] = 0


# In[4]:


# device = "cuda:0"
# try:
#     del net
# except:
#     pass
# net, _ = clip.load("ViT-B/16", device=device, jit=False)
# __ = net.eval()
# old_patch_pos_embed = torch.clone(net.visual.positional_embedding.detach())
# attn_strategy='naclip'
# gaussian_std=10.
# net.visual.set_params("reduced", attn_strategy, gaussian_std)


# In[7]:


net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
_ = net.eval()
device = "cuda:0"
net.to(device)


# In[21]:


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        # attn = q @ k.transpose(-2, -1)

        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # print(attn.shape, v.shape, (attn @ v).shape)
        x = (v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# In[33]:


myattn = Attention(dim = net.blocks[-1].attn.qkv.in_features,
                   num_heads = net.blocks[-1].attn.num_heads,
                   qkv_bias = True,
                   proj_bias=True)

myattn.to(net.blocks[-1].attn.qkv.weight.device)
myattn.load_state_dict(net.blocks[-1].attn.state_dict())
myattn.custom = True
net.blocks[-1].attn = myattn


# In[78]:


print("Valid attn", net.blocks[-1].attn.custom)


# In[17]:


images = h5py.File("/scratch/image_data_512.h5py", "r")


# In[18]:


try:
    saver.close()
except:
    pass
cur_split = 1
saver = h5py.File("/scratch/dino.h5py", "w")


# In[81]:


image_keys = sorted(list(images.keys()))

# def chunker_list(seq, size):
#     return (seq[i::size] for i in range(size))

# my_stuff = list(chunker_list(image_keys, 5))
# image_keys = my_stuff[cur_split]
print("Total keys", len(image_keys))


# In[20]:


# # smoothed.shape, smoothed.dtype
# 518/14


# In[ ]:


# array([[[2.2489083]],

#        [[2.4285715]],

#        [[2.64     ]]], dtype=float32)


# In[73]:


# plt.imshow(count_accumulator)


# In[80]:


# num_patch = 37
final_accumulator = np.zeros((num_patches, num_patches, 768)).astype(np.single)
count_accumulator = np.zeros((num_patches,num_patches)).astype(np.single)
norm_accumulator = np.zeros((num_patches, num_patches, counts)).astype(np.single)
norm_accumulator[:] = np.nan
existence_accumulator = np.zeros((num_patches,num_patches,counts)).astype(bool)
temp_accumulator = np.zeros((num_patches, num_patches, 768)).astype(np.single)

with torch.inference_mode():
    for img_k in tqdm.tqdm(image_keys):
        try:
            del current_image
        except:
            pass
        current_image = images[img_k][:]
        try:
            del current_I
            del torch_img
        except:
            pass

        try:
            del accumulator
            del offset_accumulator
        except:
            pass
        accumulator = []
        offset_accumulator = []

        current_I = Image.fromarray(current_image).resize((desired_size,desired_size))
        # display(current_I)
        img = normalize_image_deterministic(np.array(current_I))
        torch_img = torch.from_numpy(img[None]).to(device)
        offset_idx = 0
        
        for chunk in multiplied_chunk:
            flip_val = flip[offset_idx]
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            

            offset_image = shuffle_shift(torch_img, chunk[0], chunk[1])
            offset_patch  = shuffle_shift(indices_offset, selected[offset_idx][0], selected[offset_idx][1], infill=-1)

            
            if flip_val:
                offset_image = torch.flip(offset_image, [3])
                offset_patch = torch.flip(offset_patch, [3])
            offset_image[0,0,offset_image[0,0]<-50] = 2.2489083
            offset_image[0,1,offset_image[0,1]<-50] = 2.4285715
            offset_image[0,2,offset_image[0,2]<-50] = 2.64
            # start.record()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                image_features = net(offset_image, is_training=True)
            image_features = image_features["x_norm_patchtokens"]
            # print(image_features.shape, "SHAPEEEEEEE")
            # assert False
            accumulator.append(image_features[0]+0.0)
            offset_accumulator.append(offset_patch[0]+0)
            offset_idx += 1
        
        final_accumulator[:] = 0.0
        count_accumulator[:] = 0.0
        
        for i in range(counts):
            # temp_accumulator[:] = 0.0
            current_feat = accumulator[i].cpu().float().numpy().reshape(37,37,-1).astype(np.single)
            current_offset = np.moveaxis(offset_accumulator[i].cpu().numpy(), [0,1,2], [2,0,1])
            valid = np.all(current_offset>-0.5, axis=2)
            select = current_feat[valid]
            select_offset = current_offset[valid]
            final_accumulator[select_offset[:,0], select_offset[:,1]] += select
            count_accumulator[select_offset[:,0], select_offset[:,1]] += 1
        # assert False
        smoothed = final_accumulator*10.0
        saver.create_dataset(img_k, data=smoothed.astype(np.float16))
        # smoothed = accumulator[0].cpu().float().numpy().reshape(37,37,-1).astype(np.single)
        # out = smoothed/np.linalg.norm(smoothed, axis=-1, keepdims=True)
        # pca = PCA(3)
        # out = pca.fit_transform(out.reshape(-1, 768))
        # out = out.reshape(37,37,3)
        # out = out-np.min(out, axis=-1,keepdims=True)
        # out = out/np.max(out, axis=-1, keepdims=True)
        # plt.imshow(out)
        # plt.show()

        # # smoothed = final_accumulator*10.0
        # # smoothed = accumulator[0].cpu().float().numpy().reshape(56,56,-1).astype(np.single)
        # smoothed = accumulator[0].cpu().float().numpy().reshape(37,37,-1).astype(np.single)
        # out = smoothed/np.linalg.norm(smoothed, axis=-1, keepdims=True)
        # pca = PCA(3)
        # out = pca.fit_transform(out.reshape(-1, 768))
        # out = out.reshape(37,37,3)
        # out = out-np.min(out, axis=-1,keepdims=True)
        # out = out/np.max(out, axis=-1, keepdims=True)
        # plt.imshow(out)
        # plt.show()
        # assert False
        # saver.create_dataset("init", data=arr)
saver.close()


# In[70]:


# smoothed = accumulator[0].cpu().float().numpy().reshape(37,37,-1).astype(np.single)
# out = smoothed/np.linalg.norm(smoothed, axis=-1, keepdims=True)
# pca = PCA(3)
# out = pca.fit_transform(out.reshape(-1, 768))
# out = out.reshape(37,37,3)
# out = out-np.min(out, axis=-1,keepdims=True)
# out = out/np.max(out, axis=-1, keepdims=True)
# plt.imshow(out)
# plt.show()


# In[ ]:




