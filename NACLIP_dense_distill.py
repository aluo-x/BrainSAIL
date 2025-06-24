#!/usr/bin/env python
# coding: utf-8

# In[8]:





# In[9]:





# In[1]:


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
# import clip
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


# In[22]:


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-s','--split', help='Description for foo argument', required=False,default=1)
args = parser.parse_args()
print("Selecting split", args.split)

cur_split = int(args.split)


# In[2]:


import urllib

import io
import numpy as np
from PIL import Image
from math import floor
desired_size = 224*4

def shuffle_shift(input_image, offset_x,offset_y, infill = -100):
    orig_shape = input_image.shape
    temp = input_image[:,:, max(0,offset_x):min(orig_shape[2], orig_shape[2]+offset_x), max(0,offset_y):min(orig_shape[3], orig_shape[3]+offset_y)]
    temp = torch.nn.functional.pad(temp, (max(0, -offset_y),max(0,offset_y), max(0, -offset_x), max(0,offset_x)), value=infill)
    return temp

OPENAI_CLIP_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None]
OPENAI_CLIP_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None]

def normalize_image_deterministic(image_resized):
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))/(255.0)
    # print(scaled_image.shape, "shape")
    return (scaled_image - OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD
# print((1-OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD)

# _ = normalize_image_deterministic(current_image)
# print(OPENAI_CLIP_MEAN.shape, OPENAI_CLIP_STD.shape)
# del _


# In[3]:


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
counts = 51
patch_size = 16
num_patches = img_size//patch_size
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


device = "cuda:0"
try:
    del net
except:
    pass
net, _ = clip.load("ViT-B/16", device=device, jit=False)
__ = net.eval()
old_patch_pos_embed = torch.clone(net.visual.positional_embedding.detach())
attn_strategy='naclip'
gaussian_std=10.
net.visual.set_params("reduced", attn_strategy, gaussian_std)


# In[5]:


old_patch_pos_embed = torch.clone(net.visual.positional_embedding.detach())

patch_pos_embed = old_patch_pos_embed[1:]
patch_size=16
orig_size = 224
desired_size = 224*4
print(patch_pos_embed.shape)
print(patch_pos_embed.reshape(1, orig_size // patch_size, orig_size // patch_size, 768).permute(0, 3, 1, 2).shape)

patch_pos_embed_v2 = torch.nn.functional.interpolate(
                        patch_pos_embed.reshape(1, orig_size // patch_size, orig_size // patch_size, 768).permute(0, 3, 1, 2),
                        size=(desired_size // patch_size, desired_size // patch_size),
                        mode='bicubic',
                        align_corners=False, recompute_scale_factor=False
                    )
flat = patch_pos_embed_v2.permute(0, 2, 3, 1).view(1, -1, 768)
net.visual.positional_embedding.data = torch.cat((old_patch_pos_embed[:1], flat[0]), dim=0)


# In[10]:


images = h5py.File("/scratch/image_data_512.h5py", "r")


# In[25]:


try:
    saver.close()
except:
    pass
saver = h5py.File("/scratch/naclip_{}.h5py".format(cur_split), "w")


# In[20]:


image_keys = sorted(list(images.keys()))

def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))

my_stuff = list(chunker_list(image_keys, 5))
image_keys = my_stuff[cur_split]
print("Total keys", len(image_keys))


# In[17]:


# smoothed.shape, smoothed.dtype


# In[26]:


final_accumulator = np.zeros((56, 56, 512)).astype(np.single)
count_accumulator = np.zeros((56,56)).astype(np.single)
norm_accumulator = np.zeros((56, 56, counts)).astype(np.single)
norm_accumulator[:] = np.nan
existence_accumulator = np.zeros((56,56,counts)).astype(bool)
temp_accumulator = np.zeros((56, 56, 512)).astype(np.single)

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
            offset_image[0,0,offset_image[0,0]<-50] = 1.9303361
            offset_image[0,1,offset_image[0,1]<-50] = 2.0748837
            offset_image[0,2,offset_image[0,2]<-50] = 2.145897
            # start.record()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                image_features = net.encode_image(offset_image, return_all=True)
            image_features = image_features[:, 1:]
            accumulator.append(image_features[0]+0.0)
            offset_accumulator.append(offset_patch[0]+0)
            offset_idx += 1
        
        final_accumulator[:] = 0.0
        count_accumulator[:] = 0.0
        
        for i in range(counts):
            # temp_accumulator[:] = 0.0
            current_feat = accumulator[i].cpu().float().numpy().reshape(56,56,-1).astype(np.single)
            current_offset = np.moveaxis(offset_accumulator[i].cpu().numpy(), [0,1,2], [2,0,1])
            valid = np.all(current_offset>-0.5, axis=2)
            select = current_feat[valid]
            select_offset = current_offset[valid]
            final_accumulator[select_offset[:,0], select_offset[:,1]] += select
            # count_accumulator[select_offset[:,0], select_offset[:,1]] += 1
        smoothed = final_accumulator*10.0
        saver.create_dataset(img_k, data=smoothed.astype(np.float16))
        # smoothed = accumulator[0].cpu().float().numpy().reshape(56,56,-1).astype(np.single)
        # out = smoothed/np.linalg.norm(smoothed, axis=-1, keepdims=True)
        # pca = PCA(3)
        # out = pca.fit_transform(out.reshape(-1, 512))
        # out = out.reshape(56,56,3)
        # out = out-np.min(out, axis=-1,keepdims=True)
        # out = out/np.max(out, axis=-1, keepdims=True)
        # plt.imshow(out)
        # plt.show()

        # # smoothed = final_accumulator*10.0
        # smoothed = accumulator[0].cpu().float().numpy().reshape(56,56,-1).astype(np.single)
        # out = smoothed/np.linalg.norm(smoothed, axis=-1, keepdims=True)
        # pca = PCA(3)
        # out = pca.fit_transform(out.reshape(-1, 512))
        # out = out.reshape(56,56,3)
        # out = out-np.min(out, axis=-1,keepdims=True)
        # out = out/np.max(out, axis=-1, keepdims=True)
        # plt.imshow(out)
        # plt.show()
        # saver.create_dataset("init", data=arr)
saver.close()

