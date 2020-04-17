from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections
import torchvision.transforms as transforms


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # # print(image_tensor.shape)
    # # image_numpy = image_tensor[0].cpu().float().numpy()
    # # if image_numpy.shape[0] == 1:
    # #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # # # print(image_numpy.shape)
    # # mean = np.zeros(image_numpy.shape)
    # # mean[0, :, :] = (0.485/0.229)
    # # mean[1, :, :] = (0.456/0.224)
    # # mean[2, :, :] = (0.406/0.225)
    # # std = mean
    # # std[0, :, :] = 0.229
    # # std[1, :, :] = 0.224
    # # std[2, :, :] = 0.225
    # # mean = mean.transpose((1, 2, 0))
    # # std = std.transpose((1, 2, 0))
    # # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + mean) * std * 255.0
    # flag = True
    # image_tensor =  image_tensor[0]
    # if image_tensor.shape[0]==1:
    #     image_tensor = torch.cat([image_tensor,image_tensor,image_tensor],dim = 0)
    #     flag = False
    #     dn_img = image_tensor
    # # print('image_tensor')
    # # print(image_tensor.shape)
    # else:
    #     denorm = transforms.Normalize(mean=[-1 * (0.485 / 0.229), -1 * (0.456 / 0.224), -1 * (0.406 / 0.225)],
    #                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    #     dn_img = denorm(image_tensor)
    # # print('dn_img')
    # # print(dn_img.shape)
    # image_numpy = dn_img.cpu().float().numpy()
    # image_numpy[image_numpy<0]=0.0
    # image_numpy[image_numpy>1]=1.0
    # if flag==False:
    #     image_numpy = 1.0-image_numpy
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255.0)
    # # image_numpy = image_numpy.astype(np.uint8)
    # # image_pil = Image.fromarray(image_numpy)
    # # image_pil.save('./'+str(1)+'.jpg')
    # return image_numpy.astype(imtype)
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)



def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
