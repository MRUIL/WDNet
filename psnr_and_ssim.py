import pytorch_ssim
from PIL import Image
import torch
import math
from scipy.ndimage import gaussian_filter
import numpy
from numpy.lib.stride_tricks import as_strided as ast
import cv2
import os.path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "1"
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
"""
Hat tip: http://stackoverflow.com/a/5078155/1828289
"""
def mse(img1, img2):
    mse=numpy.mean( (img1 - img2) ** 2 )
    return mse
def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)
def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)
real_img_root="./dataset/CLWD/test/Watermark_free_image"
mask_img_root="./dataset/CLWD/test/Mask"
g_img_root='./results/result_img'
g_img_path=osp.join(g_img_root,'%s.jpg')
real_img_path=osp.join(real_img_root,'%s.jpg')
mask_path=osp.join(mask_img_root,'%s.png')
ids = list()
for file in os.listdir(g_img_root):
			#if(file[:-4]=='.jpg'):
			ids.append(file.strip('.jpg'))
i=0
ans_ssim=0.0
ans_psnr=0.0
rmse_all=0.0
rmse_in=0.0
for img_id in ids:
  i+=1
  mask = Image.open(mask_path%img_id)
  mask=numpy.asarray(mask)/255.0
  #print(mask.shape)
  real_img=cv2.imread(real_img_path%img_id)
  #print(real_img.shape)
  g_img=cv2.imread(g_img_path%img_id)
  real_img_tensor=torch.from_numpy(real_img).float().unsqueeze(0)/255.0
  g_img_tensor=torch.from_numpy(g_img).float().unsqueeze(0)/255.0
  real_img_tensor=real_img_tensor.cuda()
  g_img_tensor=g_img_tensor.cuda()
  
  ans_psnr+=psnr(g_img,real_img)
  mse_all=mse(g_img,real_img)
  mse_in=mse(g_img*mask,real_img*mask)*mask.shape[0]*mask.shape[1]*mask.shape[2]/(numpy.sum(mask)+1e-6)
  rmse_all+=numpy.sqrt(mse_all)
  rmse_in+=numpy.sqrt(mse_in)
  ans_ssim+=pytorch_ssim.ssim(g_img_tensor,real_img_tensor)
  print(i)
  print('psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f'%(ans_psnr/i,ans_ssim/i,rmse_in/i,rmse_all/i))
  
      
