from WDNet import generator
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
import os.path as osp
import os
import time
from torchvision import datasets, transforms
os.environ["CUDA_VISIBLE_DEVICES"] =  "1"

G=generator(3,3)
G.eval()
G.load_state_dict(torch.load(os.path.join('WDNet_G.pkl')))
G.cuda()
root = './dataset/CLWD/test'
imageJ_path=osp.join(root,'Watermarked_image','%s.jpg')
img_save_path=osp.join('./results','result_img','%s.jpg')
img_vision_path=osp.join('./results','result_vision','%s.jpg')
ids = list()
for file in os.listdir(root+'/Watermarked_image'):
			#if(file[:-4]=='.jpg'):
			ids.append(file.strip('.jpg'))
i=0
all_time=0.0
for img_id in ids:
  i+=1
  transform_norm=transforms.Compose([transforms.ToTensor()])
  img_J=Image.open(imageJ_path%img_id)
  img_source = transform_norm(img_J)
  img_source=torch.unsqueeze(img_source.cuda(),0)
  st=time.time()
  pred_target,mask,alpha,w,I_watermark=G(img_source)
  all_time+=time.time()-st
  mean_time=all_time/i
  print("mean time:%.3f"%mean_time)
  p0=torch.squeeze(img_source)
  p1=torch.squeeze(pred_target)
  p2=mask
  p3=torch.squeeze(w*mask)
  p2=torch.squeeze(torch.cat([p2,p2,p2],1))
  p0=torch.cat([p0,p1],1)
  p2=torch.cat([p2,p3],1)
  p0=torch.cat([p0,p2],2)
  p0=transforms.ToPILImage()(p0.detach().cpu()).convert('RGB')
  pred_target=transforms.ToPILImage()(p1.detach().cpu()).convert('RGB')
  pred_target.save(img_save_path%img_id)
  if i<=20:
    p0.save(img_vision_path%img_id)
  
  
  
  
