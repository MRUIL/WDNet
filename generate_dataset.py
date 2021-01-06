import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
#from compiler.ast import flatten
CUDA_VISIBLE_DEVICES=3
mark='train'
root_logo = './CLWD/watermark_logo/white/'
root_dataset='./pascal_data/train/VOC2012/JPEGImages/'
root_train='./CLWD/'
img_ids=list()
img_path=osp.join(root_dataset,'%s.jpg')
img_source_path=osp.join(root_train,mark,'Watermarked_image','%s.jpg')
img_target_path=osp.join(root_train,mark,'Watermark_free_image','%s.jpg')
balance_path=osp.join(root_train,mark,'Loss_balance','%s.png')
mask_path=osp.join(root_train,mark,'Mask','%s.png')
alpha_path=osp.join(root_train,mark,'Alpha','%s.png')
W_path=osp.join(root_train,mark,'Watermark','%s.png')
logo_path=osp.join(root_logo,mark,'%s.png')
for file in os.listdir(root_dataset):
	img_ids.append(file.strip('.jpg'))
def solve_mask(img,img_target):
	img1=np.asarray(img.permute(1,2,0).cpu())
	#print(img1)
	img2=np.asarray(img_target.permute(1,2,0).cpu())
	#print(img2)
	img3=abs(img1-img2)
	#print(img3)
	mask=img3.sum(2)>(15.0/255.0)
	mask=mask.astype(int)
	#print('oooooooooooooooooooooo')
	#print(mask)
	return mask
def solve_balance(mask):
	height,width=mask.shape
	k=mask.sum()
	#print(k)
	k=(int)(k)

	mask2=(1.0-mask)*np.random.rand(height,width)
	mask2=mask2.flatten()
	pos=np.argsort(mask2)
	balance=np.zeros(height*width)
	balance[pos[:min(250*250,4*k)]]=1
	balance=balance.reshape(height,width)
	return balance
i=(int)(0)
while i<60000:
	for id in img_ids:
		print(i)
		if i>=60000:
			break
		img=Image.open(img_path%id)
		logo_id=str(random.randint(0,63)).zfill(3)
		logo=Image.open(logo_path%logo_id)
		logo = logo.convert('RGBA')
		img_height,img_width=img.size
		img=img.resize((256,256))
		save_id=str(i+1)
		img.save(img_target_path%save_id)#save target image
		
		rotate_angle=random.randint(0,360)
		logo_rotate=logo.rotate(rotate_angle,expand = True)
		logo_height,logo_width=logo_rotate.size
		logo_height=random.randint(10,256)
		logo_width=random.randint(10,256)
		logo_resize=logo_rotate.resize((logo_height,logo_width))
		transform_totensor=transforms.Compose([transforms.ToTensor()])
		#print(logo_resize.size)
		img=transform_totensor(img)
		logo=transform_totensor(logo_resize)
		img=img.cuda()
		logo=logo.cuda()
		alpha=random.random()*0.4+0.3
		start_height=random.randint(0,256-logo_height)
		start_width=random.randint(0,256-logo_width)
		W=torch.zeros_like(img)
		img_target=img.clone()
		#print(img.shape)
		#print(logo.shape)
		#print(logo_width)
		#print(logo_height)
		img[:,start_width:start_width+logo_width,start_height:start_height+logo_height]=img[:,start_width:start_width+logo_width,start_height:start_height+logo_height]*(1.0-alpha*logo[3:4,:,:])+logo[:3,:,:]*alpha*logo[3:4,:,:]
		
		mask=solve_mask(img,img_target)
		#print(mask)
		cv2.imwrite(mask_path%save_id,np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]),2)*256.0)
		balance=solve_balance(mask)
		cv2.imwrite(balance_path%save_id,np.concatenate((balance[:,:,np.newaxis],balance[:,:,np.newaxis],balance[:,:,np.newaxis]),2)*256.0)

		W[:,start_width:logo_width+start_width,start_height:start_height+logo_height]+=logo[:3,:,:]
		img=transforms.ToPILImage()(img.cpu()).convert('RGB')
		#mask=transforms.ToPILImage()(mask).convert('RGB')
		W=transforms.ToPILImage()(W.cpu()).convert('RGB')
		balance=solve_balance(mask)

		#balance=transforms.ToPILImage()(balance).convert('RGB')
		img.save(img_source_path%save_id)
		#mask.save(mask_path%save_id)

		alpha=alpha*mask
		cv2.imwrite(alpha_path%save_id,np.concatenate((alpha[:,:,np.newaxis],alpha[:,:,np.newaxis],alpha[:,:,np.newaxis]),2)*256.0)
		
		W.save(W_path%save_id)
		#balance.save(balance_path%save_id)
		i=i+1
