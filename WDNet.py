import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from unet_parts import *
from tensorboardX import SummaryWriter
from vgg import Vgg16
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dilation=nn.Sequential(
          nn.Conv2d(512,512,3,1,2, dilation=2),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512,512,3,1,4, dilation=4),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512,512,3,1,6, dilation=6),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          )
        self.outw = OutConv(64, 3)
        self.outa = OutConv(64, 1)
        self.out_mask = OutConv(64, 1)
        self.sg=nn.Sigmoid()
        self.other=OutConv(64, 64)
        self.post_process_1=nn.Sequential(
          nn.Conv2d(64+6, 64, 3, 1, 1),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.2),
          nn.Conv2d(64, 128, 3, 1, 1),
          )
        self.post_process_2=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_3=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_4=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_5=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 3, 3, 1, 1),
          nn.Sigmoid(),
          )
    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dilation(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        w = self.outw(x)
        a = self.outa(x)
        other = self.other(x)
        other = self.sg(other)
        mask = self.out_mask(x)
        mask=self.sg(mask)
        a=self.sg(a)
        w=self.sg(w)
        a=mask*a
        I_watermark=(x0-a*w)/(1.0-a+1e-6)
        I_watermark=torch.clamp(I_watermark,0,1)
        xx1=self.post_process_1(torch.cat([other,I_watermark,x0],1))
        xx2=self.post_process_2(xx1)
        xx3=self.post_process_3(xx1+xx2)
        xx4=self.post_process_4(xx2+xx3)
        I_watermark2=self.post_process_5(xx4+xx3)
        I=I_watermark2*mask+(1.0-mask)*x0
        return I,mask,a,w,I_watermark

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=3, output_dim=1):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.input_size = input_size
        #self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        #utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        return x

class WDNet(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.input_size = args.input_size
        self.model_name = args.gan_type
        self.z_dim = 62
        self.class_num = 3
        self.sample_num = self.class_num ** 2

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]
        def weight_init(m):
          classname=m.__class__.__name__
          if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
          elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)
        # networks init
        self.G = generator(3, 3)
        self.D = discriminator(input_dim=6, output_dim=1)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.l1loss=nn.L1Loss().cuda()
            self.loss_mse = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
        self.G.apply(weight_init)
        self.D.apply(weight_init)
        #self.load()
        print('---------- Networks architecture -------------')
        #utils.print_network(self.G)
        #utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        #self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        #if self.gpu_mode:
            #self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()
        vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.D.train()
        print('training start!!')
        start_time = time.time()
        writer=SummaryWriter(log_dir='log/ex_WDNet')
        lenth=self.data_loader.dataset.__len__()
        iter_all=0
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_, mask, balance, alpha, w) in enumerate(self.data_loader):
                iter_all+=1#iter+epoch*(lenth//self.batch_size)
                if iter ==  lenth// self.batch_size:
                    break
                #y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                #y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, self.input_size, self.input_size)
                if self.gpu_mode:
                    x_, y_, mask, balance, alpha, w=x_.cuda(), y_.cuda(), mask.cuda(), balance.cuda(), alpha.cuda(), w.cuda()
                    #x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

                # update D network
                if ((iter+1)%3)==0 :
                  self.D_optimizer.zero_grad()
  
                  D_real = self.D(x_, y_)
                  D_real_loss = self.BCE_loss(D_real, torch.ones_like(D_real))
  
                  G_ ,g_mask, g_alpha, g_w,I_watermark= self.G(x_)
                  D_fake = self.D(x_,G_)
                  D_fake_loss = self.BCE_loss(D_fake, torch.zeros_like(D_fake))
  
                  D_loss = 0.5*D_real_loss + 0.5*D_fake_loss
                  #self.train_hist['D_loss'].append(D_loss.item())
                  D_writer=D_loss.item()
                  D_loss.backward()
                  self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ ,g_mask, g_alpha, g_w,I_watermark= self.G(x_)
                D_fake = self.D(x_, G_)
                G_loss = self.BCE_loss(D_fake, torch.ones_like(D_fake))
                feature_G = vgg(G_)
                feature_real=vgg(y_)
                vgg_loss=0.0
                for j in range (3):
                  vgg_loss+= self.loss_mse(feature_G[j],feature_real[j])
                #self.train_hist['G_loss'].append(G_loss.item())
                mask_loss=self.l1loss(g_mask*balance,mask*balance)*balance.size(0)*balance.size(1)*balance.size(2)*balance.size(3)/balance.sum()
                w_loss=self.l1loss(g_w*mask,w*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                alpha_loss=self.l1loss(g_alpha*mask,alpha*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                I_watermark_loss=self.l1loss(I_watermark*mask,y_*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                I_watermark2_loss=self.l1loss(G_*mask,y_*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                G_writer=G_loss.data
                G_loss=G_loss+10.0*mask_loss+10.0*w_loss+10.0*alpha_loss+50.0*(0.7*I_watermark2_loss+0.3*I_watermark_loss)+1e-2*vgg_loss
                G_loss.backward()
                self.G_optimizer.step()
                if((iter+1)%100) ==0:
                    writer.add_scalar('G_Loss', G_writer, iter_all)
                    writer.add_scalar('D_Loss', D_loss.item(), iter_all)
                    writer.add_scalar('W_Loss', w_loss, iter_all)
                    writer.add_scalar('alpha_Loss', alpha_loss, iter_all)
                    writer.add_scalar('mask_Loss', mask_loss, iter_all)
                    writer.add_scalar('I_watermark_Loss', I_watermark_loss, iter_all)
                    writer.add_scalar('I_watermark2_Loss', I_watermark2_loss, iter_all)
                    writer.add_scalar('vgg_Loss', vgg_loss, iter_all)
                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_writer))
            self.save()
        print("Training finish!... save training results")

        self.save()
    def save(self):
        torch.save(self.G.state_dict(), os.path.join('WDNet_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join('WDNet_D.pkl'))

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join('WDNet_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join('WDNet_D.pkl')))
