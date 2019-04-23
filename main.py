from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import *
from math import log10
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--lr_e', type=float, default=0.0002, help='learning rate of the encoder, default=0.0002')
parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate of the generator, default=0.0002')
parser.add_argument("--num_vae", type=int, default=0, help="the epochs of pretraining a VAE, Default=0")
parser.add_argument("--weight_neg", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--weight_rec", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--weight_kl", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--m_plus", type=float, default=100.0, help="the margin in the adversarial part, Default=100.0")
parser.add_argument('--channels', default="64, 128, 256, 512, 512, 512", type=str, help='the list of channel numbers')
parser.add_argument("--hdim", type=int, default=512, help="dim of the latent code, Default=512")
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=1000, help="Default=1000")
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=8)
parser.add_argument('--trainfiles', default="celeba_hq_attr.list", type=str, help='the list of training files')
parser.add_argument('--dataroot', default="/home/huaibo.huang/data/celeba-hq/celeba-hq-wx-256", type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data', default=28000)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--input_height', type=int, default=128, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=None, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=128, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_height', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--clip', type=float, default=100, help='the threshod for clipping gradient')
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='results/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', action='store_true', help='enables tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg",".bmp"])
    
def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=opt.nrow), cur_iter)
    

def main():
    
    global opt, model
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    is_scale_back = False
    
    #--------------build models -------------------------
    model = IntroVAE(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height).cuda()    
    if opt.pretrained:
        load_model(model, opt.pretrained)
    print(model)
            
    optimizerE = optim.Adam(model.encoder.parameters(), lr=opt.lr_e)
    optimizerG = optim.Adam(model.decoder.parameters(), lr=opt.lr_g)
    
    #-----------------load dataset--------------------------
    image_list = [x for x in listdir(opt.dataroot) if is_image_file(x)]  
    train_list = image_list[:opt.trainsize]
    assert len(train_list) > 0
    
    train_set = ImageDatasetFromFile(train_list, opt.dataroot, input_height=None, crop_height=None, output_height=opt.output_height, is_mirror=True)     
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
            
    cur_iter = 0
    
    def train_vae(epoch, iteration, batch, cur_iter):  
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)
            
        batch_size = batch.size(0)
                       
        real= Variable(batch).cuda() 
                
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================                  
        real_mu, real_logvar, z, rec = model(real) 
        
        loss_rec =  model.reconstruction_loss(rec, real, True)        
        loss_kl = model.kl_loss(real_mu, real_logvar).mean()
                    
        loss = loss_rec + loss_kl
        
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        loss.backward()                   
        optimizerE.step() 
        optimizerG.step()
     
        info += 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec.data[0], loss_kl.data[0])       
        print(info)
        
        if cur_iter % opt.test_iter is 0:  
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec], cur_iter)   
            else:
                vutils.save_image(torch.cat([real, rec], dim=0).data.cpu(), '{}/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)    
    
    def train(epoch, iteration, batch, cur_iter):  
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)
            
        batch_size = batch.size(0)
        
        noise = Variable(torch.zeros(batch_size, opt.hdim).normal_(0, 1)).cuda() 
               
        real= Variable(batch).cuda() 
        
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================ 
        fake = model.sample(noise)            
        real_mu, real_logvar, z, rec = model(real)
        rec_mu, rec_logvar = model.encode(rec.detach())
        fake_mu, fake_logvar = model.encode(fake.detach())
        
        loss_rec =  model.reconstruction_loss(rec, real, True)
        
        lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()
        lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossE_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()            
        loss_margin = lossE_real_kl + \
                      (F.relu(opt.m_plus-lossE_rec_kl) + \
                      F.relu(opt.m_plus-lossE_fake_kl)) * 0.5 * opt.weight_neg
        
                    
        lossE = loss_rec  * opt.weight_rec + loss_margin * opt.weight_kl
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        lossE.backward(retain_graph=True)
        # nn.utils.clip_grad_norm(model.encoder.parameters(), 1.0)            
        optimizerE.step()
        
        #========= Update G ==================           
        rec_mu, rec_logvar = model.encode(rec)
        fake_mu, fake_logvar = model.encode(fake)
        
        lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossG_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()
        
        lossG = (lossG_rec_kl + lossG_fake_kl)* 0.5 * opt.weight_kl      
                    
        # optimizerG.zero_grad()
        lossG.backward()
        # nn.utils.clip_grad_norm(model.decoder.parameters(), 1.0)
        optimizerG.step()
     
        info += 'Rec: {:.4f}, '.format(loss_rec.data[0])
        info += 'Kl_E: {:.4f}, {:.4f}, {:.4f}, '.format(lossE_real_kl.data[0], 
                                lossE_rec_kl.data[0], lossE_fake_kl.data[0])
        info += 'Kl_G: {:.4f}, {:.4f}, '.format(lossG_rec_kl.data[0], lossG_fake_kl.data[0])
       
        print(info)
        
        if cur_iter % opt.test_iter is 0:            
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec, fake], cur_iter)   
            else:
                vutils.save_image(torch.cat([real, rec, fake], dim=0).data.cpu(), '{}/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)             
            
                  
    #----------------Train by epochs--------------------------
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        #save models
        save_epoch = (epoch//opt.save_iter)*opt.save_iter   
        save_checkpoint(model, save_epoch, 0, '')
        
        model.train()
        
        for iteration, batch in enumerate(train_data_loader, 0):
            #--------------train------------
            if epoch < opt.num_vae:
                train_vae(epoch, iteration, batch, cur_iter)
            else:
                train(epoch, iteration, batch, cur_iter)
            
            cur_iter += 1
            
def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
            
def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "model/" + prefix +"model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()    