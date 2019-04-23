import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import math
import time
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import data_parallel


class _Residual_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()
        
        midc=int(outc*scale)
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 

class Encoder(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc = nn.Linear((cc)*4*4, 2*hdim)           
    
    def forward(self, x):        
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)                
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Decoder, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        cc = channels[-1]
        self.fc = nn.Sequential(
                      nn.Linear(hdim, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        
        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
                    
    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y
  
        
class IntroVAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(IntroVAE, self).__init__()         
        
        self.hdim = hdim
        
        self.encoder = Encoder(cdim, hdim, channels, image_size)
        
        self.decoder = Decoder(cdim, hdim, channels, image_size)
        
      
    def forward(self, x):        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)        
        return mu, logvar, z, y
        
    def sample(self, z):        
        y = self.decode(z)
        return y
    
    def encode(self, x):  
        mu, logvar = data_parallel(self.encoder, x)
        return mu, logvar
        
    def decode(self, z):        
        y = data_parallel(self.decoder, z)
        return y
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5) # (batch, 2)
        return v_kl
    
    def reconstruction_loss(self, prediction, target, size_average=False):        
        error = (prediction - target).view(prediction.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=-1)
        
        if size_average:
            error = error.mean()
        else:
            error = error.sum()
               
        return error
        

