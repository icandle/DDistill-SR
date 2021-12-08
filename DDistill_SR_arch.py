# -*- coding: utf-8 -*-
import torch.nn as nn
import math

import torch
import torch.nn.functional as F

import numpy as np
import copy

from utils.arch_util import ESA, PA

from RDU import DynamicDiverseBranchBlock as DDBB
from RDU import RepVGGBlock as DRVB
from RDU import DRCB
from RDU import ResDDBB,ResSRB,StaticDCD
import functools


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, 
                 internal_channels_1x1_3x3=None, num_experts=4, padding_mode='zeros',
                 deploy=False, dynamic=True, L=None, nonlinear=None, single_init=False, style = 'DBB', res=False, n_channels = None):
        super(Unit,self).__init__()
        
        #res = true: dynamic = true, style in DBB, VGG
        if n_channels is None:
            n_channels = in_channels
        
        self.style = style
        self.dynamic = dynamic
        self.res = res
        
        if style =='DBB':
            self.unit = DDBB(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,internal_channels_1x1_3x3,deploy,dynamic,L,nonlinear,single_init)
        elif style =='VGG':
            self.unit = DRVB(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,padding_mode,nonlinear,deploy,dynamic,L)
        elif style =='CC':
            self.unit = DRCB(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,padding_mode,deploy,num_experts)
        elif style =='ResDBB':
            self.unit = ResDDBB(in_channels,out_channels,n_channels,kernel_size,stride,padding,dilation,groups,internal_channels_1x1_3x3,deploy,dynamic,L,nonlinear,single_init)
        elif style =='SRD':
            self.unit = ResSRB(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,internal_channels_1x1_3x3,deploy,dynamic,L,nonlinear,single_init)
        elif style =='StaticDCD':
            self.unit = StaticDCD(in_channels, out_channels, kernel_size,stride,padding,dilation,groups,dynamic,L,nonlinear)
        else:
            self.unit = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,padding_mode)
            
        if nonlinear == None:
            self.act = nn.Identity()
        else:
            self.act = nonlinear
            
    def forward(self, inputs):
        if self.style == 'SRB':
            return self.act(inputs+self.unit(inputs))
        elif self.style == 'Static':
            return self.act(self.unit(inputs))        
            
      
        if self.dynamic and self.style in {'DBB','VGG','ResDBB','SRD','StaticDCD'}:
            out, res = self.unit(inputs)
            if self.res: return out, res
            else: return out
        else: return self.unit(inputs) 
            
class SeparableConv(nn.Module):
    def __init__(self, n_feats=50, k=3,deploy=True,dynamic=True,L=None,style='DBB',res=False, same = False):
        super(SeparableConv, self).__init__()
        self.res = res
        
        if not same:
            nf_feats = 2*n_feats
        else:
            nf_feats = n_feats
            
        if self.res:
             self.conv2 =nn.Conv2d(nf_feats, n_feats, 1, 1, 0)
             
        self.unit = Unit(n_feats,nf_feats,k,1,(k-1)//2,groups=n_feats,deploy=deploy,dynamic=dynamic,L=L,style=style,res =res)   
        self.conv1 =nn.Conv2d(nf_feats, n_feats, 1, 1, 0)
       
        self.act = nn.LeakyReLU(0.05, True)

    def forward(self, x):
        if self.res:
            out,res = self.unit(x)
            out = self.conv1(self.act(out))
            res = self.conv2(res)
            out += x
            out = self.act(out)
            res = self.act(res)
            return out, res
        else:
            out = self.unit(x)
            out = self.conv1(self.act(out))
            out += x
            out = self.act(out)
            return out           

class DFDB(nn.Module):
    def __init__(self, in_channels,deploy=True,dynamic=True,L=None,style='DBB',res=True):
        super(DFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.res = res
        
        self.act = nn.LeakyReLU(0.05, True)
        
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c1_r = Unit(self.remaining_channels, self.rc, 3, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res,nonlinear=self.act)
        self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c2_r = Unit(self.remaining_channels, self.rc, 3, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res,nonlinear=self.act)
        self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c3_r = Unit(self.remaining_channels, self.rc, 3, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res,nonlinear=self.act)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        
        if self.res:
            self.PAconv = nn.Conv2d(in_channels, in_channels, 1)
            ##add sconv to enhance res
            #self.sconv = SeparableConv(in_channels,5,deploy,dynamic,L)
            
            self.conv1x1 = nn.Conv2d(4*in_channels, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.sigmoid = nn.Sigmoid()
            
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):  
        if self.res:
            distilled_c1 = self.act(self.c1_d(input))
            r_c1,res1 = self.c1_r(input)

            distilled_c2 = self.act(self.c2_d(r_c1))
            r_c2,res2 = self.c2_r(r_c1)

            distilled_c3 = self.act(self.c3_d(r_c2))
            r_c3,res3 = self.c3_r(r_c2)
        
            r_c4 = self.c4(r_c3)
        
            out = torch.cat((distilled_c1, distilled_c2, distilled_c3, r_c4), dim=1)
            
            out_fused = self.c5(out)
            
            # dfdb res
            scale = self.sigmoid(self.PAconv(out_fused))
            res = self.conv1x1(torch.cat((input, res1, res2, res3), dim=1))
            res = torch.mul(scale, res)
            
            out_fused = self.esa(out_fused + res) 
            #print(out_fused.size())
            return out_fused , res
        
        else: 
            distilled_c1 = self.act(self.c1_d(input))
            r_c1 = self.c1_r(input)

            distilled_c2 = self.act(self.c2_d(r_c1))
            r_c2 = self.c2_r(r_c1)

            distilled_c3 = self.act(self.c3_d(r_c2))
            r_c3 = self.c3_r(r_c2)
        
            r_c4 = self.c4(r_c3)
        
            out = torch.cat((distilled_c1, distilled_c2, distilled_c3, r_c4), dim=1)
            out_fused = self.esa(self.c5(out)) 
          
            return out_fused
        

class DFDB_lite(nn.Module):
    def __init__(self, in_channels,deploy=True,dynamic=True,L=None,style='DBB',res=True):
        super(DFDB_lite, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.res = res
        
        if style == 'ResDBB':
            self.act = nn.LeakyReLU(0.05, True)
            self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c1_r = Unit(self.remaining_channels, self.rc, 7, groups=self.rc, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res,nonlinear=self.act,n_channels=2*self.rc)
            self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c2_r = Unit(self.remaining_channels, self.rc, 5, groups=self.rc, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res,nonlinear=self.act,n_channels=2*self.rc)
            self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c3_r = Unit(self.remaining_channels, self.rc, 3, groups=1, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res,nonlinear=self.act,n_channels=2*self.rc)
            self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        
        else:
            self.act = nn.LeakyReLU(0.05, True)
            self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c1_r = SeparableConv(self.remaining_channels, 7, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res)
            self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c2_r = SeparableConv(self.remaining_channels, 5, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res)
            self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c3_r = Unit(self.remaining_channels, self.rc, 3, deploy=deploy,dynamic=dynamic,L=L,style=style,res=res,nonlinear=self.act)
            self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
            self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        
        if self.res:
            self.PAconv = nn.Conv2d(in_channels, in_channels, 1)
            self.conv1x1 = nn.Conv2d(4*in_channels, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
            #self.sconv = SeparableConv(in_channels,5,deploy,dynamic,L,same = True)
            self.sigmoid = nn.Sigmoid()
            
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1,res1 = self.c1_r(input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2,res2 = self.c2_r(r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3,res3 = self.c3_r(r_c2)
        
        r_c4 = self.c4(r_c3)
        
        out = torch.cat((distilled_c1, distilled_c2, distilled_c3, r_c4), dim=1)
        out_fused = self.c5(out)
        
        if self.res:
            scale = self.sigmoid(self.PAconv(out_fused))
            res = self.conv1x1(torch.cat((input, res1, res2, res3), dim=1))
            res = torch.mul(scale, res)
            
            out_fused = self.esa(out_fused + res)
            return out_fused , res
        
        else: return self.esa(out_fused) 

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    #pa = PA(in_channels)
    conv = nn.Conv2d(in_channels, out_channels* (upscale_factor ** 2), 3, 1, 1, bias=True)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
   
    return nn.Sequential(conv, pixel_shuffle)

def pixelshuffle_block_cpc(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    #pa = PA(in_channels)
    conv = nn.Conv2d(in_channels, out_channels* (upscale_factor ** 2), 3, 1, 1, bias=True)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    #new add
    conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
    
    return nn.Sequential(conv, pixel_shuffle, conv1)

class DDistill_SR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=50, num_modules=4, scale=4,lite=False, deploy=True,dynamic=True,L=None,style='DBB',res=True):
        super(DDistill_SR, self).__init__()
            
        self.scale = scale 
        self.fea_conv = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.res = res
        
        
        if lite:
            self.B1 = DFDB_lite(nf,deploy,dynamic,L,style,res)
            self.B2 = DFDB_lite(nf,deploy,dynamic,L,style,res)
            self.B3 = DFDB_lite(nf,deploy,dynamic,L,style,res)
            self.B4 = DFDB_lite(nf,deploy,dynamic,L,style,res)
        else:
            self.B1 = DFDB(nf,deploy,dynamic,L,style,res)
            self.B2 = DFDB(nf,deploy,dynamic,L,style,res)
            self.B3 = DFDB(nf,deploy,dynamic,L,style,res)
            self.B4 = DFDB(nf,deploy,dynamic,L,style,res)

        if self.res:
            self.PAconv = nn.Conv2d(nf, nf, 1)
            self.sigmoid = nn.Sigmoid()
            self.cres = nn.Sequential(
                nn.Conv2d(num_modules*nf, nf, 1, 1, 0, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )

            
        self.c =  nn.Sequential(
            nn.Conv2d(num_modules*nf, nf, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.LR_conv = Unit(nf, nf, 3, deploy=deploy,dynamic=dynamic,L=L,style=style)   
        #self.LR_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.esa = ESA(nf, nn.Conv2d)
        
        upsample_block = pixelshuffle_block_cpc
        #self.upsampler = upsample_block(nf, out_nc, self.scale)
        if self.scale == 2:
            self.upsamplerx2 = upsample_block(nf, out_nc, self.scale)
        elif self.scale ==3:
            self.upsamplerx3 = upsample_block(nf, out_nc, self.scale)
        else: 
            self.upsamplerx4 = upsample_block(nf, out_nc, self.scale)
        #self.FConv = nn.Conv2d(out_nc, out_nc, 3, 1, 1, bias=True)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        
        if self.res:
            out_B1, res1 = self.B1(out_fea)
            out_B2, res2 = self.B2(out_B1)
            out_B3, res3 = self.B3(out_B2)
            out_B4, res4 = self.B4(out_B3)
            
            out_res = self.cres(torch.cat([res1, res2, res3, res4], dim=1))
            
            out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
            #out_B = self.esa(out_B)
            
            scale = self.sigmoid(self.PAconv(out_B))
            out_res = torch.mul(scale, out_res)
            
            out_lr = self.LR_conv(out_B + out_res) + out_fea 

        else:
            out_B1 = self.B1(out_fea)
            out_B2 = self.B2(out_B1)
            out_B3 = self.B3(out_B2)
            out_B4 = self.B4(out_B3)
   
            out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
            #out_B = self.esa(out_B)
            out_lr = self.LR_conv(out_B) + out_fea 
            
        if self.scale == 2: 
            output = self.upsamplerx2(out_lr)
        elif self.scale == 3:
            output = self.upsamplerx3(out_lr)
        else:
            output = self.upsamplerx4(out_lr)
                    
        return output.clamp(0,1)
    
def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

if __name__ == "__main__":
    
    model = DDistill_SR(in_nc=3, out_nc=3, nf = 56, num_modules=4, scale=4, deploy=False, dynamic=True, L = 16, style = 'DBB')
    model.load_state_dict(torch.load(r'model-backup/RepDSRx4.pth'))
    #experiments\DFDN_DBBx4\models\245000_G.pth
    from PIL import Image
    import torchvision.transforms as transforms
    model.eval()
    
    model_inference = repvgg_model_convert(model)
 
    img = Image.open(r'Set5/LR_bicubic/X4/babyx4.png').convert('RGB')#35.jpg
        #img.show()vid.png
        #img = transforms.Resize([540, 960])(img)Set14/LR_bicubic/X4/ppt3x4.png
    x = transforms.ToTensor()(img)    
    x = x.unsqueeze(0)
    out1 = model(x)
    out2 = model_inference(x)
    #print(model)
    
    img1 = transforms.ToPILImage()(out1.squeeze(0))
    img1.show()
    img2 = transforms.ToPILImage()(out2.squeeze(0))
    img2.show()    
    f = torch.nn.L1Loss()
    
    z = f(out1,out2)
    print(z)