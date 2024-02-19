# -*- coding: utf-8 -*-
import torch.nn as nn
import math

import torch
import torch.nn.functional as F

import numpy as np
import copy


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class PA(nn.Module):
    '''PA: pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m

class DynamicBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dim = None):
        super(DynamicBranch, self).__init__()
        
        if dim is None: 
            self.dim = int(math.sqrt(in_channels))
        else:
            self.dim = dim
            
        squeeze = max(in_channels, self.dim ** 2) // 16 #max(16, self.dim)
        
        self.q = nn.Conv2d(in_channels, self.dim, 1, stride, 0, bias=False)
        self.p = nn.Conv2d(self.dim, out_channels, 1, 1, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze, bias=False),
            SEModule_small(squeeze),)
            
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, out_channels, bias=False)
        self.hs = Hsigmoid()  
        
    def forward(self, inputs):
        b, c, _, _= inputs.size()
        
        y = self.avg_pool(inputs).view(b, c)
        y = self.fc(y)
        
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b,-1,1,1)
   
        res = self.bn1(self.q(inputs))       
        _, _, h, w = res.size()
        res = res.view(b,self.dim,-1)       
        res = self.bn2(torch.matmul(phi, res)) + res       
        res = self.p(res.view(b,-1,h,w))
        
        return scale, res


class DynamicDiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, 
                 internal_channels_1x1_3x3=None,
                 deploy=False, dynamic=True, L=None, nonlinear=None, single_init=False):
        super(DynamicDiverseBranchBlock, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2
        assert deploy == True
        
        if dynamic:
            self.dy_branch = DynamicBranch(in_channels, out_channels, stride, L)

        self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

    def forward(self, inputs):
        scale, res = self.dy_branch(inputs) 
        
        out = self.dbb_reparam(inputs)
        out = scale.expand_as(out)*out + res
        return self.nonlinear(out), res   


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
        
        assert style =='DBB'
        
        self.unit = DynamicDiverseBranchBlock(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,internal_channels_1x1_3x3,deploy,dynamic,L,nonlinear,single_init)

        if nonlinear == None:
            self.act = nn.Identity()
        else:
            self.act = nonlinear
            
    def forward(self, inputs):
        out, res = self.unit(inputs)
        if self.res: return out, res
        else: return out
            
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
            self.conv1x1 = nn.Conv2d(4*in_channels, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
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

        scale = self.sigmoid(self.PAconv(out_fused))
        res = self.conv1x1(torch.cat((input, res1, res2, res3), dim=1))
        res = torch.mul(scale, res)

        out_fused = self.esa(out_fused + res) 
        return out_fused , res
        

        

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

class DDistill_SR_Inference(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=56, num_modules=4, scale=4, lite=False, deploy=True,dynamic=True,L=None,style='DBB',res=True):
        super(DDistill_SR_Inference, self).__init__()
        assert res == True
        assert deploy == True

        self.scale = scale 
        self.fea_conv = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.res = res
        
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
        
        upsample_block = pixelshuffle_block_cpc

        if self.scale == 2:
            self.upsamplerx2 = upsample_block(nf, out_nc, self.scale)
        elif self.scale ==3:
            self.upsamplerx3 = upsample_block(nf, out_nc, self.scale)
        else: 
            self.upsamplerx4 = upsample_block(nf, out_nc, self.scale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        
        out_B1, res1 = self.B1(out_fea)
        out_B2, res2 = self.B2(out_B1)
        out_B3, res3 = self.B3(out_B2)
        out_B4, res4 = self.B4(out_B3)
            
        out_res = self.cres(torch.cat([res1, res2, res3, res4], dim=1))
            
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))

            
        scale = self.sigmoid(self.PAconv(out_B))
        out_res = torch.mul(scale, out_res)
            
        out_lr = self.LR_conv(out_B + out_res) + out_fea 

            
        if self.scale == 2: 
            output = self.upsamplerx2(out_lr)
        elif self.scale == 3:
            output = self.upsamplerx3(out_lr)
        else:
            output = self.upsamplerx4(out_lr)
                    
        return output.clamp(0,1)
    
def model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

if __name__ == "__main__":
    
    model = DDistill_SR_Inference(in_nc=3, out_nc=3, nf = 56, num_modules=4, scale=4, deploy=True, dynamic=True, L = 16, style = 'DBB')
    model.load_state_dict(torch.load(r'model_zoo/DDistill-SRx4_inference.pth'),True)

    x = torch.randn((1,3,256,256))
    y = model(x)

    print(y.shape)
