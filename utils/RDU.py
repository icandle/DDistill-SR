# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch.nn as nn
import math

import torch
import torch.nn.functional as F

import numpy as np
import copy
import utils.arch_util as arch_util

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dbb_transforms import *


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
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
 
 
def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model



def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class DynamicBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dim = None):
        super(DynamicBranch, self).__init__()
        
        if dim is None: 
            self.dim = int(math.sqrt(in_channels))
        else:
            self.dim = dim
            
        squeeze = max(in_channels, self.dim ** 2) // 16
        
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
        
        #dynamic: dcd: https://github.com/liyunsheng13/dcd
        
        if dynamic:
            self.dy_branch = DynamicBranch(in_channels, out_channels, stride, L)

            
        ## end dynamic part
        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))


            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                            kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged), (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation, groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if not hasattr(self, 'dy_branch'):
            if hasattr(self, 'dbb_reparam'):
                return self.nonlinear(self.dbb_reparam(inputs))

            out = self.dbb_origin(inputs)
            if hasattr(self, 'dbb_1x1'):
                out += self.dbb_1x1(inputs)
            out += self.dbb_avg(inputs)
            out += self.dbb_1x1_kxk(inputs)
            return self.nonlinear(out)
        
        scale, res = self.dy_branch(inputs) 
        
        if hasattr(self, 'dbb_reparam'):
            out = self.dbb_reparam(inputs)
            out = scale.expand_as(out)*out + res
            return self.nonlinear(out)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        
        out = scale.expand_as(out)*out + res 
        return self.nonlinear(out)        

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

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

class SeparableConv(nn.Module):
    def __init__(self, n_feats=50, k=3, deploy=False, dynamic=True, L =None):
        super(SeparableConv, self).__init__()
        self.separable_conv = nn.Sequential(
            DynamicDiverseBranchBlock(n_feats, 2*n_feats, k, 1, (k-1)//2, groups=n_feats, deploy=deploy, dynamic=dynamic, L =L),
            nn.ReLU(True),
            nn.Conv2d(2*n_feats, n_feats, 1, 1, 0),
        )
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.separable_conv(x)
        out += x
        out = self.act(out)

        return out

class SRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, 
                 internal_channels_1x1_3x3=None,
                 deploy=False, dynamic=True, L=None, nonlinear=None, single_init=False):
        super(SRB, self).__init__()
        self.conv = DynamicDiverseBranchBlock(in_channels, out_channels, kernel_size, stride,padding,dilation,
                                              groups,internal_channels_1x1_3x3, deploy, dynamic, L, nonlinear, single_init)
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out += x
        return self.act(out)


class DRFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, deploy = False):
        super(DRFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        

        
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c1_r = SRB(self.remaining_channels, self.rc, 3, deploy = deploy)#SeparableConv(self.rc, 7, deploy = deploy)
        self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c2_r = SRB(self.remaining_channels, self.rc, 3, deploy = deploy)#SeparableConv(self.rc, 5, deploy = deploy)
        self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c3_r = SRB(self.remaining_channels, self.rc, 3, deploy = deploy)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.act = nn.ReLU()
        self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.pa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        
        r_c4 = self.c4(r_c3)
        
        out = torch.cat((distilled_c1, distilled_c2, distilled_c3, r_c4), dim=1)
        out_fused = self.pa(self.c5(out)) 
        #print(out_fused.size())
        return out_fused+input
    
if __name__ == '__main__':
    #from torchsummaryX import summary
    x = torch.randn(1, 32, 56, 56)
    for k in (1,7):
        for s in (1,2):
            dbb = DynamicDiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=k, stride=s, padding=k//2,
                                           groups=32, deploy=False, dynamic=True, L = None )
            for module in dbb.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    nn.init.uniform_(module.running_mean, 0, 0.1)
                    nn.init.uniform_(module.running_var, 0, 0.1)
                    nn.init.uniform_(module.weight, 0, 0.1)
                    nn.init.uniform_(module.bias, 0, 0.1)
            dbb.eval()
            print(dbb)
            train_y = dbb(x)
            dbb.switch_to_deploy()
            deploy_y = dbb(x)
            print(dbb)
            #summary(dbb, x)
            print('========================== The diff is')
            print(((train_y - deploy_y) ** 2).sum())