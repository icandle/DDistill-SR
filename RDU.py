# -*- coding: utf-8 -*-
import torch.nn as nn
import math

import torch
import torch.nn.functional as F

import numpy as np
import copy
import utils.arch_util as arch_util

import functools

from utils.dbb_transforms import *

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

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

# DCD
class StaticDCD(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=1,dilation=1, groups=1,
                     dynamic=True, L=None, nonlinear=None):
        super(StaticDCD,self).__init__()
        
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear        
    
        if dynamic:
            self.dy_branch = DynamicBranch(in_channels, out_channels, stride, L)
            
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        
    def forward(self, inputs):
        if not hasattr(self, 'dy_branch'):
            return self.nonlinear(self.conv(inputs))
        
        scale, res = self.dy_branch(inputs) 
        
        out = self.conv(inputs)     
        out = scale.expand_as(out)*out + res 
        return self.nonlinear(out), res            
# Dynamic DBB

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
            return self.nonlinear(out), res

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        
        out = scale.expand_as(out)*out + res 
        return self.nonlinear(out), res       

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

class ResSRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=1, dilation=1, groups=1, 
                 internal_channels_1x1_3x3=None,
                 deploy=False, dynamic=True, L=None, nonlinear=None, single_init=False):
        super(ResSRB, self).__init__()
        self.dynamic = dynamic
        
        
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear      
        
 
        self.unit = DynamicDiverseBranchBlock(in_channels, out_channels, 3, 1, 1, dilation,
                                              1,internal_channels_1x1_3x3, deploy, dynamic, L, None, single_init)
        
        
    def forward(self, inputs):
        if self.dynamic:
            out, res = self.unit(inputs)
            out += inputs
            return self.nonlinear(out), res
        else:             
            out = self.unit(inputs)
            out += inputs 
            return self.nonlinear(out)



class ResDDBB(nn.Module):
    def __init__(self, in_channels, out_channels, n_channels, kernel_size, 
                 stride=1, padding=1, dilation=1, groups=1, 
                 internal_channels_1x1_3x3=None,
                 deploy=False, dynamic=True, L=None, nonlinear=None, single_init=False):
        super(ResDDBB, self).__init__()
        self.dynamic = dynamic
        
        
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear      
        
        
        self.unit1 = DynamicDiverseBranchBlock(in_channels,n_channels, kernel_size, stride,(kernel_size-1)//2,dilation,
                                              groups,internal_channels_1x1_3x3, deploy, dynamic, L, self.nonlinear,single_init)
        self.unit2 = DynamicDiverseBranchBlock(n_channels, out_channels, 3, 1, 1, dilation,
                                              1,internal_channels_1x1_3x3, deploy, dynamic, L, None, single_init)
        
        #if self.dynamic: self.c = nn.Conv2d(out_channels+n_channels, out_channels, 1, 1 ,0)
        
    def forward(self, inputs):
        if self.dynamic:
            out, _ = self.unit1(inputs)
            out, res = self.unit2(out)
            out += inputs
            return out, res
        else:             
            out = self.unit1(inputs)
            out = self.unit2(out)
            out += inputs 
            return out

# Dynamic RepVGG

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros',nonlinear=None,  deploy=False, dynamic=True, L=None):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1
        
        #dcd: https://github.com/liyunsheng13/dcd      
        if dynamic:
            self.dy_branch = DynamicBranch(in_channels, out_channels, stride, L)    
        
        padding_11 = padding - kernel_size // 2
        
        ##
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear  

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGGBlock, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if not hasattr(self, 'dy_branch'):
            if hasattr(self, 'rbr_reparam'):
                return self.nonlinear(self.rbr_reparam(inputs))
            else:
                return self.nonlinear(self.rbr_dense(inputs)+self.rbr_1x1(inputs)+self.rbr_identity(inputs))
        
        scale,res = self.dy_branch(inputs)
        
        if hasattr(self, 'rbr_reparam'):   
            r = self.rbr_reparam(inputs)
            r = scale.expand_as(r)*r
            return self.nonlinear(r+res), res#

        else:
            r = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + self.rbr_identity(inputs)
            r = scale.expand_as(r)*r
            return self.nonlinear(r+res), res#

#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')



#Rep CondConv
class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate, temp):
        super(_routing, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)
        #self.fc.cuda()
        self.temp = temp
        
    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x/self.temp, dim = 0)
    

class DYConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', num_experts=3, dropout_rate=0.2, routing = None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DYConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))

        if routing != None:
            self._routing_fn = routing
        else:
            self._routing_fn = _routing(in_channels,num_experts,dropout_rate,30)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        self.weight1x1 = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, 1,1))
        
        self.reset_parameters()
        nn.init.kaiming_uniform_(self.weight1x1, a=math.sqrt(5))
           
    def _conv_forward(self, input, weight, padding):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        padding, self.dilation, self.groups)
    
    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            #print(routing_weights.size())
            
            kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
            kernels1x1 = torch.sum(routing_weights[: ,None, None, None, None] * self.weight1x1, 0)
            out = self._conv_forward(input, kernels,1)
            
            out = out + self._conv_forward(input, kernels1x1, 0)
            #print(self._conv_forward(input, kernels1x1, 0))
            out = out + input 
            
            res.append(out)
        return torch.cat(res, dim=0)    

class DYConvBlock(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2, routing = None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DYConvBlock, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        
        if routing != None:
            self._routing_fn = routing
        else:
            self._routing_fn = _routing(in_channels,num_experts,dropout_rate,30)
            
            
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        
        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            #print(1)
            kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            
            #print(routing_weights)
            res.append(out)
        return torch.cat(res, dim=0)
    

class DRCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, num_experts=3):
        super(DRCB, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        
        self.routing = _routing(in_channels, num_experts, 0.2, 30)
        
        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()
        if deploy:
            self.rbr_reparam = DYConvBlock(in_channels = in_channels, out_channels = out_channels, 
                                                    kernel_size =kernel_size,  stride=stride, dilation=dilation, groups=groups, num_experts= num_experts, bias=False, padding_mode=padding_mode, routing = self.routing)
        else:
            self.rbr_dense = DYConv(in_channels = in_channels, out_channels = out_channels, 
                                                    kernel_size =kernel_size,  stride=stride, dilation=dilation, groups=groups, num_experts= num_experts,  bias=False, padding_mode=padding_mode, routing = self.routing)#nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):

        if hasattr(self, 'rbr_reparam'): 
            return self.nonlinearity(self.rbr_reparam(inputs))

        return self.nonlinearity(self.rbr_dense(inputs))

    def get_equivalent_kernel_bias(self, block):
        kernel3x3,kernel1x1,kernelid, bias = self._return_tensor(block)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias 

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _return_tensor(self, branch):
        
        kernel3x3 = branch.weight
            
        kernel1x1 = branch.weight1x1
        input_dim = self.in_channels // self.groups
        kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
        for i in range(self.in_channels):
            kernel_value[i, i % input_dim, 1, 1] = 1
        self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
        kernel = self.id_tensor
        bias = branch.bias      
        return kernel3x3,kernel1x1,kernel, bias


    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias(self.rbr_dense)
        self.rbr_reparam = CondConv2D(in_channels = self.rbr_dense.in_channels, out_channels =self.rbr_dense.out_channels, 
                                        kernel_size =self.rbr_dense.kernel_size,  stride=self.rbr_dense.stride, 
                                        dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=False, padding_mode=self.rbr_dense.padding_mode, routing=self.routing)

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias = bias

   
        
        for para in self.parameters():
            para.detach_()
            
        self.__delattr__('rbr_dense')


if __name__ == '__main__':
    #from torchsummaryX import summary
    x = torch.randn(1, 32, 56, 56)
    for k in (3,7):
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
            train_y,_ = dbb(x)
            dbb.switch_to_deploy()
            deploy_y,_ = dbb(x)
            print(dbb)
            #summary(dbb, x)
            print('========================== The diff is')
            print(((train_y - deploy_y) ** 2).sum())
