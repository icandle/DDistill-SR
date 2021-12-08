import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.ops
import numpy as np
import math
import functools
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()

        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
       
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
    
    def forward(self, x):

        offset = self.offset_conv(x)

        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,                               
                                          )
        return x
    
    
# for RCAN
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        
# for other networks
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

        
class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

def scalex4(im):
    '''Nearest Upsampling by myself'''
    im1 = im[:, :1, ...].repeat(1, 16, 1, 1)
    im2 =  im[:, 1:2, ...].repeat(1, 16, 1, 1)
    im3 = im[:, 2:, ...].repeat(1, 16, 1, 1)
    
#     b, c, h, w = im.shape
#     w = torch.randn(b,16,h,w).cuda() * (5e-2)
    
#     img1 = im1 + im1 * w
#     img2 = im2 + im2 * w
#     img3 = im3 + im3 * w
    
    imhr = torch.cat((im1, im2, im3), 1)
    imhr = F.pixel_shuffle(imhr, 4)
    return imhr


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


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result



class RepDyBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepDyBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1
        
        #dcd: https://github.com/liyunsheng13/dcd
    
        self.dim = int(math.sqrt(in_channels))
        squeeze = max(in_channels, self.dim ** 2) // 16

        self.q = nn.Conv2d(in_channels, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, out_channels, 1, 1, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, out_channels, bias=False)
        self.hs = Hsigmoid()  
        
        padding_11 = padding - kernel_size // 2
        
        ##
        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepDyBlock, identity = ', self.rbr_identity)


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
        
        if hasattr(self, 'rbr_reparam'):
            
            r = self.rbr_reparam(inputs)
            r = scale.expand_as(r)*r
            
            return self.nonlinearity(r+res)#


   
        r = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + self.rbr_identity(inputs)

        r = scale.expand_as(r)*r
        
        return self.nonlinearity(r+res)#



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
    
class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, deploy = False):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c1_r = RepDyBlock(in_channels, self.rc, 3, deploy = deploy)
        self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c2_r = RepDyBlock(self.remaining_channels, self.rc, 3, deploy = deploy)
        self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c3_r = RepDyBlock(self.remaining_channels, self.rc, 3, deploy = deploy)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.act = nn.ReLU()
        self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.pa = ESA(in_channels, nn.Conv2d)#PA(in_channels)#self.esa = 

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
        return out_fused



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
    

class CondConvBlock(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', num_experts=3, dropout_rate=0.2, routing = None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConvBlock, self).__init__(
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

class CondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2, routing = None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
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
    

class RepCondConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_pa=True, num_experts=3):
        super(RepCondConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        
        self.routing = _routing(in_channels, num_experts, 0.2, 30)
        
        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()

        if use_pa:
            self.pa = PA(out_channels)
        else:
            self.pa = nn.Identity()
        if deploy:
            self.rbr_reparam = CondConv2D(in_channels = in_channels, out_channels = out_channels, 
                                                    kernel_size =kernel_size,  stride=stride, dilation=dilation, groups=groups, num_experts= num_experts, bias=False, padding_mode=padding_mode, routing = self.routing)
        else:
            self.rbr_dense = CondConvBlock(in_channels = in_channels, out_channels = out_channels, 
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

class RFDB_CC(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, deploy = False):
        super(RFDB_CC, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c1_r = RepCondConv(in_channels, self.rc, 3, deploy = deploy)
        self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c2_r = RepCondConv(self.remaining_channels, self.rc, 3, deploy = deploy)
        self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c3_r = RepCondConv(self.remaining_channels, self.rc, 3, deploy = deploy)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.act = nn.ReLU()
        self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.pa = ESA(in_channels, nn.Conv2d)# PA(in_channels)# self.esa =

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
        return out_fused



class SRB(nn.Module):
    def __init__(self, in_channels):
        super(SRB,self).__init__()
        self.conv =  nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True, dilation = 1, groups=1)
    def forward(self, input):
        return input + self.conv(input)
 
class RFDB_static(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, deploy = False):
        super(RFDB_static, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c1_r = SRB(self.rc)
        #self.c1_r = nn.Conv2d(self.remaining_channels, self.rc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c2_r = SRB(self.rc)
        self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c3_r = SRB(self.rc)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.act = nn.ReLU()
        self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.pa = ESA(in_channels, nn.Conv2d)#self.esa = PA(in_channels)#

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
        return out_fused
'''    
class RFDB_static(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, deploy = False):
        super(RFDB_static, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c1_r = nn.Conv2d(self.remaining_channels, self.rc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.c2_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c2_r = nn.Conv2d(self.remaining_channels, self.rc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.c3_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.c3_r = nn.Conv2d(self.remaining_channels, self.rc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1, bias=True, dilation = 1, groups=1)
        self.act = nn.ReLU()
        self.c5 = nn.Conv2d(4*self.dc, in_channels, 1, 1, 0, bias=True, dilation = 1, groups=1)
        self.pa = self.esa = PA(in_channels)#ESA(in_channels, nn.Conv2d)#

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
        return out_fused'''