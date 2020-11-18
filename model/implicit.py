import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torch.nn as nn
from easydict import EasyDict as edict

from . import base
import camera
import util

# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.estimator = Estimator(opt)
        self.generator = Generator(opt)
        self.renderer = Renderer(opt)

    def forward(self,opt,var,training=False):
        raise NotImplementedError

# the viewpoint estimating network
class Estimator(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        nf = 64
        self.n_cameras = 8
        self.temperature = 1
        self.cam_angles = torch.linspace(-1, 1, steps=self.n_cameras)
        feature_extractor = [
            nn.Conv2d(3, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.BatchNorm2d(nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.BatchNorm2d(nf*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.BatchNorm2d(nf*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(inplace=True),
            ]
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.fc = nn.Linear(nf*8, self.n_cameras)
        
    def forward(self,inputs):
        feat = self.feature_extractor(inputs)
        feat = feat.view(feat.shape[0], -1)
        cam_scores = self.fc(feat)
        probs = torch_F.softmax(cam_scores/self.temperature, dim=1)
        cam_angles = self.cam_angles.to(probs.device).unsqueeze(1).detach()
        output = torch.mm(probs,cam_angles)
        return output

# the implicit function
class Generator(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        # 3 + 6 * L
        point_dim = 3+(opt.impl.posenc_L*6 if opt.impl.posenc_L else 0)
        # dim before rgb and sdf head
        feat_dim = opt.arch.layers_impl[-1]
        # each is a module list, every element is a sequential to generate params of specific layer
        self.hyper_impl = self.get_module_params(opt,opt.arch.layers_impl,k0=point_dim,interm_coord=opt.arch.interm_coord)
        self.hyper_level = self.get_module_params(opt,opt.arch.layers_level,k0=feat_dim)
        self.hyper_rgb = self.get_module_params(opt,opt.arch.layers_rgb,k0=feat_dim)

    def get_module_params(self,opt,layers,k0,interm_coord=False):
        impl_params = torch.nn.ModuleList()
        # define layers that generate the params of following layers:
        # sdf: [(None, 1)]
        # rgb: [(None, 3)]
        # implicit: [(None, 128), (128, 128), (128, 128), (128, 128)]
        # None will be replaced by k0
        # Then it's [(in, out), (in, out), ...]. interm_coord means
        # feed position info into every hidden layer just like DVR
        L = util.get_layer_dims(layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = k0
            if interm_coord and li>0:
                k_in += 3+(opt.impl.posenc_L*6 if opt.impl.posenc_L else 0)
            params = self.define_hyperlayer(opt,dim_in=k_in,dim_out=k_out)
            impl_params.append(params)
        return impl_params

    def define_hyperlayer(self,opt,dim_in,dim_out):
        # return a Sequential
        # layers to generate the parameter of the implicit layers: 
        # [(None, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, None)]
        # first None will be replaced by opt.latent_dim (512)
        # last None will be replaced by (dim_in+1)*dim_out, which is weight and bias
        L = util.get_layer_dims(opt.arch.layers_hyper)
        hyperlayer = []
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = opt.latent_dim
            if li==len(L)-1: k_out = (dim_in+1)*dim_out # weight and bias
            hyperlayer.append(torch.nn.Linear(k_in,k_out))
            if li!=len(L)-1:
                hyperlayer.append(torch.nn.ReLU(inplace=False))
        hyperlayer = torch.nn.Sequential(*hyperlayer)
        return hyperlayer

    def forward(self,opt,latent):
        point_dim = 3+(opt.impl.posenc_L*6 if opt.impl.posenc_L else 0)
        feat_dim = opt.arch.layers_impl[-1]
        impl_layers = edict()
        # each one is a list of BatchLinear layer
        impl_layers.impl,self.impl_weights = self.hyperlayer_forward(opt,latent,self.hyper_impl,opt.arch.layers_impl,k0=point_dim,interm_coord=opt.arch.interm_coord,return_weights=True)
        impl_layers.level,self.level_weights = self.hyperlayer_forward(opt,latent,self.hyper_level,opt.arch.layers_level,k0=feat_dim,return_weights=True)
        impl_layers.rgb = self.hyperlayer_forward(opt,latent,self.hyper_rgb,opt.arch.layers_rgb,k0=feat_dim)
        impl_func = ImplicitFunction(opt,impl_layers)
        return impl_func

    def hyperlayer_forward(self,opt,latent,module,layers,k0,interm_coord=False,return_weights=False):
        batch_size = len(latent)
        impl_layers = []
        weights_reg = []
        L = util.get_layer_dims(layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = k0
            if interm_coord and li>0: k_in += k0
            hyperlayer = module[li]
            # get the params from the hyper layers and latent code
            # TODO: this is the place where I want to do clustering
            # maybe I want to remove layernorm, and seperate rgb and sdf
            out_raw = hyperlayer.forward(latent)
            if return_weights:
                weights_reg.append(out_raw)
            out = out_raw.view(batch_size,k_in+1,k_out)
            # use the params to build layers
            impl_layers.append(BatchLinear(weight=out[:,1:],bias=out[:,:1]))
        if return_weights: return impl_layers, weights_reg
        else: return impl_layers

class Renderer(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_ray_LSTM(opt)

    def define_ray_LSTM(self,opt):
        # feature before rgb and sdf head
        feat_dim = opt.arch.layers_impl[-1]
        # LSTM hidden_size=opt.arch.lstm_dim
        self.ray_lstm = torch.nn.LSTMCell(input_size=feat_dim,hidden_size=opt.arch.lstm_dim)
        self.lstm_pred = torch.nn.Linear(opt.arch.lstm_dim,1)
        # initialize LSTM
        for name,param in self.ray_lstm.named_parameters():
            if not "bias" in name: continue
            n = param.shape[0]
            param.data[n//4:n//2].fill_(1.)

    def forward(self,opt,impl_func,pose,intr=None,ray_idx=None):
        batch_size = len(pose)
        # in world frame, intrinsics are for perspective camera
        center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        if ray_idx is not None:
            gather_idx = ray_idx[...,None].repeat(1,1,3)
            ray = ray.gather(dim=1,index=gather_idx)
            if opt.camera.model=="orthographic":
                center = center.gather(dim=1,index=gather_idx)
        num_rays = ray_idx.shape[1] if ray_idx is not None else opt.H*opt.W
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        depth = torch.empty(batch_size,num_rays,1,device=opt.device).fill_(opt.impl.init_depth) # [B,HW,1]
        level_all = []
        state = None
        for s in range(opt.impl.srn_steps):
            points_3D = camera.get_3D_points_from_depth(opt,center,ray,depth) # [B,HW,3]
            level,feat = impl_func.forward(opt,points_3D,get_feat=True) # [B,HW,K]
            level_all.append(level)
            state = self.ray_lstm(feat.view(batch_size*num_rays,-1),state)
            delta = self.lstm_pred(state[0]).view(batch_size,num_rays,1).abs_() # [B,HW,1]
            depth = depth+delta
        points_3D_2ndlast,level_2ndlast = points_3D,level
        # final endpoint (supposedly crossing the zero-isosurface)
        points_3D = camera.get_3D_points_from_depth(opt,center,ray,depth) # [B,HW,3]
        level = impl_func.forward(opt,points_3D) # [B,HW,1]
        # can either output occupancy or sdf
        mask = level.sigmoid() if opt.impl.occup else (level<=0).float()
        level_all.append(level)
        level_all = torch.cat(level_all,dim=-1) # [B,HW,N], like [B, HW, 11], including all sdf or occ along the ray
        # get isosurface=0 intersection
        func = lambda x: impl_func.forward(opt,x)
        # if the reg doesn't work then points_3D is just close to either second last or last point
        # depending on whether both sdf are <0 or >0. The bisection would push towards one end.
        points_3D_iso0 = self.bisection(x0=points_3D_2ndlast,x1=points_3D,y0=level_2ndlast,y1=level,
                                        func=func,num_iter=opt.impl.bisection_steps) # [B,HW,3]
        level,feat = impl_func.forward(opt,points_3D_iso0,get_feat=True) # [B,HW,K]
        depth = camera.get_depth_from_3D_points(opt,center,ray,points_3D_iso0) # [B,HW,1]
        rgb = impl_func.rgb(feat).tanh_() # [B,HW,3]
        # depth is ray length here
        return rgb,depth,level,mask,level_all,points_3D_iso0 # [B,HW,K]

    def bisection(self,x0,x1,y0,y1,func,num_iter):
        for s in range(num_iter):
            x2 = (x0+x1)/2
            y2 = func(x2)
            side = ((y0<0)^(y2>0)).float() # update x0 if side else update x1
            x0,x1 = x2*side+x0*(1-side),x1*side+x2*(1-side)
            y0,y1 = y2*side+y0*(1-side),y1*side+y2*(1-side)
        x2 = (x0+x1)/2
        return x2

class ImplicitFunction(torch.nn.Module):

    def __init__(self,opt,impl_layers):
        super().__init__()
        self.opt = opt
        self.impl_layers = impl_layers
        # self.impl: a modulelist that contains sequentials, 
        # each sequential is a layer of implicit function,
        # where it consists of BatchLinear+LayerNorm+ReLU
        # self.level and self.rgb are two heads
        self.define_network(opt,impl_layers)

    def define_network(self,opt,impl_layers):
        self.impl = torch.nn.ModuleList()
        for linear in impl_layers.impl:
            layer = torch.nn.Sequential(
                linear,
                torch.nn.LayerNorm(linear.bias.shape[-1],elementwise_affine=False),
                torch.nn.ReLU(inplace=False), # avoid backprop issues with higher-order gradients
            )
            self.impl.append(layer)
        self.level = self.define_heads(opt,impl_layers.level)
        self.rgb = self.define_heads(opt,impl_layers.rgb)

    def define_heads(self,opt,impl_layers):
        layers = []
        for li,linear in enumerate(impl_layers):
            layers.append(linear)
            if li!=len(impl_layers)-1:
                layers.append(torch.nn.LayerNorm(linear.bias.shape[-1],elementwise_affine=False))
                layers.append(torch.nn.ReLU(inplace=False)) # avoid backprop issues with higher-order gradients
        return torch.nn.Sequential(*layers)

    def forward(self,opt,points_3D,get_feat=False): # [B,...,3]
        if opt.impl.posenc_L:
            # positional encoding from NeRF
            points_enc = self.positional_encoding(opt,points_3D) # [B,...,6L]
            points_enc = torch.cat([points_enc,points_3D],dim=-1) # [B,...,6L+3]
        else: points_enc = points_3D
        feat = points_enc
        # extract implicit features
        for li,layer in enumerate(self.impl):
            if opt.arch.interm_coord and li>0:
                feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
        level = self.level(feat)
        return (level,feat) if get_feat else level

    def positional_encoding(self,opt,points_3D): # [B,...,3]
        shape = points_3D.shape
        points_enc = []
        if opt.impl.posenc_L:
            freq = 2**torch.arange(opt.impl.posenc_L,dtype=torch.float32,device=opt.device)*np.pi # [L]
            spectrum = points_3D[...,None]*freq # [B,...,3,L]
            sin,cos = spectrum.sin(),spectrum.cos()
            points_enc_L = torch.cat([sin,cos],dim=-1).view(*shape[:-1],6*opt.impl.posenc_L) # [B,...,6L]
            points_enc.append(points_enc_L)
        points_enc = torch.cat(points_enc,dim=-1) # [B,...,X]
        return points_enc

class BatchLinear(torch.nn.Module):
    # use the given weight and bias to do bmm and bias sum
    def __init__(self,weight,bias=None):
        super().__init__()
        self.weight = weight
        if bias is not None:
            self.bias = bias
            assert(len(weight)==len(bias))
        else: self.bias = None
        self.batch_size = len(weight)

    def __repr__(self):
        return "BatchLinear({}, {})".format(self.weight.shape[-2],self.weight.shape[-1])

    def forward(self,x):
        assert(len(x)<=self.batch_size)
        shape = x.shape
        x = x.view(shape[0],-1,shape[-1]) # sample-wise vectorization
        y = x@self.weight[:len(x)]
        if self.bias is not None: y = y+self.bias[:len(x)]
        y = y.view(*shape[:-1],y.shape[-1]) # reshape back
        return y
