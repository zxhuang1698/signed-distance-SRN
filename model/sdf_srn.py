import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import tqdm
from easydict import EasyDict as edict

from . import implicit
import camera
import eval_3D
import util,util_vis
from util import log
from copy import deepcopy

# ============================ main engine for training and evaluation ============================

class Model(implicit.Model):

    def __init__(self,opt):
        super().__init__(opt)

    @torch.no_grad()
    def evaluate(self,opt,ep=None,training=False):
        self.graph.eval()
        loss_eval = edict()
        metric_eval = dict(dist_acc=0.,dist_cov=0.)
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)
        for it,batch in enumerate(loader):
            var = edict(batch)
            var,loss = self.evaluate_batch(opt,var,ep,it)
            for key in loss:
                loss_eval.setdefault(key,0.)
                loss_eval[key] += loss[key]*len(var.idx)
            dist_acc,dist_cov = eval_3D.compute_chamfer_dist(opt,var)
            metric_eval["dist_acc"] += dist_acc*len(var.idx)
            metric_eval["dist_cov"] += dist_cov*len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))
            if it==0 and training: 
                var_viz = edict(deepcopy(self.viz_data))
                var_viz,_ = self.evaluate_batch(opt,var_viz,ep,it)
                _,_ = eval_3D.compute_chamfer_dist(opt,var_viz)
                self.visualize(opt,var_viz,step=ep,split="eval")
                self.dump_results(opt,var_viz,train=True)
            if not training: self.dump_results(opt,var,write_new=(it==0))
        for key in loss_eval: loss_eval[key] /= len(self.test_data)
        for key in metric_eval: metric_eval[key] /= len(self.test_data)
        log.loss_eval(opt,loss_eval,chamfer=(metric_eval["dist_acc"],
                                             metric_eval["dist_cov"]))
        if training:
            # log/visualize results to tb/vis
            self.log_scalars(opt,var,loss_eval,metric=metric_eval,step=ep,split="eval")

    def evaluate_batch(self,opt,var,ep=None,it=None):
        var = util.move_to_device(var,opt.device)
        if opt.optim.test_optim:
            # if test-time optimization is enabled
            # only optimize the latent code
            latent_enc = self.graph.encoder(var.rgb_input_map)
            latent = torch.nn.Parameter(latent_enc.detach())
            var,loss = self.evaluate_inner_optim(opt,var,latent,ep=ep,it=it)
        else:
            var = self.graph.forward(opt,var,training=False)
            loss = self.graph.compute_loss(opt,var,training=False)
            loss = self.summarize_loss(opt,var,loss)
        return var,loss

    @torch.enable_grad()
    def evaluate_inner_optim(self,opt,var,latent,ep=None,it=None):
        optim_list = [dict(params=latent,lr=opt.optim.lr_test),]
        optim = self.optimizer(optim_list)
        iterator = tqdm.tqdm(range(opt.optim.iter_test),desc="test-time optim.",leave=False,position=1)
        for it2 in iterator:
            optim.zero_grad()
            var.latent = latent
            var = self.graph.forward(opt,var,training=False)
            loss = self.graph.compute_loss(opt,var,training=False)
            loss = self.summarize_loss(opt,var,loss)
            loss.all.backward()
            optim.step()
            iterator.set_postfix(loss="{:.3f}".format(loss.all))
            # log test-time optimization losses
            if it==0 and opt.tb:
                it2_show = opt.optim.iter_test*((ep+1)//opt.freq.eval)+it2
                if it2==0:
                    for k in loss: loss[k] = np.nan
                    self.log_scalars(opt,var,loss,step=it2_show,split="eval_optim")
                if (it2+1)%10==0:
                    self.log_scalars(opt,var,loss,step=it2_show,split="eval_optim")
        return var,loss

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        if split=="train":
            dist_acc,dist_cov = eval_3D.compute_chamfer_dist(opt,var)
            metric = dict(dist_acc=dist_acc,dist_cov=dist_cov,rot_err=var.rot_err,trans_err=var.trans_err)
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        util_vis.tb_image(opt,self.tb,step,split,"image_input",var.rgb_input_map,masks=var.mask_input_map,from_range=(-1,1),poses=var.pose)
        util_vis.tb_image(opt,self.tb,step,split,"mask_input",var.mask_input_map)
        util_vis.tb_image(opt,self.tb,step,split,"level_input",var.dt_input_map,cmap="hot")
        if not (opt.impl.rand_sample and split=="train"):
            util_vis.tb_image(opt,self.tb,step,split,"image_recon",var.rgb_recon_map,masks=var.mask_map,from_range=(-1,1),poses=var.pose)
            util_vis.tb_image(opt,self.tb,step,split,"depth",1/(var.depth_map-opt.impl.init_depth+1))
            normal = self.compute_normal_from_depth(opt,var.depth_map,intr=var.intr)
            util_vis.tb_image(opt,self.tb,step,split,"normal",normal,from_range=(-1,1))
            util_vis.tb_image(opt,self.tb,step,split,"mask",var.mask_map)
            util_vis.tb_image(opt,self.tb,step,split,"depth_masked",1/(var.depth_map-opt.impl.init_depth+1)*var.mask_map)
            mask_normal = var.mask_map[...,1:-1,1:-1]
            util_vis.tb_image(opt,self.tb,step,split,"normal_masked",normal*mask_normal+(-1)*(1-mask_normal),from_range=(-1,1))
            util_vis.tb_image(opt,self.tb,step,split,"level",var.level_map,cmap="hot")
            if split=="eval":
                util_vis.tb_pointcloud(opt,self.tb,step,split,"pred_gt_points",pred=var.dpc_pred,GT=var.dpc.points)
                util_vis.tb_mesh(opt,self.tb,step,split,"pred_meshes",pred=var.dpc_pred,GT=var.dpc.points)
        # visualize point cloud
        if opt.eval and opt.visdom:
            with util.suppress(stdout=True,stderr=True): # suppress weird (though unharmful) visdom errors related to remote connections
                util_vis.vis_pointcloud(opt,self.vis,step,split,pred=var.dpc_pred,GT=var.dpc.points)

    @torch.no_grad()
    def dump_results(self,opt,var,write_new=False, train=False):
        if train == False:
            current_folder = 'dump'
            os.makedirs("{}/dump/".format(opt.output_path),exist_ok=True)
        else:
            current_folder = 'vis'
            os.makedirs("{}/vis/".format(opt.output_path),exist_ok=True)
        
        util_vis.dump_images(opt,var.idx,"image_input",var.rgb_input_map,masks=var.mask_input_map,from_range=(-1,1),folder=current_folder)
        util_vis.dump_images(opt,var.idx,"image_recon",var.rgb_recon_map,masks=var.mask_map,from_range=(-1,1),folder=current_folder)
        util_vis.dump_images(opt,var.idx,"depth",1/(var.depth_map-opt.impl.init_depth+1),folder=current_folder)
        normal = self.compute_normal_from_depth(opt,var.depth_map,intr=var.intr)
        util_vis.dump_images(opt,var.idx,"normal",normal,from_range=(-1,1),folder=current_folder)
        util_vis.dump_images(opt,var.idx,"mask",var.mask_map,folder=current_folder)
        util_vis.dump_images(opt,var.idx,"mask_input",var.mask_input_map,folder=current_folder)
        util_vis.dump_images(opt,var.idx,"depth_masked",1/(var.depth_map-opt.impl.init_depth+1)*var.mask_map,folder=current_folder)
        mask_normal = var.mask_map[...,1:-1,1:-1]
        util_vis.dump_images(opt,var.idx,"normal_masked",normal*mask_normal+(-1)*(1-mask_normal),from_range=(-1,1),folder=current_folder)
        util_vis.dump_meshes(opt,var.idx,"mesh",var.mesh_pred,folder=current_folder)
        # write/append to html for convenient visualization if not training time
        if train == False:
            html_fname = "{}/dump/vis.html".format(opt.output_path)
            with open(html_fname,"w" if write_new else "a") as html:
                for i in var.idx:
                    html.write("{} ".format(i))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"image_input"))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"image_recon"))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"depth"))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"normal"))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"mask"))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"mask_input"))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"depth_masked"))
                    html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i,"normal_masked"))
                    html.write("<br>\n")
            # write chamfer distance results
            chamfer_fname = "{}/chamfer.txt".format(opt.output_path)
            with open(chamfer_fname,"w" if write_new else "a") as file:
                for i,acc,comp in zip(var.idx,var.cd_acc,var.cd_comp):
                    file.write("{} {:.8f} {:.8f}\n".format(i,acc,comp))

    @torch.no_grad()
    def compute_normal_from_depth(self,opt,depth,intr=None):
        batch_size = len(depth)
        # use camera-centered frame instead of world frame
        pose = camera.pose(t=[0,0,0]).repeat(batch_size,1,1).to(opt.device)
        center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        depth = depth.view(batch_size,opt.H*opt.W,1)
        pts_cam = camera.get_3D_points_from_depth(opt,center,ray,depth) # [B,HW,3]
        pts_cam = pts_cam.view(batch_size,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
        # https://en.wikipedia.org/wiki/Normal_(geometry), the following bullet
        # Calculating a surface normal, If a (possibly non-flat) surface S ...
        # x, y on the 2D plane does not have any meaning here, they are just the parameter of the surface, 
        # r(x, y) = (x_hat(x,y), y_hat(x,y), z_hat(x,y))
        # then the partial dirivatives dr/dx, dr/dy just provides two different tangent lines of the surface at surface point parametrized by (x,y)
        # cross product leads to normal here.
        dy = pts_cam[...,2:,:]-pts_cam[...,:-2,:] # [B,3,H-2,W]
        dx = pts_cam[...,:,2:]-pts_cam[...,:,:-2] # [B,3,H,W-2]
        dy = torch_F.normalize(dy,dim=1)[...,:,1:-1] # [B,3,H-2,W-2]
        dx = torch_F.normalize(dx,dim=1)[...,1:-1,:] # [B,3,H-2,W-2]
        normal = dx.cross(dy,dim=1)
        return normal

# ============================ computation graph for forward/backprop ============================

class Graph(implicit.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        network = getattr(torchvision.models,opt.arch.enc_network)
        self.encoder = network(pretrained=opt.arch.enc_pretrained)
        self.encoder.fc = torch.nn.Linear(self.encoder.fc.in_features,opt.latent_dim)
        if opt.loss_weight.cat_reg is not None:
            self.centers_init = False
            self.centers_param = torch.nn.Parameter(torch.zeros(opt.data.shapenet.num_classes, self.generator.code_length), requires_grad=True)

    def forward(self,opt,var,training=False):
        batch_size = len(var.idx)
        var.latent_enc = var.latent if "latent" in var else self.encoder(var.rgb_input_map)
        # predict the viewpoint
        angle_y = self.estimator(var.rgb_input_map) * np.pi
        angle_x = torch.ones(angle_y.shape).to(angle_y.device) * -0.52359885 
        angle_z = torch.ones(angle_y.shape).to(angle_y.device) * 3.14159261
        angles = torch.cat([angle_x, angle_y, angle_z], dim=1)
        rotmat = camera.euler2mat(angles)
        translations = torch.tensor([0.,0.,opt.camera.dist]).to(angle_y.device).view(1,3,1).expand(batch_size,3,1)
        var.pose = torch.cat([rotmat, translations], dim=-1)
        assert var.pose.shape == var.pose_gt.shape
        # test the viewpoint estimation accuracy
        combined_rotation = torch.bmm(var.pose_gt[:, :, :-1], var.pose[:, :, :-1].transpose(1,2))
        trans_dist = (var.pose_gt[:, :, -1] - var.pose[:, :, -1]).abs()
        with torch.no_grad():
            geodesic_cos = (combined_rotation[:,0,0] + combined_rotation[:,1,1] + combined_rotation[:,2,2] - 1) / 2
            geodesic_cos.clamp_(-1,1)
            var.rot_err = torch.mean(torch.acos(geodesic_cos)/np.pi*180)
            var.trans_err = torch.mean(trans_dist)
        var.impl_func = self.generator.forward(opt,var.latent_enc)
        if opt.impl.rand_sample and training:
            # sample random rays for optimization
            var.rgb_recon,var.depth,var.level,var.mask,var.level_all,var.last_point = self.renderer.forward(opt,var.impl_func,var.pose,intr=var.intr,ray_idx=var.ray_idx) # [B,HW,3]
        else:
            # [B,HW,3], [B,HW,1], [B,HW,1], [B,HW,1] (mask generated by SDF of last step, not bisection; if SDF>0 then mask is 1), [B,HW,steps+1]
            var.rgb_recon,var.depth,var.level,var.mask,var.level_all,var.last_point = self.renderer.forward(opt,var.impl_func,var.pose,intr=var.intr) # [B,HW,3]
            # _map is just viewed as a map
            var.rgb_recon_map = var.rgb_recon.view(batch_size,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            var.depth_map = var.depth.view(batch_size,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            var.level_map = var.level.view(batch_size,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            var.mask_map = var.mask.view(batch_size,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
        var.foreground_pts = var.last_point[var.mask_input.squeeze().bool()]
        return var

    def compute_loss(self,opt,var,training=False):
        loss = edict()
        batch_size = len(var.idx)
        # main losses
        # render loss is the RGB L1 reconstruct loss
        # shape_silh loss is the proposed loss
        # ray_intsc loss operates on the ***last steps*** based on the 2D mask (ideally inside the surface if in foreground)
        #   OCC: fit occ to 1 if mask is 1 otherwise 0 -> fit occ to mask
        #   SDF: push SDF < eps if mask is 1, otherwise SDF > eps
        # ray_free loss operates on all steps before the last step
        #   OCC: occupancy -> 0
        #   SDF: SDF > eps
        # eikonal loss make ||grad_SDF|| -> 1
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb_recon,var.rgb_input)
        if opt.loss_weight.shape_silh is not None:
            loss.shape_silh = self.shape_from_silhouette_loss(opt,var)
        # regularizations
        if opt.loss_weight.ray_intsc is not None:
            loss.ray_intsc = self.ray_intersection_loss(opt,var)
        if opt.loss_weight.ray_free is not None:
            loss.ray_free = self.ray_freespace_loss(opt,var)
        if opt.loss_weight.cube is not None:
            loss.cube = self.unit_cube_loss(opt,var)
        if opt.loss_weight.eikonal is not None:
            var.sdf_grad_norm = self.sdf_gradient_norm(opt,var.impl_func,batch_size=len(var.idx))
            loss.eikonal = self.MSE_loss(var.sdf_grad_norm,1)
        if opt.loss_weight.cat_reg is not None:
            loss.cat_reg = self.category_reg_loss(opt,var,training)
        return loss

    def unit_cube_loss(self,opt,var,margin=0.6):
        loss = self.L1_loss((var.foreground_pts-margin).relu_())
        loss += self.L1_loss((-var.foreground_pts-margin).relu_())
        return loss

    def center_dist(self, opt, var, shape_code_list, training):
        # normalize the length of each vector so the regularization won't take the 
        # shortcut of shrinking the whole space for l1 or l2 regularization
        shape_code = torch.cat(shape_code_list, dim=-1)
        loss = torch.tensor(0.).to(shape_code.device)
        num_classes = opt.data.shapenet.num_classes
        shape_code = torch.nn.functional.normalize(shape_code, dim=-1, p=2)
        assert len(shape_code.shape) == 2
        batch_size = shape_code.shape[0]
        category = var.category_label
        
        # initialize the centers if needed
        if training and (not self.centers_init):
            avg_codes = []
            log.info("initializing the centers...")
            for i in range(num_classes):
                avg_code = torch.mean(shape_code[category==i], dim=0, keepdim=True)
                avg_codes.append(avg_code)
            avg_codes = torch.cat(avg_codes, dim=0)
            self.centers_offset = avg_codes.detach()
            self.centers_init = True
        
        # calculate the distance between each datapoint and the centers  
        if self.centers_init:
            centers = self.centers_param + self.centers_offset
            for i in range(num_classes):
                if opt.loss_weight.reg_type[0] == 'l':
                    power = int(opt.loss_weight.reg_type[1:])
                    diff_vec = shape_code[category==i] - centers[i].unsqueeze(0)
                    diff_vec = diff_vec.abs()
                    diff_vec = diff_vec**power
                    loss += diff_vec.sum()
                elif opt.loss_weight.reg_type == 'cos':
                    avg_code_n = torch.nn.functional.normalize(centers[i].unsqueeze(0), dim=1, p=2)
                    assert len(avg_code_n.shape) == 2
                    loss += (1 - (shape_code[category==i] * avg_code_n).sum(dim=1)).sum()
                else:
                    raise NotImplementedError()
            loss = loss / batch_size
        return loss
    
    def category_reg_loss(self, opt, var, training):
        shape_code_list = self.generator.impl_weights + self.generator.level_weights
        loss = self.center_dist(opt, var, shape_code_list, training)
        return loss
    
    def ray_intersection_loss(self,opt,var,level_eps=0.01):
        batch_size = len(var.idx)
        level_in = var.level_all[...,-1:] # [B,HW,1]
        weight = 1/(var.dt_input+1e-8) if opt.impl.importance else None
        if opt.impl.occup:
            loss = self.BCE_loss(level_in,var.mask_input,weight=weight)
        else:
            loss = self.L1_loss((level_in+level_eps).relu_(),weight=weight,mask=var.mask_input.bool()) \
                  +self.L1_loss((-level_in+level_eps).relu_(),weight=weight,mask=~var.mask_input.bool())
        return loss

    def ray_freespace_loss(self,opt,var,level_eps=0.01):
        level_out = var.level_all[...,:-1] # [B,HW,N-1]
        if opt.impl.occup:
            loss = self.BCE_loss(level_out,torch.tensor(0.,device=opt.device))
        else:
            loss = self.L1_loss((-level_out+level_eps).relu_())
        return loss

    def shape_from_silhouette_loss(self,opt,var): # [B,N,H,W]
        batch_size = len(var.idx)
        mask_bg = var.mask_input.long()==0
        weight = 1/(var.dt_input+1e-8) if opt.impl.importance else None
        # randomly sample depth along ray
        depth_min,depth_max = opt.impl.depth_range
        num_rays = var.ray_idx.shape[1] if "ray_idx" in var else opt.H*opt.W
        # [B,HW,N,1]
        depth_samples = torch.rand(batch_size,num_rays,opt.impl.sdf_samples,1,device=opt.device)*(depth_max-depth_min)+depth_min 
        center,ray = camera.get_center_and_ray(opt,var.pose,intr=var.intr)
        if "ray_idx" in var:
            # var.ray_idx: [B, 1024]
            # gather_idx: [B, 1024, 3]
            gather_idx = var.ray_idx[...,None].repeat(1,1,3)
            # ray: [B, HW, 3]
            ray = ray.gather(dim=1,index=gather_idx)
            # ray: [B, 1024, 3]
            if opt.camera.model=="orthographic":
                center = center.gather(dim=1,index=gather_idx)
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        level_samples = var.impl_func.forward(opt,points_3D_samples)[...,0] # [B,HW,N]
        if opt.impl.occup:
            # if using occupancy representation, for all rays, sample 5 depths and convert to 3D
            # if the mask shows this ray is background, then all 5 points should have occ of 0
            # don't consider foreground
            loss = self.BCE_loss(level_samples,var.mask_input,weight=weight,mask=mask_bg)
        else:
            # compute lower bound following eq 1 when using SDF representation
            if opt.camera.model=="perspective":
                # grid_3D is u
                _,grid_3D = camera.get_camera_grid(opt,batch_size,intr=var.intr) # [B,HW,3]
                # u / ||u|| * D(u)
                offset = torch_F.normalize(grid_3D[...,:2],dim=-1)*var.dt_input_map.view(batch_size,-1,1) # [B,HW,2]
                # v = u + u / ||u|| * D(u)
                _,ray0 = camera.get_center_and_ray(opt,var.pose,intr=var.intr,offset=offset) # [B,HW,3]
                if "ray_idx" in var:
                    gather_idx = var.ray_idx[...,None].repeat(1,1,3)
                    ray0 = ray0.gather(dim=1,index=gather_idx)
                # eq 1
                ortho_dist = (ray-ray0*(ray*ray0).sum(dim=-1,keepdim=True)/(ray0*ray0).sum(dim=-1,keepdim=True)).norm(dim=-1,keepdim=True) # [B,HW,1]
                min_dist = depth_samples[...,0]*ortho_dist # [B,HW,N]
            elif opt.camera.model=="orthographic":
                min_dist = var.dt_input
            loss = self.L1_loss((min_dist-level_samples).relu_(),weight=weight,mask=mask_bg)
        return loss

    def sdf_gradient_norm(self,opt,impl_func,batch_size,N=10000):
        lower,upper = opt.impl.sdf_range
        points_3D = torch.rand(batch_size,N,3,device=opt.device)
        points_3D = points_3D*(upper-lower)+lower
        with torch.enable_grad():
            points_3D.requires_grad_(True)
            level = impl_func.forward(opt,points_3D) # [B,HW,1]
            grad = torch.autograd.grad(level.sum(),points_3D,create_graph=True)
        grad_norm = grad[0].norm(dim=-1,keepdim=True)
        return grad_norm
