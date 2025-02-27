import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pickle

import cv2
import spconv.pytorch as spconv
from pytorch3d.ops.knn import knn_points
from torchvision.models import resnet18
from models.ray_marcher import MipRayMarcher2
# from models.superresolution import SuperresolutionHybrid1X

# import functools

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = torch.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2 + arr[..., 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps 
    arr[..., 0] /= lens
    arr[..., 1] /= lens
    arr[..., 2] /= lens
    return arr 

def compute_normal(vertices, faces):
    # norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    norm = torch.zeros(vertices.shape, dtype=vertices.dtype).cuda()
    tris = vertices[:, faces] # [bs, 13776, 3, 3]
    # n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n = torch.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    vec = torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]) 
    vec = normalize_v3(vec)
    norm[:, faces[:, 0]] += vec
    norm[:, faces[:, 1]] += vec
    norm[:, faces[:, 2]] += vec
    norm = normalize_v3(norm)

    return norm

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs = joints.shape[0]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints.shape[1], 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, joints.shape[1], 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

# @profile
def get_transform_params_torch(params, smpl, v_template, shapedirs, J_regressor):
    """ obtain the transformation parameters for linear blend skinning
    """
    # device = params['shapes'].device

    # v_template = smpl['v_template'] #.to(device)

    # add shape blend shapes
    # shapedirs = smpl['shapedirs'] #.to(device)
    betas = params['shapes']
    # v_shaped = v_template + torch.sum(shapedirs * betas[None], axis=2).float()

    # v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()
    v_shaped = v_template + torch.sum(shapedirs[..., :betas.shape[-1]] * betas[:,None], axis=-1).float()

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # bs x 24 x 3 x 3
    rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

    # obtain the joints
    joints = torch.matmul(J_regressor, v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['neutral']['kintree_table'][0] #.to(device)

    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] #.to(device)
    Th = params['Th'] #.to(device)

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def to_cuda(data_dict, device, dtype=torch.float32):
    for key in data_dict.keys():
        if torch.is_tensor(data_dict[key]):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.from_numpy(data_dict[key])[None].to(device, dtype=dtype)
        if key=='tgt_smpl_param':
            for key1 in data_dict['tgt_smpl_param']:
                if torch.is_tensor(data_dict['tgt_smpl_param'][key1]):
                    data_dict['tgt_smpl_param'][key1] = data_dict['tgt_smpl_param'][key1].to(device)
                elif isinstance(data_dict['tgt_smpl_param'][key1], np.ndarray):
                    data_dict['tgt_smpl_param'][key1] = torch.from_numpy(data_dict['tgt_smpl_param'][key1])[None].to(device, dtype=dtype)
    
        if key=='big_pose_smpl_param':
            for key1 in data_dict['big_pose_smpl_param']:
                if torch.is_tensor(data_dict['big_pose_smpl_param'][key1]):
                    data_dict['big_pose_smpl_param'][key1] = data_dict['big_pose_smpl_param'][key1].to(device)
                elif isinstance(data_dict['big_pose_smpl_param'][key1], np.ndarray):
                    data_dict['big_pose_smpl_param'][key1] = torch.from_numpy(data_dict['big_pose_smpl_param'][key1])[None].to(device, dtype=dtype)

        if key=='ref_smpl_param':
            for key1 in data_dict['ref_smpl_param']:
                if torch.is_tensor(data_dict['ref_smpl_param'][key1]):
                    data_dict['ref_smpl_param'][key1] = data_dict['ref_smpl_param'][key1].to(device)
                elif isinstance(data_dict['ref_smpl_param'][key1], np.ndarray):
                    data_dict['ref_smpl_param'][key1] = torch.from_numpy(data_dict['ref_smpl_param'][key1])[None].to(device, dtype=dtype)

    return data_dict

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

class NeRF_Renderer(nn.Module):
    def __init__(self, use_2d_feature=True, use_3d_feature=True, depth_resolution=48, use_smpl_dist_mask=True, nerf_cond_type=None, smpl_type='smplx', white_bg=False):
        super(NeRF_Renderer, self).__init__()
        self.encoder_2d = ResNet18Classifier()
        self.encoder_3d = SparseConvNet(num_layers=4)
        self.rgb_enc = PositionalEncoding(num_freqs=5)
        self.nerf_cond_type = nerf_cond_type
        if self.nerf_cond_type == '480_480_upscale':
            self.conv1d_projection = nn.Conv1d(67, 32, 1)
            self.feature_projection = nn.Conv1d(259, 64, 1)
            self.decoder = OSGDecoder(64, {'decoder_output_dim': 3})
        elif self.nerf_cond_type == '512_512':
            self.conv1d_projection = nn.Conv1d(67, 32, 1)
            self.feature_projection = nn.Conv1d(259, 64, 1)
            self.decoder = OSGDecoder(64, {'decoder_output_dim': 6})     
        elif self.nerf_cond_type == '512_512_3':
            self.conv1d_projection = nn.Conv1d(67, 32, 1)
            self.feature_projection = nn.Conv1d(259, 64, 1)
            self.decoder = OSGDecoder(64, {'decoder_output_dim': 3})        
        else:
            self.conv1d_projection = nn.Conv1d(96, 32, 1) # nn.Conv1d(67, 32, 1) #nn.Conv1d(96, 32, 1)
            self.feature_projection = nn.Conv1d(288, 32, 1) # nn.Conv1d(288, 64, 1) # nn.Conv1d(259, 64, 1)
            self.decoder = NeRFDecoder(32) #NeRFDecoder(64)
        self.ray_marcher = MipRayMarcher2()
        self.use_smpl_dist_mask = use_smpl_dist_mask
        self.depth_resolution = depth_resolution
        self.white_bg = white_bg
        # self.use_2d_feature = use_2d_feature
        # self.use_3d_feature = use_3d_feature

        # load SMPL model
        self.SMPL = {}
        for gender in ['female', 'male', 'neutral']:
            if smpl_type == 'smplx':
                if gender == 'female':
                    smpl_path = os.path.join('assets/models/smplx', 'SMPLX_FEMALE.npz')
                elif gender == 'male':
                    smpl_path = os.path.join('assets/models/smplx', 'SMPLX_MALE.npz')
                else:
                    smpl_path = os.path.join('assets/models/smplx', 'SMPLX_NEUTRAL.npz')
                params_init = dict(np.load(smpl_path, allow_pickle=True))
                self.SMPL[gender] = SMPL_to_tensor(params_init, device=torch.device('cuda', torch.cuda.current_device()))
            else:
                if gender == 'female':
                    smpl_path = os.path.join('assets', 'SMPL_FEMALE.pkl')
                elif gender == 'male':
                    smpl_path = os.path.join('assets', 'SMPL_MALE.pkl')  
                else:
                    smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')              
                self.SMPL[gender] = SMPL_to_tensor(read_pickle(smpl_path), device=torch.device('cuda', torch.cuda.current_device()))


    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)

        return depths_coarse

    def writeOBJ(self, file, V, F, Vt=None, Ft=None):    
        if not Vt is None:        
            assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'            
        with open(file, 'w') as file:        
            # Vertices        
            for v in V:            
                line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'            
                file.write(line)        
            # UV verts        
            if not Vt is None:            
                for v in Vt:                
                    line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'                
                    file.write(line)        
                    # 3D Faces / UV faces        
                    if Ft:            
                        F = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(F,Ft)]        
                    else:            
                        F = [[str(i + 1) for i in f] for f in F]                
            for f in F:            
                # line = 'f ' + ' '.join(f) + '\n'  
                line = 'f ' + str(f[0]) + ' ' + str(f[1]) + ' ' + str(f[2]) + ' ' + '\n'            
                file.write(line)

    def forward(self, tgt_ray_o, tgt_ray_d, tgt_near, tgt_far, tgt_mask_at_box, tgt_img_nerf, tgt_smpl_param, tgt_world_vertex, ref_img, ref_smpl_param, ref_world_vertex, ref_K, ref_R, ref_T, big_pose_smpl_param, big_pose_world_vertex, big_pose_world_bound, gender):

        # self.writeOBJ('ref_space.obj', ref_world_vertex.cpu().numpy()[0], self.SMPL[gender[0]]['f'].cpu().numpy() + 1)

        x_lst, y_lst, w_lst, h_lst = [], [], [], []
        tgt_mask_at_box = tgt_mask_at_box.to(torch.bool)
        tgt_mask_at_box = tgt_mask_at_box.reshape(tgt_img_nerf.shape[0], tgt_img_nerf.shape[2], tgt_img_nerf.shape[3])
        bs = tgt_mask_at_box.shape[0]
        ### debug ###
        if self.nerf_cond_type == '480_480_upscale':
            for i in range(bs):
                # crop the object region
                x, y, w, h = cv2.boundingRect(tgt_mask_at_box[i].cpu().numpy().astype(np.uint8))
                x_lst.append(x)
                y_lst.append(y)
                w_lst.append(w)
                h_lst.append(h)
            x, y, w, h = min(x_lst), min(y_lst), max(w_lst), max(h_lst)     
        elif self.nerf_cond_type == '512_512_3':
            for i in range(bs):
                # crop the object region
                x, y, w, h = 0, (tgt_img_nerf.shape[-2] - tgt_img_nerf.shape[-1])//2, 512, 512
                x_lst.append(x)
                y_lst.append(y)
                w_lst.append(w)
                h_lst.append(h)
            x, y, w, h = min(x_lst), min(y_lst), max(w_lst), max(h_lst)                 
        ###
        # import pdb; pdb.set_trace()
        # tgt_img_nerf_ = tgt_img_nerf[4][:, y:y + h, x:x + w]
        # img = Image.fromarray((255*tgt_img_nerf_.permute(1,2,0).detach().cpu().numpy()).astype(np.uint8))
        # img.save('crop_img.png')


        # extract image feature
        ref_img_feature = self.encoder_2d(ref_img, extract_feature=True)
        ### debug ###
        if self.nerf_cond_type == '480_480_upscale' or self.nerf_cond_type == '512_512_3':
            tgt_ray_o = tgt_ray_o.reshape(tgt_img_nerf.shape[0], tgt_img_nerf.shape[2], tgt_img_nerf.shape[3], tgt_ray_o.shape[-1])[:, y:y + h, x:x + w].reshape(tgt_img_nerf.shape[0], -1, tgt_ray_o.shape[-1])
            tgt_ray_d = tgt_ray_d.reshape(tgt_img_nerf.shape[0], tgt_img_nerf.shape[2], tgt_img_nerf.shape[3], tgt_ray_d.shape[-1])[:, y:y + h, x:x + w].reshape(tgt_img_nerf.shape[0], -1, tgt_ray_d.shape[-1])
            tgt_near = tgt_near.reshape(tgt_img_nerf.shape[0], tgt_img_nerf.shape[2], tgt_img_nerf.shape[3])[:, y:y + h, x:x + w].reshape(tgt_img_nerf.shape[0], -1)
            tgt_far = tgt_far.reshape(tgt_img_nerf.shape[0], tgt_img_nerf.shape[2], tgt_img_nerf.shape[3])[:, y:y + h, x:x + w].reshape(tgt_img_nerf.shape[0], -1)
        ###

        # sample points
        depths_coarse = self.sample_stratified(tgt_ray_o, tgt_near[:,:,None], tgt_far[:,:,None], self.depth_resolution)
        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (tgt_ray_o.unsqueeze(-2) + depths_coarse * tgt_ray_d.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = tgt_ray_d.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        tgt_smpl_R, tgt_smpl_Th = tgt_smpl_param['R'], tgt_smpl_param['Th'] # [bs, 3, 3] [bs, 1, 3]
        smpl_query_pts = torch.matmul(sample_coordinates - tgt_smpl_Th, tgt_smpl_R).float() # [bs, N_rays*N_samples, 3]
        smpl_query_viewdir = torch.matmul(sample_directions, tgt_smpl_R)

        ### debug ###
        # smpl_query_pts = sample_coordinates - tgt_smpl_Th
        # smpl_query_viewdir = sample_directions
        ###

        # discard points away from SMPL surface
        if self.use_smpl_dist_mask:
            tar_smpl_pts = torch.matmul(tgt_world_vertex - tgt_smpl_Th, tgt_smpl_R).float() # [bs, 6890, 3]
            ### debug ###
            # tar_smpl_pts = tgt_world_vertex - tgt_smpl_Th
            ###
            distance, _, _ = knn_points(smpl_query_pts, tar_smpl_pts, K=1)
            pts_mask = torch.zeros_like(smpl_query_pts[..., 0]).int()
            if self.nerf_cond_type == '512_512_3':
                threshold = 0.05 ** 2
            else:
                threshold = 0.05 ** 2
            pts_mask[distance[..., 0] < threshold] = 1
            pts_mask[(pts_mask.sum(0)>0)[None].expand(pts_mask.shape[0], -1)] = 1
            smpl_query_pts = smpl_query_pts[pts_mask==1].reshape(pts_mask.shape[0], -1, 3) 
            smpl_query_viewdir = smpl_query_viewdir[pts_mask==1].reshape(pts_mask.shape[0], -1, 3) 

        # deform points from target space to canonical space
        coarse_canonical_pts, coarse_canonical_viewdir, v_template, posedirs, shapedirs, J_regressor, bweights, vert_ids = self.coarse_deform_target2c(tgt_smpl_param, tgt_world_vertex, big_pose_smpl_param, smpl_query_pts, query_viewdirs=smpl_query_viewdir, gender=gender)
        # coarse_canonical_pts, v_template, posedirs, shapedirs, J_regressor, bweights, vert_ids = self.coarse_deform_target2c(tgt_smpl_param, tgt_world_vertex, big_pose_smpl_param, smpl_query_pts, gender=gender)

        ### 2d image aligned feature ###
        bs = coarse_canonical_pts.shape[0]
        _, world_src_pts = self.coarse_deform_c2source(ref_smpl_param, big_pose_smpl_param, big_pose_world_vertex, coarse_canonical_pts, v_template, posedirs, shapedirs, J_regressor, bweights, vert_ids)
        # import pdb; pdb.set_trace()
        # from plyfile import PlyData, PlyElement
        # pts = world_src_pts[0]
        # x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        # pts = list(zip(x, y, z))
        # vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        # el = PlyElement.describe(vertex, 'vertex')
        # PlyData([el], text='target_pts2').write('target_pts2.ply')

        src_uv = self.projection(world_src_pts.reshape(bs, -1, 3), ref_R, ref_T, ref_K) # [bs, N, 6890, 3]
        src_uv_ = 2.0 * src_uv.unsqueeze(2) / torch.Tensor([ref_img.shape[-1], ref_img.shape[-2]]).to(ref_img.device) - 1.0
        point_pixel_feature = F.grid_sample(ref_img_feature, src_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]

        # extract pixel aligned rgb feature
        point_pixel_rgb = F.grid_sample(ref_img, src_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]
        
        sh = point_pixel_rgb.shape
        if self.nerf_cond_type != '480_480_upscale' and '512_512' not in self.nerf_cond_type:
            point_pixel_rgb = self.rgb_enc(point_pixel_rgb.reshape(-1,3)).reshape(*sh[:2], 33)[..., :32] # [bs, N_rays*N_samples, 32] 

        point_2d_feature = torch.cat((point_pixel_feature, point_pixel_rgb), dim=-1) # [bs, N_rays*N_samples, 96] 

        # import pdb; pdb.set_trace()
        # src_uv, ref_smpl_vertex_mask = self.projection(ref_world_vertex.reshape(bs, -1, 3), ref_R, ref_T, ref_K, self.SMPL['neutral']['f']) # [bs, N, 6890, 3]
        # test_image = torch.ones_like(ref_img[0].permute(1, 2, 0).clone())
        # src_uv[ref_smpl_vertex_mask==0] = 0
        # src_uv[:, range(0, 10475, 3)]=0
        # src_uv[:, range(1, 10475, 3)]=0
        # src_uv[0,:,1][src_uv[0,:,1]<0] = 0
        # src_uv[0,:,1][src_uv[0,:,1]>=test_image.shape[0]] = 0
        # src_uv[0,:,0][src_uv[0,:,0]<0] = 0
        # src_uv[0,:,0][src_uv[0,:,0]>=test_image.shape[1]] = 0
        # color_array = torch.Tensor([63, 76, 156]).cuda() / 255
        # test_image[src_uv[0,:,1].type(torch.LongTensor), src_uv[0,:,0].type(torch.LongTensor)] = color_array #1 #point_pixel_rgb[0, :, :3]
        # test_image[src_uv[0,:,1].type(torch.LongTensor)+1, src_uv[0,:,0].type(torch.LongTensor)] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)+1, src_uv[0,:,0].type(torch.LongTensor)+1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)+1, src_uv[0,:,0].type(torch.LongTensor)-1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)-1, src_uv[0,:,0].type(torch.LongTensor)] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)-1, src_uv[0,:,0].type(torch.LongTensor)+1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)-1, src_uv[0,:,0].type(torch.LongTensor)-1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor), src_uv[0,:,0].type(torch.LongTensor)+1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)+1, src_uv[0,:,0].type(torch.LongTensor)+1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)-1, src_uv[0,:,0].type(torch.LongTensor)+1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor), src_uv[0,:,0].type(torch.LongTensor)-1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)+1, src_uv[0,:,0].type(torch.LongTensor)-1] = color_array
        # test_image[src_uv[0,:,1].type(torch.LongTensor)-1, src_uv[0,:,0].type(torch.LongTensor)-1] = color_array
        # import imageio
        # imageio.imwrite('test.png', (255 * test_image).cpu().numpy().astype(np.uint8))
        # imageio.imwrite('test_.png', (255 * ref_img[0].permute(1, 2, 0)).cpu().numpy().astype(np.uint8))

        ### 3d voxel feature ###
        # get vertex feature to form sparse convolution tensor
        ref_uv, ref_smpl_vertex_mask = self.projection(ref_world_vertex.reshape(bs, -1, 3), ref_R, ref_T, ref_K, self.SMPL['neutral']['f']) # [bs, N, 6890, 3]
        ref_uv_ = 2.0 * ref_uv.unsqueeze(2) / torch.Tensor([ref_img.shape[-1], ref_img.shape[-2]]).to(ref_img.device) - 1.0
        obs_vertex_feature = F.grid_sample(ref_img_feature, ref_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]

        obs_vertex_rgb = F.grid_sample(ref_img, ref_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]

        sh = obs_vertex_rgb.shape
        if self.nerf_cond_type != '480_480_upscale' and '512_512' not in self.nerf_cond_type:
            obs_vertex_rgb = self.rgb_enc(obs_vertex_rgb.reshape(-1,3)).reshape(*sh[:2], 33)[..., :32] # [bs, N_rays*N_samples, 32] 
        obs_vertex_3d_feature = torch.cat((obs_vertex_feature, obs_vertex_rgb), dim=-1) # [bs, N_rays*N_samples, 96] 

        obs_vertex_3d_feature = self.conv1d_projection(obs_vertex_3d_feature.permute(0,2,1)).permute(0,2,1)

        obs_vertex_3d_feature[ref_smpl_vertex_mask==0] = 0

        ## vertex points in SMPL coordinates
        smpl_obs_pts = torch.matmul(ref_world_vertex.reshape(bs, -1, 3) - ref_smpl_param['Th'], ref_smpl_param['R'])

        ## coarse deform target to caonical
        coarse_obs_vertex_canonical_pts, _, _, _, _, _, _ = self.coarse_deform_target2c(ref_smpl_param, ref_world_vertex, big_pose_smpl_param, smpl_obs_pts, gender=gender) # [bs, N_rays*N_rand, 3]       

        # prepare sp input
        obs_sp_input, _ = self.prepare_sp_input(big_pose_world_vertex, coarse_obs_vertex_canonical_pts)

        canonical_sp_conv_volume = spconv.core.SparseConvTensor(obs_vertex_3d_feature.reshape(-1, obs_vertex_3d_feature.shape[-1]), obs_sp_input['coord'], obs_sp_input['out_sh'], obs_sp_input['batch_size']) # [bs, 32, 96, 320, 384] z, y, x

        grid_coords = self.get_grid_coords(coarse_canonical_pts, obs_sp_input)
        grid_coords = grid_coords[:, None, None]
        point_3d_feature = self.encoder_3d(canonical_sp_conv_volume, grid_coords) # torch.Size([b, 390, 1024*64])

        ### concatenate features
        feature_nerf = torch.cat([point_2d_feature, point_3d_feature], dim=-1)
        feature_nerf = self.feature_projection(feature_nerf.permute(0, 2, 1)).permute(0, 2, 1)
        out = {}
        chunk = 700000
        for i in range(0, feature_nerf.shape[1], chunk):
            if self.nerf_cond_type == '480_480_upscale' or '512_512' in self.nerf_cond_type:
                out_part = self.decoder(feature_nerf[:, i:i+chunk])
            else:
                # feature_nerf = self.feature_projection(feature_nerf[:, i:i+chunk].permute(0, 2, 1)).permute(0, 2, 1)
                out_part = self.decoder(coarse_canonical_pts[:, i:i+chunk], coarse_canonical_viewdir[:, i:i+chunk], feature_nerf[:, i:i+chunk])
            for k in out_part.keys():
                if k not in out.keys():
                    out[k] = []
                out[k].append(out_part[k]) 
        out = {k : torch.cat(out[k], 1) for k in out.keys()}

        # out = self.decoder(feature_nerf)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']

        if self.use_smpl_dist_mask:
            colors_coarse_w_mask = torch.zeros((batch_size, num_rays * samples_per_ray, colors_coarse.shape[-1])).to(colors_coarse.device, dtype=colors_coarse.dtype)
            colors_coarse_w_mask[pts_mask==1] = colors_coarse.reshape(-1, colors_coarse.shape[-1])
            densities_coarse_w_mask = torch.zeros((batch_size, num_rays * samples_per_ray, densities_coarse.shape[-1])).to(colors_coarse.device, dtype=colors_coarse.dtype)
            densities_coarse_w_mask[pts_mask==1] = densities_coarse.reshape(-1, densities_coarse.shape[-1])
            densities_coarse_w_mask[pts_mask==0] = -80
        else:
            colors_coarse_w_mask, densities_coarse_w_mask = colors_coarse, densities_coarse

        colors_coarse = colors_coarse_w_mask.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse_w_mask.reshape(batch_size, num_rays, samples_per_ray, 1)

        rendering_options={}
        rendering_options['white_bg'] = self.white_bg
        rendering_options['clamp_mode'] = 'softplus' if self.nerf_cond_type != '480_480_upscale' and '512_512' not in self.nerf_cond_type else 'relu'
        rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, tgt_ray_d, rendering_options)
        weights_final = weights.sum(2)
        if self.nerf_cond_type != '480_480_upscale' and self.nerf_cond_type != '512_512_3':
            rgb_img, depth_img, weights_img = rgb_final.reshape(bs, *tgt_img_nerf.shape[-2:], colors_coarse.shape[-1]), depth_final.reshape(bs, *tgt_img_nerf.shape[-2:], 1), weights_final.reshape(bs, *tgt_img_nerf.shape[-2:], 1)
        ### debug ###
        if self.nerf_cond_type == '480_480_upscale' or self.nerf_cond_type == '512_512_3':
            rgb_img = torch.zeros((bs, *tgt_img_nerf.shape[-2:], colors_coarse.shape[-1]), device=colors_coarse.device) if not self.white_bg else torch.ones((bs, *tgt_img_nerf.shape[-2:], colors_coarse.shape[-1]), device=colors_coarse.device)
            rgb_img[:, y:y + h, x:x + w] = rgb_final.reshape(bs, h, w, colors_coarse.shape[-1])
            depth_img = torch.zeros((bs, *tgt_img_nerf.shape[-2:], 1), device=colors_coarse.device) if not self.white_bg else torch.ones((bs, *tgt_img_nerf.shape[-2:], 1), device=colors_coarse.device)
            depth_img[:, y:y + h, x:x + w] = depth_final.reshape(bs, h, w, 1)
            weights_img = torch.zeros((bs, *tgt_img_nerf.shape[-2:], 1), device=colors_coarse.device) if not self.white_bg else torch.ones((bs, *tgt_img_nerf.shape[-2:], 1), device=colors_coarse.device)
            weights_img[:, y:y + h, x:x + w] = weights_final.reshape(bs, h, w, 1)
        ###

        # return rgb_final, depth_final, weights.sum(2)
        return rgb_img.permute(0, 3, 1, 2), depth_img.permute(0, 3, 1, 2), weights_img.permute(0, 3, 1, 2)

    def coarse_deform_target2c(self, params, vertices, t_params, query_pts, query_viewdirs=None, gender=None):

        bs = query_pts.shape[0]
        v_template, posedirs, shapedirs, J_regressor, bweights_vertex = [], [], [], [], []
        for key in gender:
            v_template.append(self.SMPL[key]['v_template'])
            posedirs.append(self.SMPL[key]['posedirs'])
            shapedirs.append(self.SMPL[key]['shapedirs'])
            J_regressor.append(self.SMPL[key]['J_regressor'])
            bweights_vertex.append(self.SMPL[key]['weights'])
        v_template = torch.stack(v_template)
        posedirs = torch.stack(posedirs)
        shapedirs = torch.stack(shapedirs)
        J_regressor = torch.stack(J_regressor)
        bweights_vertex = torch.stack(bweights_vertex)

        # joints transformation
        A, R, Th, joints = get_transform_params_torch(params, self.SMPL, v_template, shapedirs, J_regressor)
        # J = torch.hstack((joints, torch.ones([*joints.shape[:2], 1])))
        # J = torch.cat([joints, torch.ones([*joints.shape[:2], 1])], dim=-1)
        # self.c_joints = joints # smpl space
        # self.t_A = A # target to canonical space transformation
        # self.t_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        joints_num = joints.shape[1]
        vertices_num = vertices.shape[1]

        # transform smpl vertices from world space to smpl space
        # smpl_pts = torch.mm((vertices - Th), R)
        smpl_pts = torch.matmul((vertices - Th), R)
        # _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        _, vert_ids, _ = knn_points(query_pts.float(), smpl_pts.float(), K=1)
        # bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()
        # bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)

        bweights = torch.gather(bweights_vertex, 1, vert_ids.expand(-1, -1, bweights_vertex.shape[-1]))
        # bweights = bweights_vertex[vert_ids].view(*vert_ids.shape[:2], joints_num)

        # From smpl space target pose to smpl space canonical pose
        # A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        # can_pts = torch.sum(R_inv * can_pts[:, :, None], dim=-1)
        can_pts = torch.matmul(R_inv, can_pts[..., None]).squeeze(-1)

        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(R_inv, query_viewdirs[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:
            pose_ = params['poses']
            ident = torch.eye(3).to(pose_.device).float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.reshape(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])

            # pose_offset = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(batch_size, vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offset = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(batch_size, vertices_num*3, -1).permute(0,2,1)).view(batch_size, -1, 3)

            pose_offset = torch.gather(pose_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - pose_offset
            # can_pts = can_pts - pose_offsets[vert_ids.squeeze(0).reshape(-1)]

            # To mean shape
            # shapedirs = self.SMPL_NEUTRAL['shapedirs']  #.to(pose_.device)
            # shape_offset = shapedirs.dot(torch.reshape(params['shapes'], (10,)))
            # shape_offset = torch.einsum('ijk, k -> ij', shapedirs, torch.reshape(params['shapes'], (10,)))
            # shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)

            shape_offset = torch.matmul(shapedirs[..., :params['shapes'].shape[-1]], torch.reshape(params['shapes'], (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - shape_offset
            # can_pts = can_pts - shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # From T pose To Big Pose        
        big_pose_params = t_params
        # if self.mean_shape:
        #     big_pose_params['shapes'] = torch.zeros_like(params['shapes'])

        # smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')
        # vertex, joint = smpl_model(np.array(params['poses'][0].squeeze().cpu()), np.array(params['shapes'][0].squeeze().cpu()))
        # vertex, joint = smpl_model(np.array(big_pose_params['poses'][0].squeeze().cpu()), np.array(big_pose_params['shapes'][0].squeeze().cpu()))

        if self.mean_shape:
            pose_ = big_pose_params['poses']
            rot_mats = batch_rodrigues(pose_.reshape(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offset = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(batch_size, vertices_num*3, -1).permute(0, 2, 1)).view(batch_size, -1, 3)
            pose_offset = torch.gather(pose_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts + pose_offset

            # To mean shape
            # shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(big_pose_params['shapes'].cuda(), (batch_size, 1, 10, 1))).squeeze(-1)
            # shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            # can_pts = can_pts + shape_offset

        A, R, Th, joints = get_transform_params_torch(big_pose_params, self.SMPL, v_template, shapedirs, J_regressor)
        # J = torch.hstack((self.c_joints, torch.ones([self.c_joints.shape[0], 1])))
        # self.c_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        # can_pts = torch.sum(A[..., :3, :3] * can_pts[:, :, None], dim=-1)
        can_pts = torch.matmul(A[..., :3, :3], can_pts[..., None]).squeeze(-1)
        can_pts = can_pts + A[..., :3, 3]

        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(A[..., :3, :3], query_viewdirs[..., None]).squeeze(-1)
            return can_pts, query_viewdirs, v_template, posedirs, shapedirs, J_regressor, bweights, vert_ids

        return can_pts, v_template, posedirs, shapedirs, J_regressor, bweights, vert_ids

    def coarse_deform_c2source(self, params, t_params, t_vertices, query_pts, v_template, posedirs, shapedirs, J_regressor, bweights, vert_ids):
        bs = query_pts.shape[0]
        vertices_num = t_vertices.shape[1]

        # bweights_vertex = []
        # for key in gender:
        #     bweights_vertex.append(self.SMPL[key]['weights'])
        # bweights_vertex = torch.stack(bweights_vertex)

        # Find nearest smpl vertex
        # smpl_pts = t_vertices
        # _, vert_ids, _ = knn_points(query_pts.float(), smpl_pts.float(), K=1)
        # bweights = torch.gather(bweights_vertex, 1, vert_ids.expand(-1, -1, bweights_vertex.shape[-1]))

        # # add weights_correction, normalize weights
        # # bweights = F.softmax(bweights + 0.2*weights_correction, dim=1)
        # bweights = bweights + 0.2 * weights_correction # torch.Size([30786, 24])
        # bweights = bweights / torch.sum(bweights, dim=-1, keepdim=True)

        ### From Big To T Pose
        big_pose_params = t_params
        # big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(big_pose_params, self.SMPL, v_template, shapedirs, J_regressor)
        joints_num = joints.shape[1]
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        query_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        query_pts = torch.matmul(R_inv, query_pts[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:

            # posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = big_pose_params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.reshape(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            # pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            pose_offset = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(batch_size, vertices_num*3, -1).permute(0, 2, 1)).view(batch_size, -1, 3)
            pose_offset = torch.gather(pose_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts - pose_offset

            # From mean shape to normal shape
            # shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            # shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            # query_pts = query_pts + shape_offset[vert_ids.squeeze(0).reshape(-1)]
            shape_offset = torch.matmul(shapedirs[..., :params['shapes'].shape[-1]], torch.reshape(params['shapes'], (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + shape_offset

            # posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.reshape(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            # query_pts = query_pts + pose_offsets[vert_ids.squeeze(0).reshape(-1)]
            pose_offset = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(batch_size, vertices_num*3, -1).permute(0, 2,1)).view(batch_size, -1, 3)
            pose_offset = torch.gather(pose_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + pose_offset

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(params, self.SMPL, v_template, shapedirs, J_regressor)
        self.s_A = A
        A = torch.matmul(bweights, self.s_A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        # can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R.float())
        world_src_pts = torch.matmul(smpl_src_pts, R_inv) + Th.float()
        
        return smpl_src_pts, world_src_pts

    def projection(self, query_pts, R, T, K, face=None):

        RT = torch.cat([R, T], -1)
        # xyz = torch.repeat_interleave(query_pts, repeats=RT.shape[1], dim=1) #[bs, view_num, , 3]
        # xyz = torch.bmm(xyz.float(), RT[:, :, :, :3].transpose(2, 3).float()) + RT[:, :, :, 3:].transpose(2, 3).float()
        # xyz = torch.matmul(xyz[:, :, :, None].float(), RT[:, :, None, :, :3].transpose(3, 4).float()) + RT[:, :, None, :, 3:].transpose(3, 4).float()
        # xyz = torch.matmul(RT[:, :, None, :, :3].float(), xyz[..., None].float()) + RT[:, :, None, :, 3:].float()
        xyz = torch.matmul(RT[:, None, :, :3], query_pts[..., None]) + RT[:, None, :, 3:]
        if face is not None:
            # compute the normal vector for each vertex
            smpl_vertex_normal = compute_normal(query_pts, face) # [bs, 6890, 3]
            # smpl_vertex_mask = torch.einsum('bij,bij->bi', smpl_vertex_normal_cam, xyz.squeeze(1).squeeze(-1)) < 0 
            smpl_vertex_normal_cam = torch.matmul(RT[:, None, :, :3].float(), smpl_vertex_normal[:, :, :, None].float()) # [bs, 1, 6890, 3, 1]
            smpl_vertex_mask = (smpl_vertex_normal_cam * xyz).sum(-2).squeeze(-1) < 0 

        # xyz = torch.bmm(xyz, K.transpose(1, 2).float())
        xyz = torch.matmul(K[:, None], xyz)[..., 0].float()
        xy = xyz[..., :2] / (xyz[..., 2:] + 1e-5)

        if face is not None:
            return xy, smpl_vertex_mask 
        else:
            return xy

    def prepare_sp_input(self, vertex, xyz):

        self.big_box = True
        # obtain the bounds for coord construction
        min_xyz = torch.min(vertex, dim=1)[0]
        max_xyz = torch.max(vertex, dim=1)[0]

        if self.big_box:  # False
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[:, 2] -= 0.05
            max_xyz[:, 2] += 0.05

        bounds = torch.cat([min_xyz.unsqueeze(1), max_xyz.unsqueeze(1)], axis=1)


        dhw = xyz[:, :, [2, 1, 0]]
        min_dhw = min_xyz[:, [2, 1, 0]]
        max_dhw = max_xyz[:, [2, 1, 0]]
        voxel_size = torch.Tensor([0.005, 0.005, 0.005]).to(device=dhw.device)
        coord = torch.round((dhw - min_dhw.unsqueeze(1)) / voxel_size).to(dtype=torch.int32)

        # construct the output shape
        out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).to(dtype=torch.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x 
        sh = dhw.shape # torch.Size([1, 6890, 3])
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(coord)
        coord = coord.view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(out_sh, dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['bounds'] = bounds

        return sp_input, _#, pc_features

    def get_grid_coords(self, pts, sp_input):

        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor([0.005, 0.005, 0.005]).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

#----------------------------------------------------------------------------

class SparseConvNet(nn.Module):
    """Find the corresponding 3D feature of query point along the ray
    
    Attributes:
        conv: sparse convolutional layer 
        down: sparse convolutional layer with downsample 
    """
    def __init__(self, num_layers=2):
        super(SparseConvNet, self).__init__()
        self.num_layers = num_layers

        # self.conv0 = double_conv(3, 16, 'subm0')
        # self.down0 = stride_conv(16, 32, 'down0')

        # self.conv1 = double_conv(32, 32, 'subm1')
        # self.down1 = stride_conv(32, 64, 'down1')

        # self.conv2 = triple_conv(64, 64, 'subm2')
        # self.down2 = stride_conv(64, 128, 'down2')

        self.conv0 = double_conv(32, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 96, 'down2')

        self.conv3 = triple_conv(96, 96, 'subm3')
        self.down3 = stride_conv(96, 96, 'down3')

        self.conv4 = triple_conv(96, 96, 'subm4')

        self.channel = 32

    def forward(self, x, point_normalied_coords):
        """Find the corresponding 3D feature of query point along the ray.

        Args:
            x: Sparse Conv Tensor
            point_normalied_coords: Voxel grid coordinate, integer normalied to [-1, 1]
        
        Returns:
            features: Corresponding 3D feature of query point along the ray
        """
        features = []

        net = self.conv0(x)
        net = self.down0(net)

        # point_normalied_coords = point_normalied_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if self.num_layers > 1:
            net = self.conv1(net)
            net1 = net.dense()
            # torch.Size([1, 32, 1, 1, 4096])
            feature_1 = F.grid_sample(net1, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_1)
            self.channel = 32
            net = self.down1(net)
        
        if self.num_layers > 2:
            net = self.conv2(net)
            net2 = net.dense()
            # torch.Size([1, 64, 1, 1, 4096])
            feature_2 = F.grid_sample(net2, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_2)
            self.channel = 64
            net = self.down2(net)
        
        if self.num_layers > 3:
            net = self.conv3(net)
            net3 = net.dense()
            # 128
            feature_3 = F.grid_sample(net3, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_3)
            self.channel = 128
            net = self.down3(net)
        
        if self.num_layers > 4:
            net = self.conv4(net)
            net4 = net.dense()
            # 256
            feature_4 = F.grid_sample(net4, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_4)

        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4)).transpose(1,2)

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    # tmp = spconv.SubMConv3d(in_channels,
    #                       out_channels,
    #                       3,
    #                       bias=False,
    #                       indice_key=indice_key)
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())

#----------------------------------------------------------------------------

class ResNet18Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18Classifier, self).__init__()
        self.backbone = resnet18(pretrained=True)

    def forward(self, x, extract_feature=False):
        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if not extract_feature:
            x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if extract_feature:
            return x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x

#----------------------------------------------------------------------------

class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=None, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        # self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.freqs = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = torch.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        if x.shape[0]==0:
            embed = embed.view(x.shape[0], self.num_freqs*6)
        else:
            embed = embed.view(x.shape[0], -1)
            
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed

#----------------------------------------------------------------------------

def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out

#----------------------------------------------------------------------------

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=1.0),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=1.0)
        )
        
    def forward(self, sampled_features):
        # Aggregate features
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:]) # * (1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

class NeRFDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()

        W = 128
        self.with_viewdirs = True
        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = n_features + 39
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        nerf_input_ch_2 = n_features + W # 96 fused feature + 256 alpha feature
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)

    def forward(self, ray_points, ray_directions, sampled_features):

        bs, pt_num = ray_points.shape[0], ray_points.shape[1]
        x = self.pos_enc(ray_points.reshape(-1, 3)).reshape(bs, pt_num, -1)
        ray_directions = self.view_enc(ray_directions.reshape(-1, 3)).reshape(bs, pt_num, -1)
        x = torch.cat((x, sampled_features), dim=-1)
        h = x

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        sigma = self.alpha_linear(h)
        feature = self.feature_linear(h)

        if self.with_viewdirs:
            h = torch.cat([feature, ray_directions, sampled_features], -1)
        else:
            h = torch.cat([feature, sampled_features], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        rgb = torch.sigmoid(rgb) #* (1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF


        return {'rgb': rgb, 'sigma': sigma}

#----------------------------------------------------------------------------

# def make_layer(block, n_layers):
#     layers = []
#     for _ in range(n_layers):
#         layers.append(block())
#     return nn.Sequential(*layers)


# class ResidualDenseBlock_5C(nn.Module):
#     def __init__(self, nf=64, gc=32, bias=True):
#         super(ResidualDenseBlock_5C, self).__init__()
#         # gc: growth channel, i.e. intermediate channels
#         self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
#         self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         # initialization
#         # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         return x5 * 0.2 + x


# class RRDB(nn.Module):
#     '''Residual in Residual Dense Block'''

#     def __init__(self, nf, gc=32):
#         super(RRDB, self).__init__()
#         self.RDB1 = ResidualDenseBlock_5C(nf, gc)
#         self.RDB2 = ResidualDenseBlock_5C(nf, gc)
#         self.RDB3 = ResidualDenseBlock_5C(nf, gc)

#     def forward(self, x):
#         out = self.RDB1(x)
#         out = self.RDB2(out)
#         out = self.RDB3(out)
#         return out * 0.2 + x

# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x_diff, x_NeRF):
#         x = torch.cat([x_diff, x_NeRF], dim=1)
#         fea = self.conv_first(x)
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         fea = fea + trunk

#         # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv1(fea))
#         fea = self.lrelu(self.upconv2(fea))
#         out = self.conv_last(self.lrelu(self.HRconv(fea)))

#         # out = torch.sigmoid(out)

#         return out
