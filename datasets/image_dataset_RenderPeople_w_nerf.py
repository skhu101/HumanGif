import json
import random
from typing import List
from pathlib import Path

# from data.DNA_Rendering.dna_rendering_sample_code.SMCReader import SMCReader

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import imageio
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
# from datasets.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd
import cv2
import numpy as np 
import time
import os

from smpl.smpl_numpy import SMPL
from smplx.body_models import SMPLX

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    # TODO
    # mask_at_box = p_mask_at_box.sum(-1) >= 1

    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray(img, msk, K, R, T, bounds, image_scaling=1.0, white_bg=False):

    H, W = img.shape[:2]
    H, W = int(H * image_scaling), int(W * image_scaling)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

    K_scale = np.copy(K)
    K_scale[:2, :3] = K_scale[:2, :3] * image_scaling
    ray_o, ray_d = get_rays(H, W, K_scale, R, T)
    pose = np.concatenate([R, T], axis=1)
    
    real_bounds = np.zeros_like(bounds)
    real_bounds[0] = bounds[0] + 0.05
    real_bounds[1] = bounds[1] - 0.05
    bound_mask = get_bound_2d_mask(real_bounds, K_scale, pose, H, W)

    msk = msk * bound_mask
    img[bound_mask != 1] = 1 if white_bg else 0

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    near_all = np.zeros_like(ray_o[:,0])
    far_all = np.ones_like(ray_o[:,0])
    near_all[mask_at_box] = near 
    far_all[mask_at_box] = far 
    near = near_all
    far = far_all

    bkgd_msk = msk

    return img, ray_o, ray_d, near, far, mask_at_box, bkgd_msk

class ImageDataset(Dataset):
    def __init__(
        self,
        video_folder: str,
        image_size: int = 768,
        sample_margin: int = 30,
        data_parts: list = ["all"],
        guids: list = ["depth", "normal", "semantic_map", "dwpose", "nerf"],
        extra_region: list = [],
        bbox_crop=True,
        bbox_resize_ratio=(0.8, 1.2),
        aug_type: str = "Resize",  # "Resize" or "Padding"
        select_face=False,
        nerf_rs_scale=1.0,
        white_bg=False,
        debug=False,
    ):
        super().__init__()
        self.video_folder = video_folder
        self.image_size = image_size
        self.sample_margin = sample_margin
        self.data_parts = data_parts
        self.guids = guids
        self.extra_region = extra_region
        self.bbox_crop = bbox_crop
        self.bbox_resize_ratio = bbox_resize_ratio
        self.aug_type = aug_type
        self.select_face = select_face
        self.nerf_rs_scale = nerf_rs_scale
        self.white_bg = white_bg
        self.debug = debug
        
        self.data_lst = self.generate_data_lst()
        
        self.clip_image_processor = CLIPImageProcessor()
        self.pixel_transform, self.guid_transform = self.setup_transform()

        self.smpl_model, self.big_pose_smpl_param, self.big_pose_smpl_vertices, self.big_pose_world_bound = {}, {}, {}, {}
        for gender in ['female', 'male', 'neutral']:
            self.smpl_model[gender] = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')

            # SMPL in canonical space
            big_pose_smpl_param = {}
            big_pose_smpl_param['R'] = np.ones((3,3)).astype(np.float32)
            big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
            big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
            big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
            big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
            big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
            big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
            big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

            big_pose_smpl_vertices, _ = self.smpl_model[gender](big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
            big_pose_smpl_vertices = np.array(big_pose_smpl_vertices).astype(np.float32)
            big_pose_min_xyz = np.min(big_pose_smpl_vertices, axis=0)
            big_pose_max_xyz = np.max(big_pose_smpl_vertices, axis=0)
            big_pose_min_xyz -= 0.05
            big_pose_max_xyz += 0.05
            big_pose_min_xyz[2] -= 0.1
            big_pose_max_xyz[2] += 0.1
            big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

            self.big_pose_smpl_param[gender], self.big_pose_smpl_vertices[gender], self.big_pose_world_bound[gender] = big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound

    def generate_data_lst(self):
        video_folder = Path(self.video_folder)
        if "all" in self.data_parts:
            data_parts = sorted(video_folder.iterdir())
        else:
            data_parts = [(video_folder / p) for p in self.data_parts]

        # if 'DNA_Rendering' in self.video_folder:
        #     data_parts = [(p / 'camera0022') for p in data_parts]

        data_parts_lst = []
        for data_part in data_parts:
            for view_index in range(36):
                img_dir = data_part / f'camera{str(view_index).zfill(4)}' / 'images'
                img_name_lst = sorted([img.name for img in img_dir.glob("*.jpg")])
                for img_name in img_name_lst:
                    data_parts_lst.append(img_dir / img_name)
        data_parts = data_parts_lst

        # data_parts_lst = []
        # if 'RenderPeople' in self.video_folder:
        #     for view_index in range(36):
        #         data_parts_lst.extend([(p / f'camera{str(view_index).zfill(4)}') for p in data_parts])
        # data_parts = data_parts_lst

        data_lst = []
        for data_part in data_parts:
            # if self.is_valid(data_part):
            data_lst += [data_part]
            # for video_dir in sorted(data_part.iterdir()):
            #     if self.is_valid(video_dir):
            #         data_lst += [video_dir]
        return data_lst
    
    def is_valid(self, video_dir: Path):
        if not (video_dir / "images").is_dir(): return False
        video_length = len(list((video_dir / "images").iterdir()))
        for guid in self.guids:
            if guid != 'nerf':
                guid_length = len(list((video_dir / guid).iterdir()))
                if guid_length == 0 or guid_length != video_length:
                    return False
        if self.select_face:
            if not (video_dir / "face_images").is_dir():
                return False
            else:
                face_img_length = len(list((video_dir / "face_images").iterdir()))
                if face_img_length == 0:
                    return False
        return True
    
    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.image_size / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.image_size, self.image_size
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img
    
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose = ([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            pixel_transform = transforms.Compose([
                transforms.CenterCrop(size=self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.CenterCrop(size=self.image_size),
                transforms.ToTensor(),
            ])
            # pixel_transform = transforms.Compose([
            #     transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.5], [0.5]),
            # ])
            # guid_transform = transforms.Compose([
            #     transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
            #     transforms.ToTensor(),
            # ])
        
        return pixel_transform, guid_transform            
            
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    def set_tgt_idx(self, ref_img_idx, video_length):
        margin = self.sample_margin
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)
            
        return tgt_img_idx
    
    def get_mask(self, mask_path):
        msk = imageio.imread(mask_path)
        msk[msk!=0]=255
        return msk

    def prepare_smpl_params(self, smpl_path, pose_index):
        params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
        params = {}
        params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
        params['poses'] = np.zeros((1,72)).astype(np.float32)
        params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
        params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
        params['R'] = np.eye(3).astype(np.float32)
        params['Th'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
        return params

    def prepare_input(self, smpl_path, pose_index, gender):

        params = self.prepare_smpl_params(smpl_path, pose_index)
        xyz, _ = self.smpl_model[gender](params['poses'], params['shapes'].reshape(-1))
        xyz = (np.matmul(xyz, params['R'].transpose()) + params['Th']).astype(np.float32)
        vertices = xyz

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[2] -= 0.1
        max_xyz[2] += 0.1
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        return world_bounds, vertices, params

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        video_dir = self.data_lst[idx]
        if self.debug:
            video_dir = self.data_lst[0]
        ### debug ###
        # video_dir = Path('/group/3d/shoukanghu/champ/data/RenderPeople/train/seq_000000-rp_aaron_rigged_001/camera0000/images/0020.jpg')
        ###
        gender = 'neutral'

        # tgt image index
        tgt_view_index = int(str(video_dir).split('/')[-3][-4:])
        tgt_img_idx = int(str(video_dir).split('/')[-1].split('.')[0])
        tgt_img_name = str(video_dir).split('/')[-1]

        # reference image index
        view_index_lst = [i for i in range(36)] 
        ref_view_index = np.random.choice(view_index_lst, size=1, replace=False)[0]
        # ref_img_idx = tgt_img_idx
        # ref_view_index = int(str(video_dir).split('/')[-1][-4:])
        # ref_img_name = random.choice([img.name for img in video_dir.glob("*.jpg")])
        # ref_img_idx = int(ref_img_name.split('.')[0])
        ### debug ###
        # ref_view_index = 0 #tgt_view_index
        ###

        if self.debug:
            ref_view_index = tgt_view_index


        # set reference video_dir
        video_dir = Path(str(video_dir)[:-9].replace(f'camera{str(tgt_view_index).zfill(4)}', f'camera{str(ref_view_index).zfill(4)}'))

        # reference image view index
        img_path_lst = sorted([img.name for img in video_dir.glob("*.jpg")])
        if random.random() < 0.5:
            ref_img_idx = tgt_img_idx
        else:
            video_length = len(img_path_lst)
            ref_img_idx = self.set_tgt_idx(tgt_img_idx, video_length)

        ### debug ###
        # ref_img_idx = tgt_img_idx
        ###

        if self.debug:
            ref_img_idx = tgt_img_idx

        # load reference image and mask
        ref_img_path = os.path.join(str(video_dir), str(ref_img_idx).zfill(4)+'.jpg')            
        ref_mask_path = ref_img_path.replace('images', 'msk').replace('jpg', 'png')

        ref_img = np.array(imageio.imread(ref_img_path))
        ref_img_nerf = ref_img.copy()
        ref_msk = np.array(self.get_mask(ref_mask_path)) / 255.
        ref_img[ref_msk == 0] = 255 if self.white_bg else 0
        ref_img_nerf[ref_msk == 0] = 0

        # Load reference K, R, T
        camera_file = str(video_dir) + '/../../cameras.json'
        camera = json.load(open(camera_file))
        ref_K = np.array(camera[f'camera{str(ref_view_index).zfill(4)}']['K']).astype(np.float32)
        ref_R = np.array(camera[f'camera{str(ref_view_index).zfill(4)}']['R']).astype(np.float32)
        ref_T = np.array(camera[f'camera{str(ref_view_index).zfill(4)}']['T']).reshape(-1, 1).astype(np.float32)

        # prepare reference PIL and NeRF image 
        ref_img_pil = Image.fromarray(ref_img)
        ref_img_nerf = ref_img_nerf.astype(np.float32) / 255.

        # prepare smpl at the reference view
        smpl_path = str(video_dir) + '/../../outputs_re_fitting/refit_smpl_2nd.npz'
        _, ref_world_vertex, ref_smpl_param = self.prepare_input(smpl_path, ref_img_idx, gender)

        # tgt image index
        # img_path_lst = sorted([img.name for img in video_dir.glob("*.jpg")])
        # tgt_img_name = img_path_lst[tgt_img_idx]
        # view_index_lst = [i for i in range(36)] 
        # tgt_view_index = np.random.choice(view_index_lst, size=1, replace=False)[0]
        # tgt_img_idx = ref_img_idx
        # video_length = len(img_path_lst)
        # tgt_img_idx = self.set_tgt_idx(ref_img_idx, video_length)
        # tgt_img_idx = int(tgt_img_name.split('.')[0])

        # ### debug ###
        # tgt_img_idx, tgt_view_index = 0, 0
        # tgt_img_name = img_path_lst[tgt_img_idx]
        # tgt_view_index = ref_view_index
        # tgt_img_idx = ref_img_idx
        # ###

        # load tgt image and mask
        tgt_img_path = os.path.join(str(video_dir).replace(f'camera{str(ref_view_index).zfill(4)}', f'camera{str(tgt_view_index).zfill(4)}'), str(tgt_img_idx).zfill(4)+'.jpg')            
        tgt_mask_path = tgt_img_path.replace('images', 'msk').replace('jpg', 'png')

        tgt_img = np.array(imageio.imread(tgt_img_path))
        tgt_img_nerf = tgt_img.copy()
        tgt_msk = np.array(self.get_mask(tgt_mask_path)) / 255.
        tgt_img[tgt_msk == 0] = 255 if self.white_bg else 0
        tgt_img_nerf[tgt_msk == 0] = 0

        # Load tgt K, R, T
        tgt_K = np.array(camera[f'camera{str(tgt_view_index).zfill(4)}']['K']).astype(np.float32)
        tgt_R = np.array(camera[f'camera{str(tgt_view_index).zfill(4)}']['R']).astype(np.float32)
        tgt_T = np.array(camera[f'camera{str(tgt_view_index).zfill(4)}']['T']).reshape(-1, 1).astype(np.float32)

        # prepare tgt PIL and NeRF image 
        tgt_img_pil = Image.fromarray(tgt_img)
        tgt_img_nerf = tgt_img_nerf.astype(np.float32) / 255.

        # prepare smpl at the target view
        tgt_world_bound, tgt_world_vertex, tgt_smpl_param = self.prepare_input(smpl_path, tgt_img_idx, gender)

        # Sample rays in target space world coordinate
        tgt_img_nerf, tgt_ray_o, tgt_ray_d, tgt_near, tgt_far, tgt_mask_at_box, tgt_bkgd_msk = sample_ray(
                tgt_img_nerf, tgt_msk, tgt_K, tgt_R, tgt_T, tgt_world_bound, image_scaling=self.nerf_rs_scale, white_bg=False)

        # guidance images
        tgt_guid_pil_lst = []
        tgt_video_dir = Path(str(video_dir)[:-11] + str(tgt_view_index).zfill(4))
        for guid in self.guids:
            if guid != 'nerf':
                # video_dir = Path(str(video_dir)[:-11] + str(tgt_view_index).zfill(4))
                ### debug ###
                # guid = 'msk'
                ###
                guid_img_path = tgt_video_dir / guid / tgt_img_name.replace('jpg', 'png')
                if guid == "semantic_map":
                    # mask_img_path = video_dir / "mask" / tgt_img_name
                    # guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                else:
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                tgt_guid_pil_lst += [guid_img_pil]
            

        # augmentation
        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.pixel_transform, state)
        tgt_guid = self.augmentation(tgt_guid_pil_lst, self.guid_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.pixel_transform, state)
        clip_img = self.clip_image_processor(
            images=ref_img_pil, return_tensor="pt"
        ).pixel_values[0]
        
        sample = dict(
            tgt_img=tgt_img,
            tgt_guid=tgt_guid,
            ref_img=ref_img_vae,
            clip_img=clip_img,

            gender=gender,

            # canonical space
            big_pose_smpl_param=self.big_pose_smpl_param[gender],
            big_pose_world_vertex=self.big_pose_smpl_vertices[gender],
            big_pose_world_bound=self.big_pose_world_bound[gender],

            # reference view
            ref_img_nerf=np.transpose(ref_img_nerf, (2,0,1)),
            ref_smpl_param=ref_smpl_param,
            ref_world_vertex=ref_world_vertex,
            ref_K=ref_K,
            ref_R=ref_R,
            ref_T=ref_T,

            # target view
            tgt_img_nerf=np.transpose(tgt_img_nerf, (2,0,1)),
            tgt_smpl_param=tgt_smpl_param,
            tgt_world_vertex=tgt_world_vertex, 
            tgt_ray_o=tgt_ray_o,
            tgt_ray_d=tgt_ray_d,
            tgt_near=tgt_near,
            tgt_far=tgt_far,
            tgt_mask_at_box=tgt_mask_at_box,
            tgt_bkgd_msk=tgt_bkgd_msk,         
        )
        
        return sample
