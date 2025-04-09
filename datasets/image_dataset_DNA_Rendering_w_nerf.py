import json
import random
from typing import List
from pathlib import Path

from data.DNA_Rendering.dna_rendering_sample_code.SMCReader import SMCReader

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
# from datasets.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd
import cv2
import numpy as np 
import time

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
        image_ratio=0.25,
        nerf_rs_scale=1.0,
        white_bg=False,
        pretrain_nerf=False,
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
        self.image_ratio = image_ratio
        self.nerf_rs_scale = nerf_rs_scale
        self.white_bg = white_bg
        self.pretrain_nerf = pretrain_nerf
        
        self.data_lst = self.generate_data_lst()
        
        self.clip_image_processor = CLIPImageProcessor()
        self.pixel_transform, self.guid_transform = self.setup_transform()

        self.smpl_model, self.big_pose_smpl_param, self.big_pose_smpl_vertices, self.big_pose_world_bound = {}, {}, {}, {}
        for gender in ['female', 'male', 'neutral']:
            self.smpl_model[gender] = SMPLX('assets/models/smplx/', smpl_type='smplx',
                                        gender=gender, use_face_contour=True, flat_hand_mean=False, use_pca=False, 
                                        num_pca_comps=24, num_betas=10,
                                        num_expression_coeffs=10,
                                        ext='npz')

            # SMPL in canonical space
            big_pose_smpl_param = {}
            big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
            big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
            big_pose_smpl_param['global_orient'] = np.zeros((1,3)).astype(np.float32)
            big_pose_smpl_param['betas'] = np.zeros((1,10)).astype(np.float32)
            big_pose_smpl_param['body_pose'] = np.zeros((1,63)).astype(np.float32)
            big_pose_smpl_param['jaw_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_smpl_param['left_hand_pose'] = np.zeros((1,45)).astype(np.float32)
            big_pose_smpl_param['right_hand_pose'] = np.zeros((1,45)).astype(np.float32)
            big_pose_smpl_param['leye_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_smpl_param['reye_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_smpl_param['expression'] = np.zeros((1,10)).astype(np.float32)
            big_pose_smpl_param['transl'] = np.zeros((1,3)).astype(np.float32)
            big_pose_smpl_param['body_pose'][0, 2] = 45/180*np.array(np.pi)
            big_pose_smpl_param['body_pose'][0, 5] = -45/180*np.array(np.pi)
            big_pose_smpl_param['body_pose'][0, 20] = -30/180*np.array(np.pi)
            big_pose_smpl_param['body_pose'][0, 23] = 30/180*np.array(np.pi)

            big_pose_smpl_param_tensor= {}
            for key in big_pose_smpl_param.keys():
                big_pose_smpl_param_tensor[key] = torch.from_numpy(big_pose_smpl_param[key])

            body_model_output = self.smpl_model[gender](
                global_orient=big_pose_smpl_param_tensor['global_orient'],
                betas=big_pose_smpl_param_tensor['betas'],
                body_pose=big_pose_smpl_param_tensor['body_pose'],
                jaw_pose=big_pose_smpl_param_tensor['jaw_pose'],
                left_hand_pose=big_pose_smpl_param_tensor['left_hand_pose'],
                right_hand_pose=big_pose_smpl_param_tensor['right_hand_pose'],
                leye_pose=big_pose_smpl_param_tensor['leye_pose'],
                reye_pose=big_pose_smpl_param_tensor['reye_pose'],
                expression=big_pose_smpl_param_tensor['expression'],
                transl=big_pose_smpl_param_tensor['transl'],
                return_full_pose=True,
            )

            big_pose_smpl_param['poses'] = body_model_output.full_pose.detach()
            big_pose_smpl_param['shapes'] = np.concatenate([big_pose_smpl_param['betas'], big_pose_smpl_param['expression']], axis=-1)
            big_pose_smpl_vertices = np.array(body_model_output.vertices.detach()).reshape(-1,3).astype(np.float32)
            
            # obtain the original bounds for point sampling
            big_pose_min_xyz = np.min(big_pose_smpl_vertices, axis=0)
            big_pose_max_xyz = np.max(big_pose_smpl_vertices, axis=0)
            big_pose_min_xyz -= 0.05
            big_pose_max_xyz += 0.05
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

        # data_parts_lst = []
        # if 'DNA_Rendering' in self.video_folder:
        #     for view_index in range(48):
        #         data_parts_lst.extend([(p / f'camera{str(view_index).zfill(4)}') for p in data_parts])
        # data_parts = data_parts_lst

        data_parts_lst = []
        for data_part in data_parts:
            for view_index in range(48):
                img_dir = data_part / f'camera{str(view_index).zfill(4)}' / 'images'
                if view_index == 22:
                    img_name_lst = sorted([img.name for img in img_dir.glob("*.png")])
                else:
                    img_name_lst = sorted([img.name for img in img_dir.glob("*.jpg")])
                for img_name in img_name_lst[::5]:
                    data_parts_lst.append(img_dir / img_name)
        data_parts = data_parts_lst

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
    
    def prepare_smpl_data(self, smpl_dict, gender):

        smpl_data = {}
        smpl_data['global_orient'] = smpl_dict['fullpose'][0].reshape(-1)
        smpl_data['body_pose'] = smpl_dict['fullpose'][1:22].reshape(-1)
        smpl_data['jaw_pose'] = smpl_dict['fullpose'][22].reshape(-1)
        smpl_data['leye_pose'] = smpl_dict['fullpose'][23].reshape(-1)
        smpl_data['reye_pose'] = smpl_dict['fullpose'][24].reshape(-1)
        smpl_data['left_hand_pose'] = smpl_dict['fullpose'][25:40].reshape(-1)
        smpl_data['right_hand_pose'] = smpl_dict['fullpose'][40:55].reshape(-1)
        smpl_data['transl'] = smpl_dict['transl'].reshape(-1)
        smpl_data['betas'] = smpl_dict['betas'].reshape(-1)[:10]
        smpl_data['expression'] = np.zeros(10) #smpl_dict['expression'].reshape(-1)

        # smpl_data['betas'] = np.zeros((10)).astype(np.float32)
        # smpl_data['body_pose'] = np.zeros((63)).astype(np.float32)
        # smpl_data['jaw_pose'] = np.zeros((3)).astype(np.float32)
        # smpl_data['left_hand_pose'] = np.zeros((45)).astype(np.float32)
        # smpl_data['right_hand_pose'] = np.zeros((45)).astype(np.float32)
        # smpl_data['leye_pose'] = np.zeros((3)).astype(np.float32)
        # smpl_data['reye_pose'] = np.zeros((3)).astype(np.float32)
        # smpl_data['expression'] = np.zeros((10)).astype(np.float32)
        # smpl_data['body_pose'][2] = 45/180*np.array(np.pi)
        # smpl_data['body_pose'][5] = -45/180*np.array(np.pi)
        # smpl_data['body_pose'][20] = -30/180*np.array(np.pi)
        # smpl_data['body_pose'][23] = 30/180*np.array(np.pi)

        # load smpl data
        smpl_param = {
            'global_orient': np.expand_dims(smpl_data['global_orient'].astype(np.float32), axis=0),
            'transl': np.expand_dims(smpl_data['transl'].astype(np.float32), axis=0),
            'body_pose': np.expand_dims(smpl_data['body_pose'].astype(np.float32), axis=0),
            'jaw_pose': np.expand_dims(smpl_data['jaw_pose'].astype(np.float32), axis=0),
            'betas': np.expand_dims(smpl_data['betas'].astype(np.float32), axis=0),
            'expression': np.expand_dims(smpl_data['expression'].astype(np.float32), axis=0),
            'leye_pose': np.expand_dims(smpl_data['leye_pose'].astype(np.float32), axis=0),
            'reye_pose': np.expand_dims(smpl_data['reye_pose'].astype(np.float32), axis=0),
            'left_hand_pose': np.expand_dims(smpl_data['left_hand_pose'].astype(np.float32), axis=0),
            'right_hand_pose': np.expand_dims(smpl_data['right_hand_pose'].astype(np.float32), axis=0),
            }

        smpl_param['R'] = np.eye(3).astype(np.float32)
        smpl_param['Th'] = smpl_param['transl'].astype(np.float32)

        smpl_param_tensor = {}
        for key in smpl_param.keys():
            smpl_param_tensor[key] = torch.from_numpy(smpl_param[key])

        body_model_output = self.smpl_model[gender](
            global_orient=smpl_param_tensor['global_orient'],
            betas=smpl_param_tensor['betas'],
            body_pose=smpl_param_tensor['body_pose'],
            jaw_pose=smpl_param_tensor['jaw_pose'],
            left_hand_pose=smpl_param_tensor['left_hand_pose'],
            right_hand_pose=smpl_param_tensor['right_hand_pose'],
            leye_pose=smpl_param_tensor['leye_pose'],
            reye_pose=smpl_param_tensor['reye_pose'],
            expression=smpl_param_tensor['expression'],
            transl=smpl_param_tensor['transl'],
            return_full_pose=True,
        )

        smpl_param['poses'] = body_model_output.full_pose.detach()
        smpl_param['shapes'] = np.concatenate([smpl_param['betas'], smpl_param['expression']], axis=-1)

        world_vertex = np.array(body_model_output.vertices.detach()).reshape(-1,3).astype(np.float32)
        # obtain the original bounds for point sampling
        min_xyz = np.min(world_vertex, axis=0)
        max_xyz = np.max(world_vertex, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        return smpl_param, world_vertex, world_bound


    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        video_dir = self.data_lst[idx]

        # ### plot ###
        # video_dir = Path('data/DNA_Rendering/train/Part_2_0195_01/camera0029/images/000001.png')
        # ###

        # load smc_reader and smc_annots_reader
        part_name = str(video_dir).split('/')[-4][:-8]
        smc_name = str(video_dir).split('/')[-4][-7:]
        tgt_view_index = int(str(video_dir).split('/')[-3][-4:])
        tgt_img_idx = int(str(video_dir).split('/')[-1][:-4])
        smc_path = f'data/DNA_Rendering/{part_name}/dna_rendering_part{part_name[-1]}_main/{smc_name}.smc'
        smc_reader = SMCReader(smc_path)
        annots_file_path = smc_path.replace('main', 'annotations').split('.')[0] + '_annots.smc'
        smc_annots_reader = SMCReader(annots_file_path)

        gender = smc_reader.actor_info['gender']

        # reference image view index
        view_index_lst = [i for i in range(48)] 
        ref_view_index = np.random.choice(view_index_lst, size=1, replace=False)[0]


        video_dir = Path(str(video_dir)[:-18].replace(f'camera{str(tgt_view_index).zfill(4)}', f'camera{str(ref_view_index).zfill(4)}'))

        # reference image index
        normal_dir = video_dir / "normal"
        if self.select_face:
            face_img_dir = video_dir / "face_images"
            face_img_lst = [img.name for img in face_img_dir.glob("*.png")]
            ref_img_name = random.choice(face_img_lst)
        else:
            ref_normal_name = random.choice([img.name for img in normal_dir.glob("*.png")])
        img_path_lst = sorted([img.name for img in normal_dir.glob("*.png")])
        # ref_img_idx = int(ref_normal_name.split('.')[0])
        tgt_img_name = img_path_lst[tgt_img_idx]

        # reference image view index
        if self.pretrain_nerf:
            ref_img_idx = tgt_img_idx
        else:
            if random.random() < 0.5:
                ref_img_idx = tgt_img_idx
            else:
                video_length = len(img_path_lst)
                ref_img_idx = self.set_tgt_idx(tgt_img_idx, video_length)

        # ### plot ###
        # ref_img_idx = 1
        # ref_view_index = 29
        # ###

        # load reference image and mask
        ref_img = smc_reader.get_img('Camera_5mp', ref_view_index, Frame_id=ref_img_idx, Image_type='color')
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_msk = smc_annots_reader.get_mask(ref_view_index, Frame_id=ref_img_idx)

        # Load reference K, R, T
        cam_params = smc_annots_reader.get_Calibration(ref_view_index)
        ref_K, ref_D = cam_params['K'].astype(np.float32), cam_params['D'].astype(np.float32)

        # load camera 
        ref_c2w = np.eye(4)
        ref_c2w[:3,:3] = cam_params['RT'][:3, :3]
        ref_c2w[:3,3:4] = cam_params['RT'][:3, 3].reshape(-1, 1)
        ref_w2c = np.linalg.inv(ref_c2w)
        ref_R = ref_w2c[:3,:3].astype(np.float32)
        ref_T = ref_w2c[:3, 3].reshape(-1, 1).astype(np.float32)

        # undistort image and mask
        H, W = int(ref_img.shape[0]), int(ref_img.shape[1])
        newcamera, roi = cv2.getOptimalNewCameraMatrix(ref_K, ref_D, (W, H), 0) 
        mapx, mapy = cv2.initUndistortRectifyMap(ref_K, ref_D, None, newcamera, (W, H), 5)
        ref_img = cv2.remap(ref_img, mapx, mapy, cv2.INTER_LINEAR)
        ref_msk = cv2.remap(ref_msk, mapx, mapy, cv2.INTER_LINEAR)
        ref_img_nerf = ref_img.copy()

        ref_img[ref_msk == 0] = 255 if self.white_bg else 0
        ref_img_nerf[ref_msk == 0] = 0
        if self.image_ratio != 1.:
            H, W = int(ref_img.shape[0] * self.image_ratio), int(ref_img.shape[1] * self.image_ratio)
            ref_img = cv2.resize(ref_img, (W, H), interpolation=cv2.INTER_AREA)
            ref_img_nerf = cv2.resize(ref_img_nerf, (W, H), interpolation=cv2.INTER_AREA)
            ref_K[:2] = ref_K[:2] * self.image_ratio
        ref_img_pil = Image.fromarray(ref_img)

        if self.nerf_rs_scale != 1.:
            H_nerf, W_nerf = int(ref_img_nerf.shape[0] * self.nerf_rs_scale), int(ref_img_nerf.shape[1] * self.nerf_rs_scale)
            ref_img_nerf = cv2.resize(ref_img_nerf, (W_nerf, H_nerf), interpolation=cv2.INTER_AREA)
            ref_K[:2] = ref_K[:2] * self.nerf_rs_scale
        ref_img_nerf = ref_img_nerf.astype(np.float32) / 255.

        # prepare smpl at the reference view
        ref_smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=ref_img_idx)
        ref_smpl_param, ref_world_vertex, _ = self.prepare_smpl_data(ref_smpl_dict, gender)

        # tgt image index
        # view_index_lst = [i for i in range(smc_reader.Camera_5mp_info['num_device'])] 
        # tgt_view_index = np.random.choice(view_index_lst, size=1, replace=False)[0]

        # video_length = len(img_path_lst)
        # tgt_img_idx = self.set_tgt_idx(ref_img_idx, video_length)
        # tgt_img_name = img_path_lst[tgt_img_idx]
        # tgt_img_idx = int(tgt_img_name.split('.')[0])

        # load tgt image and mask
        tgt_img = smc_reader.get_img('Camera_5mp', tgt_view_index, Frame_id=tgt_img_idx, Image_type='color')
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
        tgt_msk = smc_annots_reader.get_mask(tgt_view_index, Frame_id=tgt_img_idx)
        tgt_msk[tgt_msk!=0] = 1

        # Load tgt K, R, T
        cam_params = smc_annots_reader.get_Calibration(tgt_view_index)
        tgt_K, tgt_D = cam_params['K'].astype(np.float32), cam_params['D'].astype(np.float32)

        # load camera 
        tgt_c2w = np.eye(4)
        tgt_c2w[:3,:3] = cam_params['RT'][:3, :3]
        tgt_c2w[:3,3:4] = cam_params['RT'][:3, 3].reshape(-1, 1)
        tgt_w2c = np.linalg.inv(tgt_c2w)
        tgt_R = tgt_w2c[:3, :3].astype(np.float32)
        tgt_T = tgt_w2c[:3, 3].reshape(-1, 1).astype(np.float32)

        # undistort tgt image and mask
        H, W = int(tgt_img.shape[0]), int(tgt_img.shape[1])
        newcamera, roi = cv2.getOptimalNewCameraMatrix(tgt_K, tgt_D, (W, H), 0) 
        mapx, mapy = cv2.initUndistortRectifyMap(tgt_K, tgt_D, None, newcamera, (W, H), 5)
        tgt_img = cv2.remap(tgt_img, mapx, mapy, cv2.INTER_LINEAR)
        tgt_msk = cv2.remap(tgt_msk, mapx, mapy, cv2.INTER_LINEAR)
        # tgt_img = cv2.undistort(tgt_img, tgt_K, tgt_D)
        # tgt_msk = cv2.undistort(tgt_msk, tgt_K, tgt_D)
        tgt_img_nerf = tgt_img.copy()

        tgt_img[tgt_msk == 0] = 255 if self.white_bg else 0
        tgt_img_nerf[tgt_msk == 0] = 0
        if self.image_ratio != 1.:
            H, W = int(tgt_img.shape[0] * self.image_ratio), int(tgt_img.shape[1] * self.image_ratio)
            tgt_img = cv2.resize(tgt_img, (W, H), interpolation=cv2.INTER_AREA)
            tgt_msk = cv2.resize(tgt_msk, (W, H), interpolation=cv2.INTER_AREA)
            tgt_img_nerf = cv2.resize(tgt_img_nerf, (W, H), interpolation=cv2.INTER_AREA)
            tgt_K[:2] = tgt_K[:2] * self.image_ratio
        tgt_img_pil = Image.fromarray(tgt_img)
        tgt_img_nerf = tgt_img_nerf.astype(np.float32) / 255.


        # prepare smpl at the target view
        tgt_smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=tgt_img_idx)
        tgt_smpl_param, tgt_world_vertex, tgt_world_bound = self.prepare_smpl_data(tgt_smpl_dict, gender)

        # Sample rays in target space world coordinate
        tgt_img_nerf, tgt_ray_o, tgt_ray_d, tgt_near, tgt_far, tgt_mask_at_box, tgt_bkgd_msk = sample_ray(
                tgt_img_nerf, tgt_msk, tgt_K, tgt_R, tgt_T, tgt_world_bound, image_scaling=self.nerf_rs_scale, white_bg=False)

        # # reference image
        # img_dir = video_dir / "images"
        # if self.select_face:
        #     face_img_dir = video_dir / "face_images"
        #     face_img_lst = [img.name for img in face_img_dir.glob("*.png")]
        #     ref_img_name = random.choice(face_img_lst)
        # else:
        #     ref_img_name = random.choice([img.name for img in img_dir.glob("*.png")])
        # ref_img_path = img_dir / ref_img_name
        # ref_img_pil = Image.open(ref_img_path)
        
        # img_path_lst = sorted([img.name for img in img_dir.glob("*.png")])
        # ref_img_idx = img_path_lst.index(ref_img_name)
        
        # # target image
        # video_length = len(img_path_lst)
        # tgt_img_idx = self.set_tgt_idx(ref_img_idx, video_length)
        # tgt_img_name = img_path_lst[tgt_img_idx]
        # tgt_img_pil = Image.open(img_dir / img_path_lst[tgt_img_idx])

        # guidance images
        tgt_guid_pil_lst = []
        for guid in self.guids:
            if guid != 'nerf':
                video_dir = Path(str(video_dir)[:-4] + str(tgt_view_index).zfill(4))
                guid_img_path = video_dir / guid / tgt_img_name
                if guid == "semantic_map":
                    # mask_img_path = video_dir / "mask" / tgt_img_name
                    # guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                else:
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                W, H = guid_img_pil.size
                resize_ratio = min(tgt_img_pil.size) / min(W, H)
                new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
                guid_img_pil = guid_img_pil.resize((new_W, new_H))
                tgt_guid_pil_lst += [guid_img_pil]
        # bbox crop
        # if self.bbox_crop:
        #     human_bbox_json_path = video_dir / "human_bbox.json"
        #     with open(human_bbox_json_path) as bbox_fp:
        #         human_bboxes = json.load(bbox_fp)
        #     resize_scale = random.uniform(*self.bbox_resize_ratio)
        #     ref_W, ref_H = ref_img_pil.size
        #     ref_bbox = process_bbox(human_bboxes[ref_img_idx], ref_H, ref_W, resize_scale)
        #     ref_img_pil = crop_bbox(ref_img_pil, ref_bbox)
        #     tgt_bbox = process_bbox(human_bboxes[tgt_img_idx], ref_H, ref_W, resize_scale)
        #     tgt_img_pil = crop_bbox(tgt_img_pil, tgt_bbox)
        #     tgt_guid_pil_lst = [crop_bbox(guid_pil, tgt_bbox) for guid_pil in tgt_guid_pil_lst]
        
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
