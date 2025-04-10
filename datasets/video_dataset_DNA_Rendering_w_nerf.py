import json
import random
from pathlib import Path

from data.DNA_Rendering.dna_rendering_sample_code.SMCReader import SMCReader

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
# from datasets.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd

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

def sample_ray(img, msk, K, R, T, bounds, image_scaling=1.0):

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
    img[bound_mask != 1] = 0

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

class VideoDataset(Dataset):
    def __init__(
        self,
        video_folder: str,
        image_size: int = 512,
        sample_frames: int = 24,
        sample_rate: int = 4,
        data_parts: list = ["all"],
        guids: list = ["depth", "normal", "semantic_map", "dwpose", "hrnet"],
        extra_region: list = [],
        bbox_crop: bool = True,
        bbox_resize_ratio: tuple = (0.8, 1.2),
        aug_type: str = "Resize",
        select_face: bool = False,
        crossview_num: int = 4,
        image_ratio: float = 0.375,
        nerf_rs_scale: float = 0.625,
    ):
        super().__init__()
        self.video_folder = video_folder
        self.image_size = image_size
        self.sample_frames = sample_frames
        self.sample_rate = sample_rate
        self.data_parts = data_parts
        self.guids = guids
        self.extra_region = extra_region
        self.bbox_crop = bbox_crop
        self.bbox_resize_ratio = bbox_resize_ratio
        self.aug_type = aug_type        
        self.select_face = select_face
        self.crossview_num = crossview_num
        self.image_ratio = image_ratio
        self.nerf_rs_scale = nerf_rs_scale
        
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

            big_pose_smpl_param['poses'] = np.array(body_model_output.full_pose.detach()).astype(np.float32)
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

        data_parts_lst = []
        if 'DNA_Rendering' in self.video_folder:
            for view_index in range(48):
                # data_parts_lst.append((p / f'camera{str(view_index).zfill(4)}'))
                data_parts_lst.extend([(p / f'camera{str(view_index).zfill(4)}') for p in data_parts])
        data_parts = data_parts_lst

        data_lst = []
        for data_part in data_parts:
            # if self.is_valid(data_part):
            data_lst += [data_part]
            # for video_dir in tqdm(sorted(data_part.iterdir())):
            #     if self.is_valid(video_dir):
            #         data_lst += [video_dir]
        return data_lst
    
    def is_valid(self, video_dir: Path):
        if not (video_dir / "images").is_dir(): return False
        video_length = len(list((video_dir / "images").iterdir()))
        for guid in self.guids:
            guid_length = len(list((video_dir / guid).iterdir()))
            if guid_length != video_length:
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
                guid_transform = transforms.Compose([
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
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform
    
    def set_clip_idx(self, video_length, sample_frames):
        clip_length = min(video_length, (sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        clip_idxes = np.linspace(
            start_idx, start_idx + clip_length - 1, sample_frames, dtype=int
        ).tolist()
        return clip_idxes
    
    def get_mean_bbox(self, clip_idxes, bboxes):
        clip_bbox_lst = []
        for c_idx in clip_idxes:
            clip_bbox = bboxes[c_idx]
            clip_bbox_lst.append(np.array(clip_bbox))
        clip_bbox_mean = np.stack(clip_bbox_lst, axis=0).mean(0, keepdims=False)
        return clip_bbox_mean
        
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            ret_lst = []
            for img in images:
                if isinstance(img, list):
                    transformed_sub_images = [transform(sub_img) for sub_img in img]
                    sub_ret_tensor = torch.cat(transformed_sub_images, dim=0)  # (c*n, h, w)
                    ret_lst.append(sub_ret_tensor)
                else:
                    transformed_images = transform(img)
                    ret_lst.append(transformed_images)  # (c*1, h, w)
            ret_tensor = torch.stack(ret_lst, dim=0)  # (f, c*n, h, w)     
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
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

        smpl_param['poses'] = np.array(body_model_output.full_pose.detach()).astype(np.float32)
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

        part_name = str(video_dir).split('/')[-2][:-8]
        smc_name = str(video_dir).split('/')[-2][-7:]
        ref_view_index = int(str(video_dir).split('/')[-1][-4:])
        smc_path = f'data/DNA_Rendering/{part_name}/dna_rendering_part{part_name[-1]}_main/{smc_name}.smc'
        smc_reader = SMCReader(smc_path)
        annots_file_path = smc_path.replace('main', 'annotations').split('.')[0] + '_annots.smc'
        smc_annots_reader = SMCReader(annots_file_path)

        gender = smc_reader.actor_info['gender']

        # reference image index
        normal_dir = video_dir / "normal"
        if self.select_face:
            face_img_dir = video_dir / "face_images"
            face_img_lst = [img.name for img in face_img_dir.glob("*.png")]
            ref_img_name = random.choice(face_img_lst)
        else:
            ref_normal_name = random.choice([img.name for img in normal_dir.glob("*.png")])
        img_path_lst = sorted([img.name for img in normal_dir.glob("*.png")])
        ref_img_idx = int(ref_normal_name.split('.')[0])

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

        white_background = False
        ref_img[ref_msk == 0] = 255 if white_background else 0

        if self.image_ratio != 1.:
            H, W = int(ref_img.shape[0] * self.image_ratio), int(ref_img.shape[1] * self.image_ratio)
            ref_img = cv2.resize(ref_img, (W, H), interpolation=cv2.INTER_AREA)
            ref_K[:2] = ref_K[:2] * self.image_ratio
        ref_img_pil = Image.fromarray(ref_img)

        if self.nerf_rs_scale != 1.:
            H_nerf, W_nerf = int(ref_img.shape[0] * self.nerf_rs_scale), int(ref_img.shape[1] * self.nerf_rs_scale)
            ref_img_nerf = cv2.resize(ref_img, (W_nerf, H_nerf), interpolation=cv2.INTER_AREA)
            ref_K[:2] = ref_K[:2] * self.nerf_rs_scale
        else:
            ref_img_nerf = ref_img
        ref_img_nerf = ref_img_nerf.astype(np.float32) / 255.

        # prepare smpl at the reference view
        ref_smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=ref_img_idx)
        ref_smpl_param, ref_world_vertex, _ = self.prepare_smpl_data(ref_smpl_dict, gender)

        # tgt image index
        if self.crossview_num == 1:
            tgt_view_index_lst = [ref_view_index]
        else:
            view_index_lst = [i for i in range(smc_reader.Camera_5mp_info['num_device'])] 
            tgt_view_index_lst = np.random.choice(view_index_lst, size=self.crossview_num, replace=False)
        
        video_length = len(img_path_lst)
        clip_idxes = self.set_clip_idx(video_length, self.sample_frames//self.crossview_num)

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
        
        # # tgt frames indexes
        # video_length = len(list(img_dir.iterdir()))
        # clip_idxes = self.set_clip_idx(video_length)
        
        # # calculate bbox first
        # if self.bbox_crop:
        #     human_bbox_json_path = video_dir / "human_bbox.json"
        #     with open(human_bbox_json_path) as bbox_fp:
        #         human_bboxes = json.load(bbox_fp)
        #     mean_bbox = self.get_mean_bbox(clip_idxes, human_bboxes)
        #     resize_scale = random.uniform(*self.bbox_resize_ratio)
        #     ref_W, ref_H = ref_img_pil.size
        #     tgt_bbox = process_bbox(mean_bbox, ref_H, ref_W, resize_scale)


        # img_path_lst = sorted([img.name for img in img_dir.glob("*.png")])
        tgt_vidpil_lst = []
        tgt_guid_vidpil_lst = []
        
        tgt_img_nerf_lst = []
        tgt_smpl_param_dct, tgt_world_vertex_lst = {}, []
        tgt_ray_o_lst, tgt_ray_d_lst, tgt_near_lst, tgt_far_lst, tgt_mask_at_box_lst, tgt_bkgd_msk_lst = [], [], [], [], [], []

        # tgt frames
        # guid frames: [[frame0: n_type x pil], [frame1: n x pil], [frame2: n x pil], ...]
        for c_idx in clip_idxes:
            # tgt_img_path = img_dir / img_path_lst[c_idx]
            # tgt_img_pil = Image.open(tgt_img_path)
            # # tgt_img_pil = crop_bbox(tgt_img_pil, tgt_bbox)
            # tgt_vidpil_lst.append(tgt_img_pil)
            for tgt_view_index in tgt_view_index_lst:
                video_dir = Path(str(video_dir)[:-4] + str(tgt_view_index).zfill(4))
                # load tgt image and mask
                tgt_img_name = img_path_lst[c_idx]
                tgt_img_idx = int(tgt_img_name.split('.')[0])
                tgt_img = smc_reader.get_img('Camera_5mp', tgt_view_index, Frame_id=tgt_img_idx, Image_type='color')
                tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
                tgt_msk = smc_annots_reader.get_mask(tgt_view_index, Frame_id=tgt_img_idx)

                # Load reference K, R, T
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

                white_background = False
                tgt_img[tgt_msk == 0] = 255 if white_background else 0
                if self.image_ratio != 1.:
                    H, W = int(tgt_img.shape[0] * self.image_ratio), int(tgt_img.shape[1] * self.image_ratio)
                    tgt_img = cv2.resize(tgt_img, (W, H), interpolation=cv2.INTER_AREA)
                    tgt_msk = cv2.resize(tgt_msk, (W, H), interpolation=cv2.INTER_AREA)
                    tgt_K[:2] = tgt_K[:2] * self.image_ratio
                tgt_img_pil = Image.fromarray(tgt_img)
                tgt_img_nerf = tgt_img.astype(np.float32) / 255.
                tgt_vidpil_lst.append(tgt_img_pil)

                # prepare smpl at the target view
                tgt_smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=tgt_img_idx)
                tgt_smpl_param, tgt_world_vertex, tgt_world_bound = self.prepare_smpl_data(tgt_smpl_dict, gender)

                # Sample rays in target space world coordinate
                tgt_img_nerf, tgt_ray_o, tgt_ray_d, tgt_near, tgt_far, tgt_mask_at_box, tgt_bkgd_msk = sample_ray(
                        tgt_img_nerf, tgt_msk, tgt_K, tgt_R, tgt_T, tgt_world_bound, image_scaling=self.nerf_rs_scale)

                # tgt_smpl_param_lst.append(tgt_smpl_param)
                for key in tgt_smpl_param:
                    if key not in tgt_smpl_param_dct.keys():
                        tgt_smpl_param_dct[key] = [tgt_smpl_param[key]]
                    else:
                        tgt_smpl_param_dct[key].append(tgt_smpl_param[key])

                tgt_img_nerf_lst.append(tgt_img_nerf)
                tgt_world_vertex_lst.append(tgt_world_vertex)
                tgt_ray_o_lst.append(tgt_ray_o) 
                tgt_ray_d_lst.append(tgt_ray_d) 
                tgt_near_lst.append(tgt_near) 
                tgt_far_lst.append(tgt_far) 
                tgt_mask_at_box_lst.append(tgt_mask_at_box) 
                tgt_bkgd_msk_lst.append(tgt_bkgd_msk) 

                # tgt_img_name = tgt_img_path.name
                tgt_guid_pil_lst = []
                for guid in self.guids:
                    if guid != 'nerf':
                        guid_img_path = video_dir / guid / tgt_img_name
                        if guid == "semantic_map":
                            # mask_img_path = video_dir / "mask" / tgt_img_name            
                            # guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                            guid_img_pil = Image.open(guid_img_path).convert("RGB")
                        else:
                            guid_img_pil = Image.open(guid_img_path).convert("RGB")
                        if self.bbox_crop:
                            guid_img_pil = crop_bbox(guid_img_pil, tgt_bbox)
                        W, H = guid_img_pil.size
                        resize_ratio = min(tgt_img_pil.size) / min(W, H)
                        new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
                        guid_img_pil = guid_img_pil.resize((new_W, new_H))
                        tgt_guid_pil_lst.append(guid_img_pil)
                
                tgt_guid_vidpil_lst.append(tgt_guid_pil_lst)

        for key in tgt_smpl_param_dct:
            tgt_smpl_param_dct[key] = np.concatenate([tgt_smpl_param_dct[key]], axis=0)

        tgt_img_nerf_lst = np.concatenate([tgt_img_nerf_lst], axis=0)
        tgt_world_vertex_lst = np.concatenate([tgt_world_vertex_lst], axis=0)
        tgt_ray_o_lst = np.concatenate([tgt_ray_o_lst], axis=0)
        tgt_ray_d_lst = np.concatenate([tgt_ray_d_lst], axis=0)
        tgt_near_lst = np.concatenate([tgt_near_lst], axis=0)
        tgt_far_lst = np.concatenate([tgt_far_lst], axis=0)
        tgt_mask_at_box_lst = np.concatenate([tgt_mask_at_box_lst], axis=0)
        tgt_bkgd_msk_lst = np.concatenate([tgt_bkgd_msk_lst], axis=0)
        # ref_img_idx = img_path_lst.index(ref_img_name)
        # if self.bbox_crop:
        #     ref_bbox = process_bbox(human_bboxes[ref_img_idx], ref_H, ref_W, resize_scale)
        #     ref_img_pil = crop_bbox(ref_img_pil, ref_bbox)
        
        state = torch.get_rng_state()
        tgt_vid = self.augmentation(tgt_vidpil_lst, self.pixel_transform, state)
        tgt_guid_vid = self.augmentation(tgt_guid_vidpil_lst, self.guid_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.pixel_transform, state)
        clip_img = self.clip_image_processor(
            images=ref_img_pil, return_tensor="pt"
        ).pixel_values[0]
        
        # tgt_vidpil_lst[0].save('tgt_vidpil_lst.png')
        # imageio.imwrite('tgt_vidpil_lst.png', (np.array(tgt_vidpil_lst[0])[:,:,[2,1,0]]*255).astype(np.uint8))
        # imageio.imwrite('tgt_vid.png', (tgt_vid[0]*255)[[1,2,0],:,:].permute(1,2,0).cpu().numpy().astype(np.uint8))
        # imageio.imwrite('tgt_guid_vid.png', (tgt_guid_vid[0, :3]*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
        # imageio.imwrite('ref_img_vae.png', (ref_img_vae*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
        sample = dict(
            tgt_vid=tgt_vid,
            tgt_guid_vid=tgt_guid_vid,
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
            tgt_img_nerf=np.transpose(tgt_img_nerf_lst, (0, 3, 1, 2)),
            tgt_smpl_param=tgt_smpl_param_dct,
            tgt_world_vertex=tgt_world_vertex_lst, 
            tgt_ray_o=tgt_ray_o_lst,
            tgt_ray_d=tgt_ray_d_lst,
            tgt_near=tgt_near_lst,
            tgt_far=tgt_far_lst,
            tgt_mask_at_box=tgt_mask_at_box_lst,
            tgt_bkgd_msk=tgt_bkgd_msk_lst,  
        )
        
        return sample
