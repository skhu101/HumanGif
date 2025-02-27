import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from pathlib import Path
from collections import OrderedDict
import copy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision.utils import save_image
from datetime import timedelta
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image, ImageOps
import imageio
import json
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from einops import rearrange

from models.champ_model import HumanDiffModel
from models.guidance_encoder import GuidanceEncoder
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl

from datasets.video_dataset_RenderPeople_w_nerf import VideoDataset, sample_ray
# from datasets.data_utils import mask_to_bkgd
from utils.tb_tracker import TbTracker
from utils.util import seed_everything, delete_additional_ckpt, compute_snr
from utils.video_utils import save_videos_grid, save_videos_from_pil

from pipelines.pipeline_guidance2video import MultiGuidance2VideoPipeline

from models.mutual_self_attention import torch_dfs
from models.attention import TemporalBasicTransformerBlock
from models.attention import BasicTransformerBlock

from pytorch_msssim import ssim
import lpips
from models.recon_NeRF import NeRF_Renderer, to_cuda

import cv2
from einops import rearrange
# from data.DNA_Rendering.dna_rendering_sample_code.SMCReader import SMCReader
from smpl.smpl_numpy import SMPL
# from smplx.body_models import SMPLX

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def resize_pil(img_pil, img_size):
    # resize a PIL image and zero padding the short edge
    W, H = img_pil.size

    resize_ratio = img_size / min(W, H)
    new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
    img_pil = img_pil.resize((new_W, new_H))
    
    left = (new_W - img_size)/2
    top = (new_H - img_size)/2
    right = (new_W + img_size)/2
    bottom = (new_H + img_size)/2

    padding_border = (left, top, right, bottom)
    img_pil = img_pil.crop(padding_border)
    
    return img_pil

def padding_pil(img_pil, img_size):
    # resize a PIL image and zero padding the short edge
    W, H = img_pil.size
    resize_ratio = img_size / max(W, H)
    new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
    img_pil = img_pil.resize((new_W, new_H))
    
    left = (img_size - new_W) // 2
    right = img_size - new_W - left
    top = (img_size - new_H) // 2
    bottom = img_size - new_H - top
    
    padding_border = (left, top, right, bottom)
    img_pil = ImageOps.expand(img_pil, border=padding_border, fill=0)
    
    return img_pil

def concat_pil(img_pil_lst):
    # horizontally concat PIL images 
    # NOTE(ZSH): assume all images are of same size
    W, H = img_pil_lst[0].size
    num_img = len(img_pil_lst)
    new_width = num_img * W
    new_image = Image.new("RGB", (new_width, H), color=0)
    for img_idx, img in enumerate(img_pil_lst):
        new_image.paste(img, (W * img_idx, 0))  
    
    return new_image

def prepare_smpl_initial_data():
    smpl_model, big_pose_smpl_param_dct, big_pose_smpl_vertices_dct, big_pose_world_bound_dct = {}, {}, {}, {}
    for gender in ['female', 'male', 'neutral']:
        smpl_model[gender] = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')

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

        big_pose_smpl_vertices, _ = smpl_model[gender](big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
        big_pose_smpl_vertices = np.array(big_pose_smpl_vertices).astype(np.float32)
        big_pose_min_xyz = np.min(big_pose_smpl_vertices, axis=0)
        big_pose_max_xyz = np.max(big_pose_smpl_vertices, axis=0)
        big_pose_min_xyz -= 0.05
        big_pose_max_xyz += 0.05
        big_pose_min_xyz[2] -= 0.1
        big_pose_max_xyz[2] += 0.1
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)
        
        big_pose_smpl_param_dct[gender], big_pose_smpl_vertices_dct[gender], big_pose_world_bound_dct[gender] = big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound

    return smpl_model, big_pose_smpl_param_dct, big_pose_smpl_vertices_dct, big_pose_world_bound_dct

def get_mask(mask_path):
    msk = imageio.imread(mask_path)
    msk[msk!=0]=255
    return msk

def prepare_smpl_params(smpl_path, pose_index):
    params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
    params = {}
    params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
    params['poses'] = np.zeros((1,72)).astype(np.float32)
    params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
    params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
    params['R'] = np.eye(3).astype(np.float32)
    params['Th'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
    return params

def prepare_input(smpl_path, pose_index, gender, smpl_model):

    params = prepare_smpl_params(smpl_path, pose_index)
    xyz, _ = smpl_model[gender](params['poses'], params['shapes'].reshape(-1))
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

def load_image(img_path, smpl_model, white_bg=False, image_ratio=1.0, nerf_rs_scale=1.0, tgt_view=False):

    img_path = str(img_path)
    gender = 'neutral'

    # tgt image index
    view_index = int(str(img_path).split('/')[-3][-4:])
    img_idx = int(str(img_path).split('/')[-1].split('.')[0])

    # load image and mask     
    mask_path = img_path.replace('images', 'msk').replace('jpg', 'png')

    img = np.array(imageio.imread(img_path))
    img_nerf = img.copy()
    msk = np.array(get_mask(mask_path)) / 255.
    img[msk == 0] = 255 if white_bg else 0
    img_nerf[msk == 0] = 0

    # Load K, R, T
    camera_file = str(img_path)[:-8] + '/../../cameras.json'
    camera = json.load(open(camera_file))
    K = np.array(camera[f'camera{str(view_index).zfill(4)}']['K']).astype(np.float32)
    R = np.array(camera[f'camera{str(view_index).zfill(4)}']['R']).astype(np.float32)
    T = np.array(camera[f'camera{str(view_index).zfill(4)}']['T']).reshape(-1, 1).astype(np.float32)

    # prepare PIL and NeRF image 
    img_pil = Image.fromarray(img)
    img_nerf = img_nerf.astype(np.float32) / 255.

    # prepare smpl
    smpl_path = str(img_path)[:-8] + '/../../outputs_re_fitting/refit_smpl_2nd.npz'
    world_bound, world_vertex, smpl_param = prepare_input(smpl_path, img_idx, gender, smpl_model)

    data_nerf = {}
    if not tgt_view:
        data_nerf['gender'] = [gender]
        data_nerf['ref_img_nerf'] = np.transpose(img_nerf, (2,0,1))
        data_nerf['ref_smpl_param'] = smpl_param
        data_nerf['ref_world_vertex'] = world_vertex
        data_nerf['ref_K'] = K
        data_nerf['ref_R'] = R
        data_nerf['ref_T'] = T    
    else:
        img_nerf, ray_o, ray_d, near, far, mask_at_box, bkgd_msk = sample_ray(
        img_nerf, msk, K, R, T, world_bound, image_scaling=nerf_rs_scale, white_bg=False)

        data_nerf['tgt_img_nerf'] = np.transpose(img_nerf, (2,0,1))
        data_nerf['tgt_smpl_param'] = smpl_param
        data_nerf['tgt_world_vertex'] = world_vertex
        data_nerf['tgt_ray_o'] = ray_o
        data_nerf['tgt_ray_d'] = ray_d
        data_nerf['tgt_near'] = near
        data_nerf['tgt_far'] = far
        data_nerf['tgt_mask_at_box'] = mask_at_box
        data_nerf['tgt_bkgd_msk'] = bkgd_msk

    return img_pil, data_nerf

def validate(
    ref_img_path,
    guid_folder,
    guid_types,
    guid_start_idx,
    clip_length,
    width, height,
    pipe,
    generator,
    denoising_steps=20,
    guidance_scale=3.5,
    aug_type="Resize",
    smpl_model=None, 
    big_pose_smpl_param=None, 
    big_pose_smpl_vertices=None, 
    big_pose_world_bound=None,
    image_ratio=1.0,
    nerf_rs_scale=1.0,
    white_bg=True,
):
    # ref_img_pil = Image.open(ref_img_path)
    # ref_img_pil = load_image(ref_img_path)
    ref_img_pil, ref_data_nerf = load_image(ref_img_path, smpl_model, white_bg=white_bg)
    if aug_type =="Padding":
        ref_img_pil = padding_pil(ref_img_pil, height)
    elif aug_type =="Resize":
        ref_img_pil = resize_pil(ref_img_pil, height)
    guid_folder = Path(guid_folder)
    view_index_lst = [0, 10, 22, 30]
    guid_img_pil_lst = []
    for guid_type in guid_types:
        if guid_type != 'nerf':
            guid_img_lst = sorted((guid_folder / guid_type).iterdir())
            guid_img_clip_lst_ = guid_img_lst[guid_start_idx: guid_start_idx + clip_length]

            guid_img_clip_lst = []
            for guid_img_clip in guid_img_clip_lst_:
                for view_id in view_index_lst:
                    guid_img_clip_cam = Path(str(guid_img_clip).replace('camera0000', f'camera{str(view_id).zfill(4)}'))
                    guid_img_clip_lst.append(guid_img_clip_cam)
            single_guid_pil_lst = []
            for guid_img_path in guid_img_clip_lst:
                if guid_type == "semantic_map":
                    # mask_img_path = guid_folder / "mask" / guid_img_path.name
                    # guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                else:
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                if aug_type == "Padding":
                    guid_img_pil = padding_pil(guid_img_pil, height)
                elif aug_type == "Resize":
                    guid_img_pil = resize_pil(guid_img_pil, height)
                single_guid_pil_lst += [guid_img_pil]
            guid_img_pil_lst.append(single_guid_pil_lst)
        
        if guid_type == 'normal':
            # load tgt nerf data 
            tgt_data_nerf_dct = {}
            tgt_data_nerf_dct['tgt_smpl_param'] = {}
            for guid_img_path in guid_img_clip_lst:
                tgt_img_path = Path(str(guid_img_path).replace('normal', 'images').replace('.png', '.jpg'))
                tgt_img_pil, tgt_data_nerf = load_image(tgt_img_path, smpl_model, nerf_rs_scale=nerf_rs_scale, image_ratio=image_ratio, white_bg=white_bg, tgt_view=True)
                for key in tgt_data_nerf.keys():
                    if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                        if key not in tgt_data_nerf_dct.keys():
                            tgt_data_nerf_dct[key] = [tgt_data_nerf[key]]
                        else:
                            tgt_data_nerf_dct[key].append(tgt_data_nerf[key])
                    elif key in ['tgt_smpl_param']:
                        for smpl_key in tgt_data_nerf[key].keys():
                            if smpl_key not in tgt_data_nerf_dct[key].keys():
                                tgt_data_nerf_dct[key][smpl_key] = [tgt_data_nerf[key][smpl_key]]
                            else:
                                tgt_data_nerf_dct[key][smpl_key].append(tgt_data_nerf[key][smpl_key])

            for key in tgt_data_nerf_dct.keys():
                if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                    tgt_data_nerf_dct[key] = np.concatenate([tgt_data_nerf_dct[key]], axis=0)
                    # print(key, tgt_data_nerf_dct[key].shape)
                elif key in ['tgt_smpl_param']:
                    for smpl_key in tgt_data_nerf_dct[key].keys():
                        tgt_data_nerf_dct[key][smpl_key] = np.concatenate([tgt_data_nerf_dct[key][smpl_key]], axis=0)
                        # print(key, smpl_key, tgt_data_nerf_dct[key][smpl_key].shape)

    batch_data_nerf = {}
    gender = ref_data_nerf['gender'][0]
    batch_data_nerf['big_pose_smpl_param'] = big_pose_smpl_param[gender]
    batch_data_nerf['big_pose_world_vertex'] = big_pose_smpl_vertices[gender]
    batch_data_nerf['big_pose_world_bound'] = big_pose_world_bound[gender]
    batch_data_nerf.update(ref_data_nerf)
    batch_data_nerf.update(tgt_data_nerf_dct)
    batch_data_nerf['gender'] = batch_data_nerf['gender'] * clip_length * len(view_index_lst)

    batch_data_nerf = to_cuda(batch_data_nerf, torch.device('cuda', torch.cuda.current_device()))

    val_videos, rgb_nerf_pred = pipe(
        ref_img_pil,
        guid_img_pil_lst,
        guid_types,
        width,
        height,
        clip_length * len(view_index_lst),
        denoising_steps,
        guidance_scale,
        crossview_num=len(view_index_lst),
        batch_data_nerf=batch_data_nerf,
        generator=generator,
    )#.videos
    rgb_nerf_img_pil_lst = []
    for i in range(rgb_nerf_pred.shape[1]):
        rgb_nerf_img_pil = Image.fromarray((255*rgb_nerf_pred[0, i]).permute(1,2,0).cpu().numpy().astype(np.uint8))
        if aug_type =="Padding":
            rgb_nerf_img_pil = padding_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        elif aug_type =="Resize":
            rgb_nerf_img_pil = resize_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        rgb_nerf_img_pil_lst += [rgb_nerf_img_pil]
    guid_img_pil_lst.append(rgb_nerf_img_pil_lst)

    return val_videos, ref_img_pil, guid_img_pil_lst


def log_validation(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    accelerator,
    width,
    height,
    seed=42,
    # dtype=torch.float16,
    dtype=torch.float32,
):
    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    NeRF_renderer = unwrap_model.NeRF_renderer
    guid_types = unwrap_model.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(unwrap_model, f"guidance_encoder_{g}") for g in guid_types
    }
    for _, module in guidance_encoder_group.items():
        module.to(dtype=dtype)

    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    
    smpl_model, big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound = prepare_smpl_initial_data()

    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=dtype)
    pipeline = MultiGuidance2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        NeRF_renderer=NeRF_renderer,
        **guidance_encoder_group,
        scheduler=scheduler,
        nerf_cond_type=cfg.NeRF.nerf_cond_type,
        use_diff_img_loss=cfg.use_diff_img_loss,
    )
    pipeline = pipeline.to(accelerator.device)
    
    # ref_img_lst_ = cfg.validation.ref_images.copy()
    # guid_folder_lst_ = cfg.validation.guidance_folders.copy()
    # guid_idxes_ = cfg.validation.guidance_indexes.copy()
    clip_length = cfg.validation.clip_length
    
    ref_img_lst, guid_folder_lst, guid_idxes = [], [], []
    test_folder = cfg.validation.guidance_folders[0] + '/../../../test'
    test_subject_names = os.listdir(test_folder)[:5]
    for test_subject in test_subject_names:
        ref_img_lst.append(test_folder + '/' + test_subject + '/camera0000/images/0000.jpg')
        guid_folder_lst.append(test_folder + '/' + test_subject + '/camera0000')
        guid_idxes.append(cfg.validation.guidance_indexes[0])

    val_results = []
    for val_idx, (ref_img_path, guid_folder, guid_start_idx) in enumerate(
        zip(ref_img_lst, guid_folder_lst, guid_idxes)):
        
        video_tensor, ref_img_pil, guid_img_pil_lst = validate(
            ref_img_path=ref_img_path,
            guid_folder=guid_folder,
            guid_types=guid_types,
            guid_start_idx=guid_start_idx,
            clip_length=clip_length,
            width=width,
            height=height,
            pipe=pipeline,
            generator=generator,
            aug_type=cfg.data.aug_type,
            smpl_model=smpl_model, 
            big_pose_smpl_param=big_pose_smpl_param, 
            big_pose_smpl_vertices=big_pose_smpl_vertices, 
            big_pose_world_bound=big_pose_world_bound,
            image_ratio=cfg.data.image_ratio,
            nerf_rs_scale=cfg.data.nerf_rs_scale,
            white_bg=cfg.data.white_bg,
        )
    
        video_tensor = video_tensor[0, ...].permute(1, 2, 3, 0).cpu().numpy()
        W, H = ref_img_pil.size
        
        video_pil_lst = []
        for frame_idx, image_tensor in enumerate(video_tensor):
            result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
            result_img_pil = result_img_pil.resize((W, H))
            frame_guid_pil_lst = [g[frame_idx].resize((W, H)) for g in guid_img_pil_lst]
            # result_pil_lst = [result_img_pil, ref_img_pil, *frame_guid_pil_lst]
            result_pil_lst = [ref_img_pil, *frame_guid_pil_lst, result_img_pil]
            concated_pil = concat_pil(result_pil_lst)
            
            video_pil_lst.append(concated_pil)
            
        val_results.append({"name": f"val_{val_idx}", "video": video_pil_lst})
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del tmp_denoising_unet
    del pipeline
    torch.cuda.empty_cache()

    return val_results


def validate_nv(
    ref_img_path,
    guid_folder,
    guid_types,
    guid_start_idx,
    clip_length,
    width, height,
    pipe,
    generator,
    denoising_steps=20,
    guidance_scale=3.5,
    aug_type="Resize",
    smpl_model=None, 
    big_pose_smpl_param=None, 
    big_pose_smpl_vertices=None, 
    big_pose_world_bound=None,
    image_ratio=1.0,
    nerf_rs_scale=1.0,
    white_bg=True,
):
    # ref_img_pil = Image.open(ref_img_path)
    # ref_img_pil = load_image(ref_img_path)
    ref_img_pil, ref_data_nerf = load_image(ref_img_path, smpl_model, white_bg=white_bg)
    if aug_type =="Padding":
        ref_img_pil = padding_pil(ref_img_pil, height)
    elif aug_type =="Resize":
        ref_img_pil = resize_pil(ref_img_pil, height)
    guid_folder = Path(guid_folder)
    view_index_lst = [0, 10, 22, 30]
    guid_img_pil_lst = []
    for guid_type in guid_types:
        if guid_type != 'nerf':
            guid_img_lst = sorted((guid_folder / guid_type).iterdir())
            guid_img_clip_lst_ = guid_img_lst[guid_start_idx: guid_start_idx + clip_length * 3 : 3]

            guid_img_clip_lst = []
            for guid_img_clip in guid_img_clip_lst_:
                for view_id in view_index_lst:
                    guid_img_clip_cam = Path(str(guid_img_clip).replace('camera0000', f'camera{str(view_id).zfill(4)}'))
                    guid_img_clip_lst.append(guid_img_clip_cam)
            single_guid_pil_lst = []
            for guid_img_path in guid_img_clip_lst:
                if guid_type == "semantic_map":
                    # mask_img_path = guid_folder / "mask" / guid_img_path.name
                    # guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                else:
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                if aug_type == "Padding":
                    guid_img_pil = padding_pil(guid_img_pil, height)
                elif aug_type == "Resize":
                    guid_img_pil = resize_pil(guid_img_pil, height)
                single_guid_pil_lst += [guid_img_pil]
            guid_img_pil_lst.append(single_guid_pil_lst)
        
        if guid_type == 'normal':
            # load tgt nerf data 
            tgt_data_nerf_dct = {}
            tgt_data_nerf_dct['tgt_smpl_param'] = {}
            for guid_img_path in guid_img_clip_lst:
                tgt_img_path = Path(str(guid_img_path).replace('normal', 'images').replace('.png', '.jpg'))
                tgt_img_pil, tgt_data_nerf = load_image(tgt_img_path, smpl_model, nerf_rs_scale=nerf_rs_scale, image_ratio=image_ratio, white_bg=white_bg, tgt_view=True)
                for key in tgt_data_nerf.keys():
                    if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                        if key not in tgt_data_nerf_dct.keys():
                            tgt_data_nerf_dct[key] = [tgt_data_nerf[key]]
                        else:
                            tgt_data_nerf_dct[key].append(tgt_data_nerf[key])
                    elif key in ['tgt_smpl_param']:
                        for smpl_key in tgt_data_nerf[key].keys():
                            if smpl_key not in tgt_data_nerf_dct[key].keys():
                                tgt_data_nerf_dct[key][smpl_key] = [tgt_data_nerf[key][smpl_key]]
                            else:
                                tgt_data_nerf_dct[key][smpl_key].append(tgt_data_nerf[key][smpl_key])

            for key in tgt_data_nerf_dct.keys():
                if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                    tgt_data_nerf_dct[key] = np.concatenate([tgt_data_nerf_dct[key]], axis=0)
                    # print(key, tgt_data_nerf_dct[key].shape)
                elif key in ['tgt_smpl_param']:
                    for smpl_key in tgt_data_nerf_dct[key].keys():
                        tgt_data_nerf_dct[key][smpl_key] = np.concatenate([tgt_data_nerf_dct[key][smpl_key]], axis=0)
                        # print(key, smpl_key, tgt_data_nerf_dct[key][smpl_key].shape)

    batch_data_nerf = {}
    gender = ref_data_nerf['gender'][0]
    batch_data_nerf['big_pose_smpl_param'] = big_pose_smpl_param[gender]
    batch_data_nerf['big_pose_world_vertex'] = big_pose_smpl_vertices[gender]
    batch_data_nerf['big_pose_world_bound'] = big_pose_world_bound[gender]
    batch_data_nerf.update(ref_data_nerf)
    batch_data_nerf.update(tgt_data_nerf_dct)
    batch_data_nerf['gender'] = batch_data_nerf['gender'] * clip_length * len(view_index_lst)

    batch_data_nerf = to_cuda(batch_data_nerf, torch.device('cuda', torch.cuda.current_device()))

    val_videos, rgb_nerf_pred = pipe(
        ref_img_pil,
        guid_img_pil_lst,
        guid_types,
        width,
        height,
        clip_length * len(view_index_lst),
        denoising_steps,
        guidance_scale,
        crossview_num=len(view_index_lst),
        batch_data_nerf=batch_data_nerf,
        generator=generator,
    )#.videos
    rgb_nerf_img_pil_lst = []
    for i in range(rgb_nerf_pred.shape[1]):
        rgb_nerf_img_pil = Image.fromarray((255*rgb_nerf_pred[0, i]).permute(1,2,0).cpu().numpy().astype(np.uint8))
        if aug_type =="Padding":
            rgb_nerf_img_pil = padding_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        elif aug_type =="Resize":
            rgb_nerf_img_pil = resize_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        rgb_nerf_img_pil_lst += [rgb_nerf_img_pil]
    guid_img_pil_lst.append(rgb_nerf_img_pil_lst)

    tgt_mask_at_box = tgt_data_nerf_dct['tgt_mask_at_box'].reshape(tgt_data_nerf_dct['tgt_mask_at_box'].shape[0], tgt_img_pil.size[1], tgt_img_pil.size[0])
    val_videos[np.repeat(tgt_mask_at_box[None, None], 3, axis=1) == 0] = 0 if not white_bg else 1

    return val_videos, ref_img_pil, guid_img_pil_lst


def log_validation_nv(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    accelerator,
    width,
    height,
    seed=42,
    # dtype=torch.float16,
    dtype=torch.float32,
):
    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    NeRF_renderer = unwrap_model.NeRF_renderer
    guid_types = unwrap_model.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(unwrap_model, f"guidance_encoder_{g}") for g in guid_types
    }
    for _, module in guidance_encoder_group.items():
        module.to(dtype=dtype)

    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    
    smpl_model, big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound = prepare_smpl_initial_data()

    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=dtype)
    pipeline = MultiGuidance2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        NeRF_renderer=NeRF_renderer,
        **guidance_encoder_group,
        scheduler=scheduler,
        nerf_cond_type=cfg.NeRF.nerf_cond_type,
        use_diff_img_loss=cfg.use_diff_img_loss,
        test_nerf=cfg.test_nerf,
    )
    pipeline = pipeline.to(accelerator.device)
    
    # ref_img_lst_ = cfg.validation.ref_images.copy()
    # guid_folder_lst_ = cfg.validation.guidance_folders.copy()
    # guid_idxes_ = cfg.validation.guidance_indexes.copy()
    clip_length = cfg.validation.clip_length
    
    view_index_lst = [0, 10, 22, 30]
    ref_img_lst, guid_folder_lst, guid_idxes = [], [], []
    test_folder = cfg.validation.guidance_folders[0] + '/../../../test'
    test_subject_names = os.listdir(test_folder)[0:]
    for test_subject in test_subject_names:
        ref_img_lst.append(test_folder + '/' + test_subject + '/camera0000/images/0003.jpg')
        guid_folder_lst.append(test_folder + '/' + test_subject + '/camera0000')
        guid_idxes.append(cfg.validation.guidance_indexes[0])

    save_dir = f'{cfg.output_dir}/{cfg.exp_name}'
    os.makedirs(os.path.join(save_dir, 'pred_nv'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'nerf_nv'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'gt_nv'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'nerf_gt_nv'), exist_ok=True)

    val_results = []
    for val_idx, (ref_img_path, guid_folder, guid_start_idx) in enumerate(
        zip(ref_img_lst, guid_folder_lst, guid_idxes)):
        
        video_tensor, ref_img_pil, guid_img_pil_lst = validate_nv(
            ref_img_path=ref_img_path,
            guid_folder=guid_folder,
            guid_types=guid_types,
            guid_start_idx=guid_start_idx,
            clip_length=clip_length,
            width=width,
            height=height,
            pipe=pipeline,
            generator=generator,
            aug_type=cfg.data.aug_type,
            smpl_model=smpl_model, 
            big_pose_smpl_param=big_pose_smpl_param, 
            big_pose_smpl_vertices=big_pose_smpl_vertices, 
            big_pose_world_bound=big_pose_world_bound,
            image_ratio=cfg.data.image_ratio,
            nerf_rs_scale=cfg.data.nerf_rs_scale,
            white_bg=cfg.data.white_bg,
        )
    
        video_tensor = video_tensor[0, ...].permute(1, 2, 3, 0).cpu().numpy()
        W, H = ref_img_pil.size
        
        video_pil_lst = []
        for frame_idx, image_tensor in enumerate(video_tensor):
            result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
            result_img_pil = result_img_pil.resize((W, H))
            frame_guid_pil_lst = [g[frame_idx].resize((W, H)) for g in guid_img_pil_lst]
            # result_pil_lst = [result_img_pil, ref_img_pil, *frame_guid_pil_lst]
            result_pil_lst = [ref_img_pil, *frame_guid_pil_lst, result_img_pil]
            concated_pil = concat_pil(result_pil_lst)
            video_pil_lst.append(concated_pil)

            view_idx = view_index_lst[frame_idx % len(view_index_lst)]
            image_name = ref_img_path.split('/')[-4] + '_' + str(view_idx).zfill(4) + '_' + str(guid_start_idx + frame_idx//len(view_index_lst) * 3).zfill(4) + '.jpg'
            tgt_img_path = ref_img_path[:-8].replace(ref_img_path.split('/')[-3], f'camera{str(view_idx).zfill(4)}') + str(guid_start_idx + frame_idx//len(view_index_lst) * 3).zfill(4) + '.jpg'
            tgt_img_pil, _ = load_image(tgt_img_path, smpl_model, white_bg=True)
            if cfg.data.aug_type =="Padding":
                tgt_img_pil = padding_pil(tgt_img_pil, height)
            elif cfg.data.aug_type =="Resize":
                tgt_img_pil = resize_pil(tgt_img_pil, height)

            tgt_img_nerf_pil, _ = load_image(tgt_img_path, smpl_model, white_bg=False)
            if cfg.data.aug_type =="Padding":
                tgt_img_nerf_pil = padding_pil(tgt_img_nerf_pil, height)
            elif cfg.data.aug_type =="Resize":
                tgt_img_nerf_pil = resize_pil(tgt_img_nerf_pil, height)

            nerf_img_pil = frame_guid_pil_lst[-1]
            nerf_img_pil.save(f"{save_dir}/nerf_nv/{image_name}")
            tgt_img_nerf_pil.save(f"{save_dir}/nerf_gt_nv/{image_name}")
            result_img_pil.save(f"{save_dir}/pred_nv/{image_name}")
            tgt_img_pil.save(f"{save_dir}/gt_nv/{image_name}")
            
        val_results.append({"name": f"val_{val_idx}", "video": video_pil_lst})
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del tmp_denoising_unet
    del pipeline
    torch.cuda.empty_cache()

    return val_results

def validate_np(
    ref_img_path,
    guid_folder,
    guid_types,
    guid_start_idx,
    clip_length,
    width, height,
    pipe,
    generator,
    denoising_steps=20,
    guidance_scale=3.5,
    aug_type="Resize",
    smpl_model=None, 
    big_pose_smpl_param=None, 
    big_pose_smpl_vertices=None, 
    big_pose_world_bound=None,
    image_ratio=1.0,
    nerf_rs_scale=1.0,
    white_bg=True,
):
    # ref_img_pil = Image.open(ref_img_path)
    # ref_img_pil = load_image(ref_img_path)
    ref_img_pil, ref_data_nerf = load_image(ref_img_path, smpl_model, white_bg=white_bg)
    if aug_type =="Padding":
        ref_img_pil = padding_pil(ref_img_pil, height)
    elif aug_type =="Resize":
        ref_img_pil = resize_pil(ref_img_pil, height)
    guid_folder = Path(guid_folder)
    view_index_lst = [0]
    guid_img_pil_lst = []
    for guid_type in guid_types:
        if guid_type != 'nerf':
            guid_img_lst = sorted((guid_folder / guid_type).iterdir())
            guid_img_clip_lst_ = guid_img_lst[guid_start_idx: guid_start_idx + clip_length] + guid_img_lst[1:4]

            guid_img_clip_lst = []
            for guid_img_clip in guid_img_clip_lst_:
                for view_id in view_index_lst:
                    guid_img_clip_cam = Path(str(guid_img_clip).replace('camera0000', f'camera{str(view_id).zfill(4)}'))
                    guid_img_clip_lst.append(guid_img_clip_cam)
            single_guid_pil_lst = []
            for guid_img_path in guid_img_clip_lst:
                if guid_type == "semantic_map":
                    # mask_img_path = guid_folder / "mask" / guid_img_path.name
                    # guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                else:
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                if aug_type == "Padding":
                    guid_img_pil = padding_pil(guid_img_pil, height)
                elif aug_type == "Resize":
                    guid_img_pil = resize_pil(guid_img_pil, height)
                single_guid_pil_lst += [guid_img_pil]
            guid_img_pil_lst.append(single_guid_pil_lst)
        
        if guid_type == 'normal':
            # load tgt nerf data 
            tgt_data_nerf_dct = {}
            tgt_data_nerf_dct['tgt_smpl_param'] = {}
            for guid_img_path in guid_img_clip_lst:
                tgt_img_path = Path(str(guid_img_path).replace('normal', 'images').replace('.png', '.jpg'))
                tgt_img_pil, tgt_data_nerf = load_image(tgt_img_path, smpl_model, nerf_rs_scale=nerf_rs_scale, image_ratio=image_ratio, white_bg=white_bg, tgt_view=True)
                for key in tgt_data_nerf.keys():
                    if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                        if key not in tgt_data_nerf_dct.keys():
                            tgt_data_nerf_dct[key] = [tgt_data_nerf[key]]
                        else:
                            tgt_data_nerf_dct[key].append(tgt_data_nerf[key])
                    elif key in ['tgt_smpl_param']:
                        for smpl_key in tgt_data_nerf[key].keys():
                            if smpl_key not in tgt_data_nerf_dct[key].keys():
                                tgt_data_nerf_dct[key][smpl_key] = [tgt_data_nerf[key][smpl_key]]
                            else:
                                tgt_data_nerf_dct[key][smpl_key].append(tgt_data_nerf[key][smpl_key])

            for key in tgt_data_nerf_dct.keys():
                if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                    tgt_data_nerf_dct[key] = np.concatenate([tgt_data_nerf_dct[key]], axis=0)
                    # print(key, tgt_data_nerf_dct[key].shape)
                elif key in ['tgt_smpl_param']:
                    for smpl_key in tgt_data_nerf_dct[key].keys():
                        tgt_data_nerf_dct[key][smpl_key] = np.concatenate([tgt_data_nerf_dct[key][smpl_key]], axis=0)
                        # print(key, smpl_key, tgt_data_nerf_dct[key][smpl_key].shape)

    batch_data_nerf = {}
    gender = ref_data_nerf['gender'][0]
    batch_data_nerf['big_pose_smpl_param'] = big_pose_smpl_param[gender]
    batch_data_nerf['big_pose_world_vertex'] = big_pose_smpl_vertices[gender]
    batch_data_nerf['big_pose_world_bound'] = big_pose_world_bound[gender]
    batch_data_nerf.update(ref_data_nerf)
    batch_data_nerf.update(tgt_data_nerf_dct)
    batch_data_nerf['gender'] = batch_data_nerf['gender'] * clip_length * len(view_index_lst)

    batch_data_nerf = to_cuda(batch_data_nerf, torch.device('cuda', torch.cuda.current_device()))

    val_videos, rgb_nerf_pred = pipe(
        ref_img_pil,
        guid_img_pil_lst,
        guid_types,
        width,
        height,
        clip_length * len(view_index_lst),
        denoising_steps,
        guidance_scale,
        crossview_num=len(view_index_lst),
        batch_data_nerf=batch_data_nerf,
        generator=generator,
    )#.videos
    rgb_nerf_img_pil_lst = []
    for i in range(rgb_nerf_pred.shape[1]):
        rgb_nerf_img_pil = Image.fromarray((255*rgb_nerf_pred[0, i]).permute(1,2,0).cpu().numpy().astype(np.uint8))
        if aug_type =="Padding":
            rgb_nerf_img_pil = padding_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        elif aug_type =="Resize":
            rgb_nerf_img_pil = resize_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        rgb_nerf_img_pil_lst += [rgb_nerf_img_pil]
    guid_img_pil_lst.append(rgb_nerf_img_pil_lst)

    tgt_mask_at_box = tgt_data_nerf_dct['tgt_mask_at_box'].reshape(tgt_data_nerf_dct['tgt_mask_at_box'].shape[0], tgt_img_pil.size[1], tgt_img_pil.size[0])
    val_videos[np.repeat(tgt_mask_at_box[None, None], 3, axis=1) == 0] = 0 if not white_bg else 1

    return val_videos, ref_img_pil, guid_img_pil_lst


def log_validation_np(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    accelerator,
    width,
    height,
    seed=42,
    # dtype=torch.float16,
    dtype=torch.float32,
):
    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    NeRF_renderer = unwrap_model.NeRF_renderer
    guid_types = unwrap_model.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(unwrap_model, f"guidance_encoder_{g}") for g in guid_types
    }
    for _, module in guidance_encoder_group.items():
        module.to(dtype=dtype)

    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    
    smpl_model, big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound = prepare_smpl_initial_data()

    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=dtype)
    pipeline = MultiGuidance2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        NeRF_renderer=NeRF_renderer,
        **guidance_encoder_group,
        scheduler=scheduler,
        nerf_cond_type=cfg.NeRF.nerf_cond_type,
        use_diff_img_loss=cfg.use_diff_img_loss,
        test_nerf=cfg.test_nerf,
    )
    pipeline = pipeline.to(accelerator.device)
    
    # ref_img_lst_ = cfg.validation.ref_images.copy()
    # guid_folder_lst_ = cfg.validation.guidance_folders.copy()
    # guid_idxes_ = cfg.validation.guidance_indexes.copy()
    clip_length = 24 #cfg.validation.clip_length
    
    ref_img_lst, guid_folder_lst, guid_idxes = [], [], []
    test_folder = cfg.validation.guidance_folders[0] + '/../../../test'
    test_subject_names = os.listdir(test_folder)[0:]
    for test_subject in test_subject_names:
        ref_img_lst.append(test_folder + '/' + test_subject + '/camera0000/images/0000.jpg')
        guid_folder_lst.append(test_folder + '/' + test_subject + '/camera0000')
        guid_idxes.append(cfg.validation.guidance_indexes[0])

    view_index_lst =[0]
    save_dir = f'{cfg.output_dir}/{cfg.exp_name}'
    os.makedirs(os.path.join(save_dir, 'nerf_np'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pred_np'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'gt_np'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'nerf_gt_np'), exist_ok=True)

    val_results = []
    for val_idx, (ref_img_path, guid_folder, guid_start_idx) in enumerate(
        zip(ref_img_lst, guid_folder_lst, guid_idxes)):
        
        video_tensor, ref_img_pil, guid_img_pil_lst = validate_np(
            ref_img_path=ref_img_path,
            guid_folder=guid_folder,
            guid_types=guid_types,
            guid_start_idx=guid_start_idx,
            clip_length=clip_length,
            width=width,
            height=height,
            pipe=pipeline,
            generator=generator,
            aug_type=cfg.data.aug_type,
            smpl_model=smpl_model, 
            big_pose_smpl_param=big_pose_smpl_param, 
            big_pose_smpl_vertices=big_pose_smpl_vertices, 
            big_pose_world_bound=big_pose_world_bound,
            image_ratio=cfg.data.image_ratio,
            nerf_rs_scale=cfg.data.nerf_rs_scale,
            white_bg=cfg.data.white_bg,
        )
    
        video_tensor = video_tensor[0, ...].permute(1, 2, 3, 0).cpu().numpy()
        W, H = ref_img_pil.size
        
        video_pil_lst = []
        for frame_idx, image_tensor in enumerate(video_tensor):
            result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
            result_img_pil = result_img_pil.resize((W, H))
            frame_guid_pil_lst = [g[frame_idx].resize((W, H)) for g in guid_img_pil_lst]
            # result_pil_lst = [result_img_pil, ref_img_pil, *frame_guid_pil_lst]
            result_pil_lst = [ref_img_pil, *frame_guid_pil_lst, result_img_pil]
            concated_pil = concat_pil(result_pil_lst)
            video_pil_lst.append(concated_pil)
            
            if frame_idx <= 20:
                view_idx = view_index_lst[0]
                image_name = ref_img_path.split('/')[-4] + '_camera' + str(view_idx).zfill(4) + '_' + str(guid_start_idx + frame_idx).zfill(4) + '.jpg'
                tgt_img_path = ref_img_path[:-8].replace(ref_img_path.split('/')[-3], f'camera{str(view_idx).zfill(4)}') + str(guid_start_idx + frame_idx).zfill(4) + '.jpg'
                tgt_img_pil, _ = load_image(tgt_img_path, smpl_model, white_bg=True)
                if cfg.data.aug_type =="Padding":
                    tgt_img_pil = padding_pil(tgt_img_pil, height)
                elif cfg.data.aug_type =="Resize":
                    tgt_img_pil = resize_pil(tgt_img_pil, height)

                tgt_img_nerf_pil, _ = load_image(tgt_img_path, smpl_model, white_bg=False)
                if cfg.data.aug_type =="Padding":
                    tgt_img_nerf_pil = padding_pil(tgt_img_nerf_pil, height)
                elif cfg.data.aug_type =="Resize":
                    tgt_img_nerf_pil = resize_pil(tgt_img_nerf_pil, height)

                nerf_img_pil = frame_guid_pil_lst[-1]
                nerf_img_pil.save(f"{save_dir}/nerf_np/{image_name}")
                tgt_img_nerf_pil.save(f"{save_dir}/nerf_gt_np/{image_name}")
                result_img_pil.save(f"{save_dir}/pred_np/{image_name}")
                tgt_img_pil.save(f"{save_dir}/gt_np/{image_name}")


        val_results.append({"name": f"val_{val_idx}", "video": video_pil_lst})
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del tmp_denoising_unet
    del pipeline
    torch.cuda.empty_cache()

    return val_results


def validate_np_video(
    ref_img_path,
    guid_folder,
    guid_types,
    guid_start_idx,
    clip_length,
    width, height,
    pipe,
    generator,
    denoising_steps=20,
    guidance_scale=3.5,
    aug_type="Resize",
    smpl_model=None, 
    big_pose_smpl_param=None, 
    big_pose_smpl_vertices=None, 
    big_pose_world_bound=None,
    image_ratio=1.0,
    nerf_rs_scale=1.0,
    white_bg=True,
):
    # ref_img_pil = Image.open(ref_img_path)
    # ref_img_pil = load_image(ref_img_path)
    ref_img_pil, ref_data_nerf = load_image(ref_img_path, smpl_model, white_bg=white_bg)
    if aug_type =="Padding":
        ref_img_pil = padding_pil(ref_img_pil, height)
    elif aug_type =="Resize":
        ref_img_pil = resize_pil(ref_img_pil, height)
    guid_folder = Path(guid_folder)
    view_index_lst = [0, 10, 22, 30]
    guid_img_pil_lst = []
    for guid_type in guid_types:
        if guid_type != 'nerf':
            guid_img_lst = sorted((guid_folder / guid_type).iterdir())
            guid_img_clip_lst = guid_img_lst[guid_start_idx: guid_start_idx + clip_length] + guid_img_lst[1:4]

            # guid_img_clip_lst = []
            # for guid_img_clip in guid_img_clip_lst_:
            #     for view_id in view_index_lst:
            #         guid_img_clip_cam = Path(str(guid_img_clip).replace('camera0000', f'camera{str(view_id).zfill(4)}'))
            #         guid_img_clip_lst.append(guid_img_clip_cam)
            single_guid_pil_lst = []
            for guid_img_path in guid_img_clip_lst:
                if guid_type == "semantic_map":
                    # mask_img_path = guid_folder / "mask" / guid_img_path.name
                    # guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                else:
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                if aug_type == "Padding":
                    guid_img_pil = padding_pil(guid_img_pil, height)
                elif aug_type == "Resize":
                    guid_img_pil = resize_pil(guid_img_pil, height)
                single_guid_pil_lst += [guid_img_pil]
            guid_img_pil_lst.append(single_guid_pil_lst)
        
        if guid_type == 'normal':
            # load tgt nerf data 
            tgt_data_nerf_dct = {}
            tgt_data_nerf_dct['tgt_smpl_param'] = {}
            for guid_img_path in guid_img_clip_lst:
                tgt_img_path = Path(str(guid_img_path).replace('normal', 'images').replace('.png', '.jpg'))
                tgt_img_pil, tgt_data_nerf = load_image(tgt_img_path, smpl_model, nerf_rs_scale=nerf_rs_scale, image_ratio=image_ratio, white_bg=white_bg, tgt_view=True)
                for key in tgt_data_nerf.keys():
                    if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                        if key not in tgt_data_nerf_dct.keys():
                            tgt_data_nerf_dct[key] = [tgt_data_nerf[key]]
                        else:
                            tgt_data_nerf_dct[key].append(tgt_data_nerf[key])
                    elif key in ['tgt_smpl_param']:
                        for smpl_key in tgt_data_nerf[key].keys():
                            if smpl_key not in tgt_data_nerf_dct[key].keys():
                                tgt_data_nerf_dct[key][smpl_key] = [tgt_data_nerf[key][smpl_key]]
                            else:
                                tgt_data_nerf_dct[key][smpl_key].append(tgt_data_nerf[key][smpl_key])

            for key in tgt_data_nerf_dct.keys():
                if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_bkgd_msk']:
                    tgt_data_nerf_dct[key] = np.concatenate([tgt_data_nerf_dct[key]], axis=0)
                    # print(key, tgt_data_nerf_dct[key].shape)
                elif key in ['tgt_smpl_param']:
                    for smpl_key in tgt_data_nerf_dct[key].keys():
                        tgt_data_nerf_dct[key][smpl_key] = np.concatenate([tgt_data_nerf_dct[key][smpl_key]], axis=0)
                        # print(key, smpl_key, tgt_data_nerf_dct[key][smpl_key].shape)

    batch_data_nerf = {}
    gender = ref_data_nerf['gender'][0]
    batch_data_nerf['big_pose_smpl_param'] = big_pose_smpl_param[gender]
    batch_data_nerf['big_pose_world_vertex'] = big_pose_smpl_vertices[gender]
    batch_data_nerf['big_pose_world_bound'] = big_pose_world_bound[gender]
    batch_data_nerf.update(ref_data_nerf)
    batch_data_nerf.update(tgt_data_nerf_dct)
    batch_data_nerf['gender'] = batch_data_nerf['gender'] * clip_length * len(view_index_lst)

    batch_data_nerf = to_cuda(batch_data_nerf, torch.device('cuda', torch.cuda.current_device()))

    val_videos, rgb_nerf_pred = pipe(
        ref_img_pil,
        guid_img_pil_lst,
        guid_types,
        width,
        height,
        clip_length,
        denoising_steps,
        guidance_scale,
        crossview_num=1,
        batch_data_nerf=batch_data_nerf,
        generator=generator,
    )#.videos
    rgb_nerf_img_pil_lst = []
    for i in range(rgb_nerf_pred.shape[1]):
        rgb_nerf_img_pil = Image.fromarray((255*rgb_nerf_pred[0, i]).permute(1,2,0).cpu().numpy().astype(np.uint8))
        if aug_type =="Padding":
            rgb_nerf_img_pil = padding_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        elif aug_type =="Resize":
            rgb_nerf_img_pil = resize_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1]))
        rgb_nerf_img_pil_lst += [rgb_nerf_img_pil]
    guid_img_pil_lst.append(rgb_nerf_img_pil_lst)

    return val_videos, ref_img_pil, guid_img_pil_lst

def log_validation_np_video(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    accelerator,
    width,
    height,
    seed=42,
    # dtype=torch.float16,
    dtype=torch.float32,
):
    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    NeRF_renderer = unwrap_model.NeRF_renderer
    guid_types = unwrap_model.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(unwrap_model, f"guidance_encoder_{g}") for g in guid_types
    }
    for _, module in guidance_encoder_group.items():
        module.to(dtype=dtype)

    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    
    smpl_model, big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound = prepare_smpl_initial_data()

    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=dtype)
    pipeline = MultiGuidance2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        NeRF_renderer=NeRF_renderer,
        **guidance_encoder_group,
        scheduler=scheduler,
        nerf_cond_type=cfg.NeRF.nerf_cond_type,
        use_diff_img_loss=cfg.use_diff_img_loss,
    )
    pipeline = pipeline.to(accelerator.device)
    
    # ref_img_lst_ = cfg.validation.ref_images.copy()
    # guid_folder_lst_ = cfg.validation.guidance_folders.copy()
    # guid_idxes_ = cfg.validation.guidance_indexes.copy()
    clip_length = 24 #cfg.validation.clip_length
    
    ref_img_lst, guid_folder_lst, guid_idxes = [], [], []
    test_folder = cfg.validation.guidance_folders[0] + '/../../../test'
    test_subject_names = os.listdir(test_folder)
    for test_subject in test_subject_names:
        ref_img_lst.append(test_folder + '/' + test_subject + '/camera0000/images/0000.jpg')
        guid_folder_lst.append(test_folder + '/' + test_subject + '/camera0000')
        guid_idxes.append(cfg.validation.guidance_indexes[0])

    val_results = []
    for val_idx, (ref_img_path, guid_folder, guid_start_idx) in enumerate(
        zip(ref_img_lst, guid_folder_lst, guid_idxes)):
        
        video_tensor, ref_img_pil, guid_img_pil_lst = validate_np_video(
            ref_img_path=ref_img_path,
            guid_folder=guid_folder,
            guid_types=guid_types,
            guid_start_idx=guid_start_idx,
            clip_length=clip_length,
            width=width,
            height=height,
            pipe=pipeline,
            generator=generator,
            aug_type=cfg.data.aug_type,
            smpl_model=smpl_model, 
            big_pose_smpl_param=big_pose_smpl_param, 
            big_pose_smpl_vertices=big_pose_smpl_vertices, 
            big_pose_world_bound=big_pose_world_bound,
            image_ratio=cfg.data.image_ratio,
            nerf_rs_scale=cfg.data.nerf_rs_scale,
            white_bg=cfg.data.white_bg,
        )
    
        video_tensor = video_tensor[0, ...].permute(1, 2, 3, 0).cpu().numpy()
        W, H = ref_img_pil.size
        
        video_pil_lst = []
        for frame_idx, image_tensor in enumerate(video_tensor):
            result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
            result_img_pil = result_img_pil.resize((W, H))
            frame_guid_pil_lst = [g[frame_idx].resize((W, H)) for g in guid_img_pil_lst]
            # result_pil_lst = [result_img_pil, ref_img_pil, *frame_guid_pil_lst]
            result_pil_lst = [ref_img_pil, *frame_guid_pil_lst, result_img_pil]
            concated_pil = concat_pil(result_pil_lst)
            
            video_pil_lst.append(concated_pil)
            
        val_results.append({"name": f"val_{val_idx}", "video": video_pil_lst})
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del tmp_denoising_unet
    del pipeline
    torch.cuda.empty_cache()

    return val_results

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()

    # for guidance_type in cfg.data.guids:
    #     guidance_encoder_group[guidance_type] = GuidanceEncoder(
    #         guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
    #         guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
    #         block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
    #     )

    for guidance_type in cfg.data.guids:
        if guidance_type == 'nerf' and cfg.NeRF.nerf_cond_type == '512_512':
            guidance_encoder_group[guidance_type] = GuidanceEncoder(
                guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
                guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels * 2,
                block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
            )
        else:
            guidance_encoder_group[guidance_type] = GuidanceEncoder(
                guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
                guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
                block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
            )

    return guidance_encoder_group

def load_stage1_state_dict(
    NeRF_renderer,
    denoising_unet,
    reference_unet,
    guidance_encoder_group,
    stage1_ckpt_dir, stage1_ckpt_step="latest",
):
    if stage1_ckpt_step == "latest":
        ckpt_files = sorted(os.listdir(stage1_ckpt_dir), key=lambda x: int(x.split("-")[-1].split(".")[0]))
        latest_pth_name = (Path(stage1_ckpt_dir) / ckpt_files[-1]).stem
        stage1_ckpt_step = int(latest_pth_name.split("-")[-1])
    
    NeRF_renderer.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"NeRF_renderer-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    for k, module in guidance_encoder_group.items():
        module.load_state_dict(
            torch.load(
                osp.join(stage1_ckpt_dir, f"guidance_encoder_{k}-{stage1_ckpt_step}.pth"),
                map_location="cpu",
            ),
            strict=False,
        )
    
    logger.info(f"Loaded stage1 models from {stage1_ckpt_dir}, step={stage1_ckpt_step}")

def img_loss_coef(original_samples: torch.Tensor, timesteps: torch.IntTensor, alphas_cumprod_fun: None):

    alphas_cumprod_fun = alphas_cumprod_fun.to(device=original_samples.device)
    alphas_cumprod = alphas_cumprod_fun.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)  

    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod  

def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    tb_tracker = TbTracker(cfg.exp_name, cfg.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with=tb_tracker,
        project_dir=f'{cfg.output_dir}/{cfg.exp_name}',
        kwargs_handlers=[kwargs],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    if cfg.seed is not None:
        seed_everything(cfg.seed)
        
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )
        
    NeRF_renderer = NeRF_Renderer(use_smpl_dist_mask=cfg.NeRF.use_smpl_dist_mask, smpl_type=cfg.NeRF.smpl_type, nerf_cond_type=cfg.NeRF.nerf_cond_type, depth_resolution=cfg.NeRF.depth_resolution, white_bg=False)

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    
    if cfg.unet_additional_kwargs.use_crossview_module:
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            cfg.mm_path,
            view_module_path=cfg.view_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                cfg.unet_additional_kwargs
            ),
        ).to(device="cuda")
    elif cfg.unet_additional_kwargs.use_motion_module:
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            cfg.mm_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                cfg.unet_additional_kwargs
            ),
        ).to(device="cuda")
    else:
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                cfg.unet_additional_kwargs
            ),
        ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    guidance_encoder_group = setup_guidance_encoder(cfg)

    # if not cfg.test_nv and not cfg.test_np:
    load_stage1_state_dict(
        NeRF_renderer,
        denoising_unet,
        reference_unet,
        guidance_encoder_group,
        cfg.stage1_ckpt_dir,
        cfg.stage1_ckpt_step,
    )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    NeRF_renderer.requires_grad_(False)
    for module in guidance_encoder_group.values():
        module.requires_grad_(False)

    if cfg.unet_additional_kwargs.use_crossview_module:
        for name, module in denoising_unet.named_modules():
            if "motion_modules" in name:
                for params in module.parameters():
                    params.requires_grad = True


    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )   
    
    model = HumanDiffModel(
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
        guidance_encoder_group,
        NeRF_renderer=NeRF_renderer,
        nerf_cond_type=cfg.NeRF.nerf_cond_type,
        use_diff_img_loss=cfg.use_diff_img_loss,
    )
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate
        
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps * 2,
    ) 
    
    train_dataset = VideoDataset(
        video_folder=cfg.data.video_folder,
        image_size=cfg.data.image_size,
        sample_frames=cfg.data.sample_frames,
        sample_rate=cfg.data.sample_rate,
        data_parts=cfg.data.data_parts,
        guids=cfg.data.guids,
        extra_region=None,
        bbox_crop=cfg.data.bbox_crop,
        bbox_resize_ratio=tuple(cfg.data.bbox_resize_ratio),
        crossview_num=cfg.data.crossview_num,
        image_ratio=cfg.data.image_ratio,
        nerf_rs_scale=cfg.data.nerf_rs_scale,
        white_bg=cfg.data.white_bg,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )
    
    logger.info("Start training ...")
    logger.info(f"Num Samples: {len(train_dataset)}")
    logger.info(f"Train Batchsize: {cfg.data.train_bs}")
    logger.info(f"Num Epochs: {num_train_epochs}")
    logger.info(f"Total Steps: {cfg.solver.max_train_steps}")
    
    global_step, first_epoch = 0, 0
    
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = f"{cfg.output_dir}/{cfg.exp_name}/checkpoints"
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch    
    
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    if cfg.test_nv:
        if accelerator.is_main_process:

            if accelerator.num_processes == 1:
                for module in torch_dfs(model.denoising_unet):
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                        module.eval_flag=True
            else:
                for module in torch_dfs(model.module.denoising_unet):
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                        module.eval_flag=True

            sample_dicts = log_validation_nv(
                cfg=cfg,
                vae=vae,
                image_enc=image_enc,
                model=model,
                scheduler=val_noise_scheduler,
                accelerator=accelerator,
                width=cfg.data.image_size,
                height=cfg.data.image_size,
                seed=cfg.seed
            )
        exit()

    if cfg.test_np:
        if accelerator.is_main_process:

            if accelerator.num_processes == 1:
                for module in torch_dfs(model.denoising_unet):
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                        module.eval_flag=True
            else:
                for module in torch_dfs(model.module.denoising_unet):
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                        module.eval_flag=True

            sample_dicts = log_validation_np(
                cfg=cfg,
                vae=vae,
                image_enc=image_enc,
                model=model,
                scheduler=val_noise_scheduler,
                accelerator=accelerator,
                width=cfg.data.image_size,
                height=cfg.data.image_size,
                seed=cfg.seed
            )
        exit()

    if accelerator.num_processes == 1:
        for module in torch_dfs(model.denoising_unet):
            if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                module.eval_flag=False
    else:
        for module in torch_dfs(model.module.denoising_unet):
            if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                module.eval_flag=False

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Convert videos to latent space
                batch = to_cuda(batch, torch.device('cuda', torch.cuda.current_device()))
                pixel_values_vid = batch["tgt_vid"].to(weight_dtype)
                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215
                
                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                
                tgt_guid_videos = batch["tgt_guid_vid"]  # (bs, f, c, H, W)
                tgt_guid_videos = tgt_guid_videos.transpose(
                    1, 2
                )  # (bs, c, f, H, W)
                
                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["ref_img"],
                        batch["clip_img"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)                

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                
                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # sqrt_alpha_prod_t, sqrt_beta_prod_t = img_loss_coef(latents, timesteps, train_noise_scheduler.alphas_cumprod)

                model_pred = model(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    clip_image_embeds,
                    tgt_guid_videos,
                    uncond_fwd=uncond_fwd,
                    vae=vae,
                    batch_data_nerf=batch,
                    latents=latents,
                    noise=noise,
                    # sqrt_alpha_prod_t=sqrt_alpha_prod_t,
                    # sqrt_beta_prod_t=sqrt_beta_prod_t,
                    crossview_num=cfg.data.crossview_num,
                )
                
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            save_dir = f'{cfg.output_dir}/{cfg.exp_name}'
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                tb_tracker.add_scalar(tag='train loss', scalar_value=train_loss, global_step=global_step)
                train_loss = 0.0                                                                
                #　save checkpoints
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 6)
                        accelerator.save_state(save_path)
                        
                #  sanity check
                if global_step % cfg.validation.validation_steps == 0 or global_step == 1:
                    ref_forcheck = batch['ref_img'] * 0.5 + 0.5
                    img_forcheck = batch['tgt_vid'] * 0.5 + 0.5
                    ref_forcheck = ref_forcheck.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
                    img_forcheck = rearrange(img_forcheck, 'b f c h w -> b c f h w')
                    guid_forcheck = list(torch.chunk(batch['tgt_guid_vid'], batch['tgt_guid_vid'].shape[2]//3, dim=2))
                    guid_forcheck = [rearrange(g, 'b f c h w -> b c f h w') for g in guid_forcheck]
                    video_forcheck = torch.cat([ref_forcheck, img_forcheck] + guid_forcheck, dim=0).cpu()
                    save_videos_grid(video_forcheck, f'{save_dir}/sanity_check/data-{global_step:06d}-rank{accelerator.device.index}.gif', fps=30, n_rows=3)
    
                if global_step % cfg.validation.validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:

                        if accelerator.num_processes == 1:
                            for module in torch_dfs(model.denoising_unet):
                                if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                                    module.eval_flag=True
                        else:
                            for module in torch_dfs(model.module.denoising_unet):
                                if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                                    module.eval_flag=True

                        sample_dicts = log_validation(
                            cfg=cfg,
                            vae=vae,
                            image_enc=image_enc,
                            model=model,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.image_size,
                            height=cfg.data.image_size,
                            seed=cfg.seed
                        )

                        for sample_dict in sample_dicts:
                            sample_name = sample_dict["name"]
                            video = sample_dict["video"]
                            count = 0
                            img_list = []
                            for img in video:
                                img_list.append(np.array(img))
                                count += 1
                                if count % 4 == 0:
                                    img = np.concatenate(img_list, 0)
                                    img = Image.fromarray((img).astype(np.uint8))
                                    img.save(f"{save_dir}/validation/{global_step:06d}-{sample_name}-{count}.png")
                                    img_list = []
                            # save_videos_from_pil(video, f'{save_dir}/validation/6fps-{global_step:06d}-{sample_name}.mp4', fps=6)
                            # save_videos_from_pil(video, f'{save_dir}/validation/30fps-{global_step:06d}-{sample_name}.mp4', fps=30)

                        sample_dicts = log_validation_np_video(
                            cfg=cfg,
                            vae=vae,
                            image_enc=image_enc,
                            model=model,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.image_size,
                            height=cfg.data.image_size,
                            seed=cfg.seed
                        )

                        for sample_dict in sample_dicts:
                            sample_name = sample_dict["name"]
                            video = sample_dict["video"]
                            save_videos_from_pil(video, f'{save_dir}/validation/6fps-{global_step:06d}-{sample_name}.mp4', fps=6)
                            # save_videos_from_pil(video, f'{save_dir}/validation/30fps-{global_step:06d}-{sample_name}.mp4', fps=30)

                        if accelerator.num_processes == 1:
                            for module in torch_dfs(model.denoising_unet):
                                if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                                    module.eval_flag=False
                        else:
                            for module in torch_dfs(model.module.denoising_unet):
                                if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
                                    module.eval_flag=False

                    # accelerator.wait_for_everyone()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 3,
            }
            progress_bar.set_postfix(**logs)

            # save model after each epoch
            # if accelerator.is_main_process and (epoch + 1) % cfg.save_model_epoch_interval == 0 :
            if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process: 
                # save motion module only
                unwrap_model = accelerator.unwrap_model(model)
                save_checkpoint(
                    unwrap_model.denoising_unet,
                    f"{save_dir}/saved_models",
                    "motion_module",
                    global_step,
                    total_limit=None,
                )  
                # save_checkpoint(
                #     unwrap_model.reference_unet,
                #     f"{save_dir}/saved_models",
                #     "reference_unet",
                #     global_step,
                #     total_limit=None,
                # )
                # save_checkpoint(
                #     unwrap_model.denoising_unet,
                #     f"{save_dir}/saved_models",
                #     "denoising_unet",
                #     global_step,
                #     total_limit=None,
                # )
                # for guid_type in unwrap_model.guidance_types:
                #     save_checkpoint(
                #         getattr(unwrap_model, f"guidance_encoder_{guid_type}"),
                #         f"{save_dir}/saved_models",
                #         f"guidance_encoder_{guid_type}",
                #         global_step,
                #         total_limit=None,
                #     )

            if global_step >= cfg.solver.max_train_steps:
                break                    

    accelerator.wait_for_everyone()
    accelerator.end_training()

def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    # state_dict = model.state_dict()
    # torch.save(state_dict, save_path)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, save_path)

    # mm_state_dict = OrderedDict()
    # state_dict = model.state_dict()
    # for key in state_dict:
    #     if "motion_module" in key:
    #         mm_state_dict[key] = state_dict[key]

    # torch.save(mm_state_dict, save_path)

    
    
if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    save_dir = os.path.join(config.output_dir, config.exp_name)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)
    # save config, script
    shutil.copy(args.config, os.path.join(save_dir, 'sanity_check', f'{config.exp_name}.yaml'))
    shutil.copy(os.path.abspath(__file__), os.path.join(save_dir, 'sanity_check'))
    
    main(config)
        