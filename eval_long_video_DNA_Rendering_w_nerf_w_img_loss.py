import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl
from models.guidance_encoder import GuidanceEncoder
from models.champ_model import HumanDiffModel

from models.mutual_self_attention import torch_dfs
from models.attention import TemporalBasicTransformerBlock, BasicTransformerBlock

from pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline

from utils.video_utils import resize_tensor_frames, save_videos_grid, pil_list_to_tensor

from datasets.video_dataset_DNA_Rendering_w_nerf import sample_ray

from models.recon_NeRF import NeRF_Renderer, to_cuda

import cv2
from data.DNA_Rendering.dna_rendering_sample_code.SMCReader import SMCReader
from smplx.body_models import SMPLX

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

def prepare_smpl_initial_data():
    smpl_model, big_pose_smpl_param_dct, big_pose_smpl_vertices_dct, big_pose_world_bound_dct = {}, {}, {}, {}
    for gender in ['female', 'male', 'neutral']:
        smpl_model[gender] = SMPLX('assets/models/smplx/', smpl_type='smplx',
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

        body_model_output = smpl_model[gender](
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

        big_pose_smpl_param_dct[gender], big_pose_smpl_vertices_dct[gender], big_pose_world_bound_dct[gender] = big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound

    return smpl_model, big_pose_smpl_param_dct, big_pose_smpl_vertices_dct, big_pose_world_bound_dct

def prepare_smpl_data(smpl_dict, gender, smpl_model):

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

    body_model_output = smpl_model[gender](
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

def load_image(img_path, smpl_model, nerf_rs_scale=0.125, image_ratio=0.25, tgt_view=False):
    part_name = str(img_path).split('/')[-4][:-8]
    smc_name = str(img_path).split('/')[-4][-7:]
    view_index = int(str(img_path).split('/')[-3][-4:])
    img_idx = int(str(img_path).split('/')[-1][:-4])
    smc_path = f'data/DNA_Rendering/{part_name}/dna_rendering_part{part_name[-1]}_main/{smc_name}.smc'
    smc_reader = SMCReader(smc_path)
    annots_file_path = smc_path.replace('main', 'annotations').split('.')[0] + '_annots.smc'
    smc_annots_reader = SMCReader(annots_file_path)

    gender = smc_reader.actor_info['gender']

    # load reference image and mask
    img = smc_reader.get_img('Camera_5mp', view_index, Frame_id=img_idx, Image_type='color')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    msk = smc_annots_reader.get_mask(view_index, Frame_id=img_idx)

    # Load reference K, R, T
    cam_params = smc_annots_reader.get_Calibration(view_index)
    K, D = cam_params['K'], cam_params['D']

    # load camera 
    c2w = np.eye(4)
    c2w[:3,:3] = cam_params['RT'][:3, :3]
    c2w[:3,3:4] = cam_params['RT'][:3, 3].reshape(-1, 1)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3,:3].astype(np.float32)
    T = w2c[:3, 3].reshape(-1, 1).astype(np.float32)

    # undistort image and mask
    H, W = int(img.shape[0]), int(img.shape[1])
    newcamera, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0) 
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcamera, (W, H), 5)
    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    msk = cv2.remap(msk, mapx, mapy, cv2.INTER_LINEAR)

    white_background = False
    img[msk == 0] = 1 if white_background else 0

    if image_ratio != 1.:
        H, W = int(img.shape[0] * image_ratio), int(img.shape[1] * image_ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        K[:2] = K[:2] * image_ratio

    img_pil = Image.fromarray(img)
    img_nerf = img.astype(np.float32) / 255.

    # prepare smpl at the reference view
    smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=img_idx)
    smpl_param, world_vertex, world_bound = prepare_smpl_data(smpl_dict, gender, smpl_model)

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
        _, _, _, _, _, mask_at_box_full, _ = sample_ray(
        img_nerf, msk, K, R, T, world_bound, image_scaling=1.0)

        img_nerf, ray_o, ray_d, near, far, mask_at_box, bkgd_msk = sample_ray(
        img_nerf, msk, K, R, T, world_bound, image_scaling=nerf_rs_scale)

        data_nerf['tgt_img_nerf'] = np.transpose(img_nerf, (2,0,1))
        data_nerf['tgt_smpl_param'] = smpl_param
        data_nerf['tgt_world_vertex'] = world_vertex
        data_nerf['tgt_ray_o'] = ray_o
        data_nerf['tgt_ray_d'] = ray_d
        data_nerf['tgt_near'] = near
        data_nerf['tgt_far'] = far
        data_nerf['tgt_mask_at_box'] = mask_at_box
        data_nerf['tgt_mask_at_box_full'] = mask_at_box_full
        data_nerf['tgt_bkgd_msk'] = bkgd_msk

    return img_pil, data_nerf

def setup_savedir(cfg):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if cfg.exp_name is None:
        savedir = f"results/exp-{time_str}"
    else:
        savedir = f"results/{cfg.exp_name}-{time_str}"

    os.makedirs(savedir, exist_ok=True)

    return savedir


def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    for guidance_type in cfg.guidance_types:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        ).to(device="cuda", dtype=weight_dtype)

    return guidance_encoder_group


def process_semantic_map(semantic_map_path: Path):
    image_name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / image_name
    semantic_array = np.array(Image.open(semantic_map_path))
    mask_array = np.array(Image.open(mask_path).convert("RGB"))
    semantic_pil = Image.fromarray(np.where(mask_array > 0, semantic_array, 0))

    return semantic_pil


def combine_guidance_data(cfg, guidance_data_folder=None):
    guidance_types = cfg.guidance_types
    # guidance_data_folder = cfg.data.guidance_data_folder

    guidance_pil_group = dict()
    for guidance_type in guidance_types:
        if guidance_type != 'nerf':
            guidance_pil_group[guidance_type] = []
            guidance_image_lst = sorted(
                Path(osp.join(guidance_data_folder, guidance_type)).iterdir()
            )
            guidance_image_lst = (
                guidance_image_lst
                if not cfg.data.frame_range
                else guidance_image_lst[cfg.data.frame_range[0]:cfg.data.frame_range[1]]
            )

            for guidance_image_path in guidance_image_lst:
                # Add black background to semantic map
                if guidance_type == "semantic_map":
                    # guidance_pil_group[guidance_type] += [
                    #     resize_pil(process_semantic_map(guidance_image_path), cfg.height)
                    # ]
                    guidance_pil_group[guidance_type] += [
                        resize_pil(Image.open(guidance_image_path).convert("RGB"), cfg.height)
                    ]
                else:
                    guidance_pil_group[guidance_type] += [
                        resize_pil(Image.open(guidance_image_path).convert("RGB"), cfg.height)
                    ]

    # get video length from the first guidance sequence
    first_guidance_length = len(list(guidance_pil_group.values())[0])
    # ensure all guidance sequences are of equal length
    assert all(
        len(sublist) == first_guidance_length
        for sublist in list(guidance_pil_group.values())
    )

    return guidance_pil_group, first_guidance_length, guidance_image_lst


def inference(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    device="cuda",
    dtype=torch.float16,
    batch_data_nerf=None,
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    NeRF_renderer = model.NeRF_renderer
    guidance_types = cfg.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}")
        for g in guidance_types
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        NeRF_renderer=NeRF_renderer,
        **guidance_encoder_group,
        scheduler=scheduler,
        nerf_cond_type=cfg.NeRF.nerf_cond_type,
        test_nerf=cfg.test_nerf,
    )
    pipeline = pipeline.to(device, dtype)

    video, feature_nerf_image = pipeline(
        ref_image_pil,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        crossview_num=1,
        batch_data_nerf=batch_data_nerf,
        generator=generator,
    )#.videos

    del pipeline
    torch.cuda.empty_cache()

    return video, feature_nerf_image


def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # save_dir = setup_savedir(cfg)
    save_dir = cfg.exp_name
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'nerf_np'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pred_np'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'gt_np'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'video_vis'), exist_ok=True)
    
    logging.info(f"Running inference ...")

    # setup pretrained models
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    NeRF_renderer = NeRF_Renderer(use_smpl_dist_mask=cfg.NeRF.use_smpl_dist_mask, nerf_cond_type=cfg.NeRF.nerf_cond_type, depth_resolution=cfg.NeRF.depth_resolution).to(dtype=torch.float32, device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        dtype=weight_dtype, device="cuda"
    )

    if cfg.unet_additional_kwargs.use_crossview_module:
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            cfg.motion_module_path,
            view_module_path=cfg.view_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                cfg.unet_additional_kwargs
            ),
        ).to(device="cuda")
    elif cfg.unet_additional_kwargs.use_motion_module:
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            cfg.motion_module_path,
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

    # if not cfg.test_nerf:
    #     denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    #         cfg.base_model_path,
    #         cfg.motion_module_path,
    #         view_module_path=cfg.view_module_path,
    #         subfolder="unet",
    #         unet_additional_kwargs=cfg.unet_additional_kwargs,
    #     ).to(dtype=weight_dtype, device="cuda")
    # else:
    #     denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    #         cfg.base_model_path,
    #         '',
    #         subfolder="unet",
    #         unet_additional_kwargs=cfg.unet_additional_kwargs,
    #     ).to(dtype=weight_dtype, device="cuda")

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    guidance_encoder_group = setup_guidance_encoder(cfg)

    ckpt_dir = cfg.ckpt_dir
    denoising_unet.load_state_dict(
        torch.load(
            osp.join(ckpt_dir, f"denoising_unet-150000.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    NeRF_renderer.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"NeRF_renderer-150000.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            osp.join(ckpt_dir, f"reference_unet-150000.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
        if guidance_type == 'nerf' and not cfg.test_nerf:
            guidance_encoder_module.load_state_dict(
                torch.load(
                    osp.join(ckpt_dir, f"guidance_encoder_{guidance_type}-150000.pth"),
                    map_location="cpu",
                ),
                strict=False,
            )

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
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        guidance_encoder_group=guidance_encoder_group,
        NeRF_renderer=NeRF_renderer,
        nerf_cond_type=cfg.NeRF.nerf_cond_type,
    ).to("cuda") #.to("cuda", dtype=weight_dtype)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    for module in torch_dfs(model.denoising_unet):
        if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock):
            module.eval_flag=True

    ref_img_lst, guid_folder_lst = [], []
    img_folder_lst = os.listdir(cfg.data.video_folder)[0:]
    for img_folder in img_folder_lst:
        if 'DNA_Rendering' in cfg.data.video_folder:
            img_name_lst = sorted(os.listdir(cfg.data.video_folder + '/' + img_folder + '/camera0022/images')) 
            ref_img = cfg.data.video_folder + '/' + img_folder + '/camera0022/images/' + img_name_lst[0]
            guid_folder = cfg.data.video_folder + '/' + img_folder + '/camera0022'
        else:
            img_name_lst = sorted(os.listdir(cfg.data.video_folder + '/' + img_folder + '/images')) 
            ref_img = cfg.data.video_folder + '/' + img_folder + '/images/' + img_name_lst[0]
            guid_folder = cfg.data.video_folder + '/' + img_folder
        ref_img_lst.append(ref_img)
        guid_folder_lst.append(guid_folder)

    smpl_model, big_pose_smpl_param, big_pose_smpl_vertices, big_pose_world_bound = prepare_smpl_initial_data()

    for val_idx, (ref_image_path, guid_folder) in enumerate(zip(ref_img_lst, guid_folder_lst)):

        # ref_image_path = cfg.data.ref_image_path
        # ref_image_pil = Image.open(ref_image_path)
        # ref_image_pil = resize_pil(ref_image_pil, cfg.height)
        ref_image_pil, ref_data_nerf = load_image(ref_image_path, smpl_model)
        ref_image_pil = resize_pil(ref_image_pil, cfg.height)
        ref_image_w, ref_image_h = ref_image_pil.size

        guidance_pil_group, video_length, guidance_image_lst = combine_guidance_data(cfg, guid_folder)

        # load tgt nerf data 
        tgt_data_nerf_dct = {}
        tgt_data_nerf_dct['tgt_smpl_param'] = {}
        for guid_img_path in guidance_image_lst:
            tgt_img_path = Path(str(guid_img_path).replace('normal', 'images'))
            tgt_img_pil, tgt_data_nerf = load_image(tgt_img_path, smpl_model, nerf_rs_scale=cfg.data.nerf_rs_scale, image_ratio=cfg.data.image_ratio, tgt_view=True)
            for key in tgt_data_nerf.keys():
                if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_mask_at_box_full', 'tgt_bkgd_msk']:
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
            if key in ['tgt_img_nerf', 'tgt_world_vertex', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_mask_at_box_full', 'tgt_bkgd_msk']:
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
        batch_data_nerf['gender'] = batch_data_nerf['gender'] * len(guidance_image_lst)

        batch_data_nerf = to_cuda(batch_data_nerf, torch.device('cuda', torch.cuda.current_device()))

        result_video_tensor, feature_nerf_image = inference(
            cfg=cfg,
            vae=vae,
            image_enc=image_enc,
            model=model,
            scheduler=noise_scheduler,
            ref_image_pil=ref_image_pil,
            guidance_pil_group=guidance_pil_group,
            video_length=video_length,
            width=cfg.width,
            height=cfg.height,
            device="cuda",
            dtype=weight_dtype,
            batch_data_nerf=batch_data_nerf,
        )  # (1, c, f, h, w)

        tgt_mask_at_box_full = tgt_data_nerf_dct['tgt_mask_at_box_full'].reshape(tgt_data_nerf_dct['tgt_mask_at_box_full'].shape[0], tgt_img_pil.size[1], tgt_img_pil.size[0])
        x, y, w, h = 0, (tgt_mask_at_box_full.shape[-2] - tgt_mask_at_box_full.shape[-1])//2, min(tgt_img_pil.size[0], tgt_img_pil.size[1]), min(tgt_img_pil.size[0], tgt_img_pil.size[1])
        tgt_mask_at_box_crop = tgt_mask_at_box_full[:, y:y + h, x:x + w]
        result_video_tensor[np.repeat(tgt_mask_at_box_crop[None, None], 3, axis=1) == 0] = 0


        video_name = img_folder_lst[val_idx]
        result_video_tensor = resize_tensor_frames(
            result_video_tensor, (ref_image_h, ref_image_w)
        )
        save_videos_grid(result_video_tensor, osp.join(save_dir, 'video_vis', f"animation-{video_name}.mp4"))

        ref_video_tensor = transforms.ToTensor()(ref_image_pil)[None, :, None, ...].repeat(
            1, 1, video_length, 1, 1
        )
        guidance_video_tensor_lst = []
        for guidance_pil_lst in guidance_pil_group.values():
            guidance_video_tensor_lst += [
                pil_list_to_tensor(guidance_pil_lst, size=(ref_image_h, ref_image_w))
            ]
        guidance_video_tensor = torch.stack(guidance_video_tensor_lst, dim=0)

        grid_video = torch.cat([ref_video_tensor, result_video_tensor], dim=0)
        grid_video_wguidance = torch.cat(
            [ref_video_tensor, result_video_tensor, guidance_video_tensor], dim=0
        )

        save_videos_grid(grid_video, osp.join(save_dir, 'video_vis', f"grid-{video_name}.mp4"))
        save_videos_grid(grid_video_wguidance, osp.join(save_dir, 'video_vis', f"grid_wguidance-{video_name}.mp4"))

        # save gt img
        gt_img_pil_lst, gt_img_name_lst = [], []
        gt_img_lst = sorted(os.listdir(str(guid_folder) + '/images'))[cfg.data.frame_range[0]:cfg.data.frame_range[1]]
        for gt_img_path in gt_img_lst:
            # gt_img_pil = Image.open(str(guid_folder) + '/images/' + gt_img_path).convert("RGB")
            tgt_image_path = str(guid_folder) + '/images/' + gt_img_path
            gt_img_pil, ref_data_nerf = load_image(tgt_image_path, smpl_model)
            gt_img_pil = resize_pil(gt_img_pil, cfg.height)
            gt_img_pil_lst += [gt_img_pil]
            gt_img_name_lst += [str(guid_folder).split('/')[-2] + '_' + str(guid_folder).split('/')[-1] + '_' + gt_img_path]

        for image_id, image_name in enumerate(gt_img_name_lst):

            rgb_nerf_img_pil = Image.fromarray((255*feature_nerf_image[0, image_id]).permute(1,2,0).cpu().numpy().astype(np.uint8))
            rgb_nerf_img_pil = resize_pil(rgb_nerf_img_pil, min(rgb_nerf_img_pil.size[0], rgb_nerf_img_pil.size[1])).resize((768, 768))
            rgb_nerf_img_pil.save(f"{save_dir}/nerf_np/{image_name}")

            pred_img = np.array(result_video_tensor[0, :, image_id].permute(1,2,0))
            pred_img = Image.fromarray((pred_img * 255).astype(np.uint8))
            gt_img = gt_img_pil_lst[image_id]
            pred_img.save(f"{save_dir}/pred_np/{image_name}")
            gt_img.save(f"{save_dir}/gt_np/{image_name}")

        logging.info(f"Inference completed, results saved in {save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    main(cfg)
