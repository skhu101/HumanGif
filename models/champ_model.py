import torch
import torch.nn as nn
from einops import rearrange
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel


class HumanDiffModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        reference_control_writer,
        reference_control_reader,
        guidance_encoder_group,
        NeRF_renderer=None,
        nerf_cond_type=None,
        use_diff_img_loss=False
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet

        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

        if nerf_cond_type == '480_480_upscale':
            self.upsample = nn.Upsample(scale_factor=1.6, mode='bilinear')
        else:
            self.upsample = nn.Upsample(scale_factor=1.0, mode='bilinear')

        self.guidance_types = []
        self.guidance_input_channels = []

        for guidance_type, guidance_module in guidance_encoder_group.items():
            setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
            self.guidance_types.append(guidance_type)
            self.guidance_input_channels.append(guidance_module.guidance_input_channels)

        if NeRF_renderer is not None:
            self.NeRF_renderer = NeRF_renderer

        self.nerf_cond_type = nerf_cond_type
        self.use_diff_img_loss = use_diff_img_loss

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        multi_guidance_cond,
        uncond_fwd: bool = False,
        crossview_num: int = 4,
        vae=None, 
        batch_data_nerf=None,
        latents=None,
        noise=None,
        sqrt_alpha_prod_t=None, 
        sqrt_beta_prod_t=None,
    ):

        if batch_data_nerf is not None:
            if len(batch_data_nerf['tgt_ray_o'].shape) == 4:
                bs, tgt_num = batch_data_nerf['tgt_ray_o'].shape[0], batch_data_nerf['tgt_ray_o'].shape[1]
                feature_nerf_image_lst, depth_nerf_image_lst, weights_nerf_image_lst = [], [], []
                chunk = 4
                for i in range(0, batch_data_nerf['tgt_ray_o'].shape[1], chunk):
                    chunk = batch_data_nerf['tgt_ray_o'].reshape(-1, *batch_data_nerf['tgt_ray_o'].shape[2:])[i:i+chunk].shape[0]
                    minibatch_data_nerf = {}
                    for key in batch_data_nerf.keys():
                        if key in ['tgt_img_nerf', 'tgt_ray_o', 'tgt_ray_d', 'tgt_near', 'tgt_far', 'tgt_mask_at_box', 'tgt_world_vertex']:
                            minibatch_data_nerf[key] = batch_data_nerf[key].reshape(-1, *batch_data_nerf[key].shape[2:])[i:i+chunk]
                        elif key in ['ref_world_vertex', 'ref_K', 'ref_R', 'ref_T', 'big_pose_world_vertex', 'big_pose_world_bound']:
                            minibatch_data_nerf[key] = batch_data_nerf[key].expand(chunk, -1, -1)
                        elif key in ['ref_img_nerf']:
                            minibatch_data_nerf[key] = batch_data_nerf[key].expand(chunk, -1, -1, -1)                  
                        if key in ['tgt_smpl_param']:
                            minibatch_data_nerf[key] = {}
                            for smpl_key in batch_data_nerf[key]:
                                minibatch_data_nerf[key][smpl_key] = batch_data_nerf[key][smpl_key].reshape(-1, *batch_data_nerf[key][smpl_key].shape[2:])[i:i+chunk]
                        if key in ['ref_smpl_param', 'big_pose_smpl_param']:
                            minibatch_data_nerf[key] = {}
                            for smpl_key in batch_data_nerf[key]:
                                minibatch_data_nerf[key][smpl_key] = batch_data_nerf[key][smpl_key].expand(chunk, -1, -1)
                        if key in ['gender']:
                            minibatch_data_nerf[key] = batch_data_nerf[key] * chunk
                    feature_nerf_image, depth_nerf_image, weights_nerf_image = self.NeRF_renderer(minibatch_data_nerf['tgt_ray_o'], minibatch_data_nerf['tgt_ray_d'], minibatch_data_nerf['tgt_near'], minibatch_data_nerf['tgt_far'], minibatch_data_nerf['tgt_mask_at_box'], minibatch_data_nerf['tgt_img_nerf'], minibatch_data_nerf['tgt_smpl_param'], minibatch_data_nerf['tgt_world_vertex'], minibatch_data_nerf['ref_img_nerf'], minibatch_data_nerf['ref_smpl_param'], minibatch_data_nerf['ref_world_vertex'], minibatch_data_nerf['ref_K'], minibatch_data_nerf['ref_R'], minibatch_data_nerf['ref_T'], minibatch_data_nerf['big_pose_smpl_param'], minibatch_data_nerf['big_pose_world_vertex'], minibatch_data_nerf['big_pose_world_bound'], minibatch_data_nerf['gender'])
                    feature_nerf_image_lst.append(feature_nerf_image)
                    depth_nerf_image_lst.append(depth_nerf_image)
                    weights_nerf_image_lst.append(weights_nerf_image)
                feature_nerf_image = torch.cat(feature_nerf_image_lst, dim=0)
                depth_nerf_image = torch.cat(depth_nerf_image_lst, dim=0)
                weights_nerf_image = torch.cat(weights_nerf_image_lst, dim=0)
                feature_nerf_image = feature_nerf_image.reshape(bs, tgt_num, *feature_nerf_image.shape[1:])
                depth_nerf_image = depth_nerf_image.reshape(bs, tgt_num, *depth_nerf_image.shape[1:])
                weights_nerf_image = weights_nerf_image.reshape(bs, tgt_num, *weights_nerf_image.shape[1:])
            else:
                minibatch_data_nerf = batch_data_nerf
                feature_nerf_image, depth_nerf_image, weights_nerf_image = self.NeRF_renderer(minibatch_data_nerf['tgt_ray_o'], minibatch_data_nerf['tgt_ray_d'], minibatch_data_nerf['tgt_near'], minibatch_data_nerf['tgt_far'], minibatch_data_nerf['tgt_mask_at_box'], minibatch_data_nerf['tgt_img_nerf'], minibatch_data_nerf['tgt_smpl_param'], minibatch_data_nerf['tgt_world_vertex'], minibatch_data_nerf['ref_img_nerf'], minibatch_data_nerf['ref_smpl_param'], minibatch_data_nerf['ref_world_vertex'], minibatch_data_nerf['ref_K'], minibatch_data_nerf['ref_R'], minibatch_data_nerf['ref_T'], minibatch_data_nerf['big_pose_smpl_param'], minibatch_data_nerf['big_pose_world_vertex'], minibatch_data_nerf['big_pose_world_bound'], minibatch_data_nerf['gender'])
            crop_size = min(feature_nerf_image.shape[-2], feature_nerf_image.shape[-1])
            x, y, w, h = (feature_nerf_image.shape[-1] - crop_size)//2, (feature_nerf_image.shape[-2] - crop_size)//2, crop_size, crop_size
            feature_nerf_image_crop = feature_nerf_image[..., y:y + h, x:x + w]
        else:
            feature_nerf_image_crop = None

        if 'nerf' in self.guidance_types:
            guidance_cond_group = torch.split(
                multi_guidance_cond, self.guidance_input_channels[:-1], dim=1
            )
        else:
            guidance_cond_group = torch.split(
                multi_guidance_cond, self.guidance_input_channels, dim=1
            )
        guidance_fea_lst = []
        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            guidance_fea = guidance_encoder(guidance_cond)
            guidance_fea_lst += [guidance_fea]

        if 'nerf' in self.guidance_types:
            guidance_encoder = getattr(self, f"guidance_encoder_nerf")
            if len(feature_nerf_image_crop.shape) == 5:
                if feature_nerf_image_crop.shape[-1] == guidance_cond.shape[-1]:
                    guidance_fea = guidance_encoder(feature_nerf_image_crop.permute(0, 2, 1, 3, 4))
                else:
                    guidance_fea = guidance_encoder(self.upsample(feature_nerf_image_crop.reshape(-1, *feature_nerf_image_crop.shape[2:])).reshape(bs, tgt_num, feature_nerf_image_crop.shape[2], *guidance_cond.shape[-2:]).permute(0, 2, 1, 3, 4))                
            else:
                if feature_nerf_image_crop.shape[-1] == guidance_cond.shape[-1]:
                    guidance_fea = guidance_encoder(feature_nerf_image_crop[:, :, None])
                else:
                    guidance_fea = guidance_encoder(self.upsample(feature_nerf_image_crop)[:, :, None])
            guidance_fea_lst += [guidance_fea]

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)
        uncond_fwd=False
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=guidance_fea,
            encoder_hidden_states=clip_image_embeds,
            crossview_num=crossview_num,
            feature_nerf_image=None,
        ).sample

        if batch_data_nerf is not None:
            if len(batch_data_nerf['tgt_ray_o'].shape) == 4:
                return model_pred

        if self.use_diff_img_loss:
            pred_epsilon = sqrt_alpha_prod_t * model_pred.to(noise.dtype) + sqrt_beta_prod_t * noisy_latents
            if self.nerf_cond_type == '512_512_3':
                latents_ = latents + 0.01 * sqrt_beta_prod_t / (sqrt_alpha_prod_t + 1e-5) * (noise - pred_epsilon)
            else:
                latents_ = latents + 0.1 * sqrt_beta_prod_t / (sqrt_alpha_prod_t + 1e-5) * (noise - pred_epsilon)
            latents_ = 1 / 0.18215 * latents_
            latents_ = torch.clamp(latents_, min=-1000, max=1000)
            latents_ = rearrange(latents_, "b c f h w -> (b f) c h w")
            rgb_diff_pred = vae.decode(latents_).sample
            if not torch.isnan(rgb_diff_pred).any() and not torch.isinf(rgb_diff_pred).any(): 
                rgb_diff_pred = torch.clamp(rgb_diff_pred, min=-1, max=1)
            else:
                rgb_diff_pred = rgb_diff_pred

        if batch_data_nerf is not None:
            if self.use_diff_img_loss:
                return model_pred, feature_nerf_image[:, :3], weights_nerf_image, rgb_diff_pred, None
            else:
                return model_pred, feature_nerf_image[:, :3], weights_nerf_image, None, None
        else:
            return model_pred
