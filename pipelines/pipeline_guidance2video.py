import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from models.mutual_self_attention import ReferenceAttentionControl


@dataclass
class MultiGuidance2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

class MultiGuidance2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        guidance_encoder_depth=None,
        guidance_encoder_normal=None,
        guidance_encoder_semantic_map=None,
        guidance_encoder_dwpose=None,
        guidance_encoder_hrnet=None,
        guidance_encoder_nerf=None,
        NeRF_renderer=None,
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
        nerf_cond_type=None,
        use_diff_img_loss=False,
        test_nerf=False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            NeRF_renderer=NeRF_renderer,
            guidance_encoder_depth=guidance_encoder_depth,
            guidance_encoder_normal=guidance_encoder_normal,
            guidance_encoder_semantic_map=guidance_encoder_semantic_map,
            guidance_encoder_dwpose=guidance_encoder_dwpose,
            guidance_encoder_hrnet=guidance_encoder_hrnet,
            guidance_encoder_nerf=guidance_encoder_nerf,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        self.latents = None

        if nerf_cond_type == '480_480_upscale':
            self.upsample = torch.nn.Upsample(scale_factor=1.6, mode='bilinear')
        else:
            self.upsample = torch.nn.Upsample(scale_factor=1.0, mode='bilinear')

        self.use_diff_img_loss = use_diff_img_loss
        self.test_nerf = test_nerf

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        multi_guidance_lst,
        guidance_types,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        crossview_num=4,
        batch_data_nerf=None,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image, return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels
        if self.latents is None:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                width,
                height,
                video_length,
                clip_image_embeds.dtype,
                device,
                generator,
            )
            self.latents = latents
        else:
            latents = self.latents
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
        guidance_fea_lst = []
        for guid_idx, guidance_image_lst in enumerate(multi_guidance_lst):
            guidance_tensor_lst = [torch.from_numpy(np.array(guidance_image.resize((width, height)))) / 255. for guidance_image in guidance_image_lst]
            guidance_tensor = torch.stack(guidance_tensor_lst, dim=0).permute(3, 0, 1, 2) #(c, f, h, w)
            guidance_tensor = guidance_tensor.unsqueeze(0)
            
            guidance_type = guidance_types[guid_idx]
            guidance_encoder = getattr(self, f"guidance_encoder_{guidance_type}")
            guidance_tensor = guidance_tensor.to(device=device, dtype=guidance_encoder.dtype)
            guidance_fea_lst += [guidance_encoder(guidance_tensor)]
            
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
                            minibatch_data_nerf[key] = batch_data_nerf[key][i:i+chunk]
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
            if self.test_nerf:
                return feature_nerf_image_crop[:, :, :3].permute(0, 2, 1, 3, 4), feature_nerf_image[:, :, :3]

            guidance_encoder = getattr(self, f"guidance_encoder_nerf")
            if len(batch_data_nerf['tgt_ray_o'].shape) == 4:
                guidance_fea = guidance_encoder(self.upsample(feature_nerf_image_crop.reshape(-1, *feature_nerf_image_crop.shape[2:])).reshape(bs, tgt_num, feature_nerf_image_crop.shape[2], *guidance_tensor.shape[-2:]).permute(0, 2, 1, 3, 4))
            else:
                guidance_fea = guidance_encoder(self.upsample(feature_nerf_image_crop)[:, :, None])
            guidance_fea_lst += [guidance_fea]
        else:
            feature_nerf_image = None

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)
        guidance_fea = torch.cat([guidance_fea] * 2) if do_classifier_free_guidance else guidance_fea

        ## debug ##
        # latents_load = torch.from_numpy(np.load('latents.npy')).to(device).to(self.vae.dtype)
        # latents_lst = [latents_load for _ in range(video_length)]
        # latents = torch.cat(latents_lst, 2)
        ##

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        # t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    guidance_fea=guidance_fea,
                    crossview_num=crossview_num,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        if batch_data_nerf is not None:
            return images, feature_nerf_image[:, :, :3]

        return MultiGuidance2VideoPipelineOutput(videos=images)
