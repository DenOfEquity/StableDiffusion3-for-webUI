# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

import PIL.Image
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


#logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,                                              # (`SchedulerMixin`): scheduler to get timesteps from.
    num_inference_steps: Optional[int] = None,              # (`int`):            number of diffusion steps used  - priority 3
    device: Optional[Union[str, torch.device]] = None,      # (`str` or `torch.device`, *optional*): device to move timesteps to. If `None`, not moved.
    timesteps: Optional[List[int]] = None,                  # (`List[int]`, *optional*): custom timesteps, length overrides num_inference_steps - priority 1
    sigmas: Optional[List[float]] = None,                   # (`List[float]`, *optional*): custom sigmas, length overrides num_inference_steps - priority 2
    **kwargs,
):
    #   stop aborting on recoverable errors!
    #   default to using timesteps
    if timesteps is not None and "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None and "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class SD3Pipeline_DoE_combined (DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,

        controlnet: Union[
            SD3ControlNetModel, List[SD3ControlNetModel], Tuple[SD3ControlNetModel], SD3MultiControlNetModel
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.latent_channels = (
            self.vae.config.latent_channels
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.latent_channels)
        self.mask_processor  = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.vae.config.latent_channels, 
                                                 do_normalize=False, do_binarize=False, do_convert_grayscale=True)

#        self.tokenizer_max_length = (
#            self.tokenizer.model_max_length
#            if hasattr(self, "tokenizer") and self.tokenizer is not None
#            else 77
#        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )



    def check_inputs(
        self,
        strength,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if strength < 0:
            strength = 0.0
            print ("Warning: value of strength has been clamped to 0.0 from lower")
        elif strength > 1:
            strength = 1.0
            print ("Warning: value of strength has been clamped to 1.0 from higher")
            
        if prompt_embeds == None or negative_prompt_embeds == None or pooled_prompt_embeds == None or negative_pooled_prompt_embeds == None:
            raise ValueError(f"All prompt embeds must be provided.")
            
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, noise, timestep, strength, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)
        batch_size = num_images_per_prompt

        if image.shape[1] == self.vae.config.latent_channels:
            init_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError (f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts.")
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        latents_image = init_latents

        if noise == None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        elif noise.shape != shape:
            #why not just regenerate?
            raise ValueError (f"Provided noise latents have incorrect shape {noise.shape}, expected {shape}.")

        # get latents
        if strength == 1.0: #   not convinced by this, but probably exactly the same
            init_latents = noise
        else:
            init_latents = self.scheduler.scale_noise(init_latents, timestep, noise)

        latents = init_latents.to(device=device, dtype=dtype)

        return latents, latents_image.to(device=device, dtype=dtype), noise.to(device=device, dtype=dtype)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    #   controlnet
    def prepare_image(
        self,
        image,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.image_processor.preprocess(image).to(device=device, dtype=dtype)
        image = self.vae.encode(image).latent_dist.sample()
        image = (image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        image = image.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_mask_latents(
        self, mask, masked_image, num_images_per_prompt, dtype, device, generator
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(masked_image.size(2), masked_image.size(3))
        )
        mask = mask.to(device=device, dtype=dtype)

        batch_size = num_images_per_prompt

        masked_image = masked_image.to(device=device, dtype=dtype)

        masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        # masked_image_latents = (
        #     torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        # )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        strength: float = 0.6,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        guidance_rescale: float = 0.0,
        guidance_cutoff: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_pooled_projections: Optional[torch.FloatTensor] = None,

        mask_cutoff: float = 1.0,

        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        doDiffDiff = True if (image and mask_image) else False
        doInPaint = False if (image and mask_image) else False

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, SD3MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )


        # 0.01 repeat prompt embeds to match num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
        

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            strength,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._guidance_cutoff = guidance_cutoff
        self._mask_cutoff = mask_cutoff
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        device = self._execution_device
        dtype = self.transformer.dtype

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Prepare control image
        if isinstance(self.controlnet, SD3ControlNetModel):
            control_image = self.prepare_image(
                image=control_image,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,
            )
        elif isinstance(self.controlnet, SD3MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_image(
                    image=control_image_,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=False,
                )
                control_images.append(control_image_)

            control_image = control_images

        if self.controlnet != None:
            if controlnet_pooled_projections is None:
                controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
            else:
                controlnet_pooled_projections = controlnet_pooled_projections or pooled_prompt_embeds


        if image is not None:
            # 3. Preprocess image
            image = self.image_processor.preprocess(image)

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
            latent_timestep = timesteps[:1].repeat(num_images_per_prompt)# * num_inference_steps)

            # 5. Prepare latent variables
            latents, image_latents, noise = self.prepare_latents(
                image,
                latents,
                latent_timestep,
                strength, 
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

            if mask_image is not None:
                # 5.1. Prepare masked latent variables
                #### mask_image already resized /8 at start of predict()
                mask = self.mask_processor.preprocess(mask_image).to(device='cuda', dtype=torch.float16)
#                mask = mask.repeat(num_images_per_prompt, 1, 1, 1)  #necessary?


####    with real inpaint model:
####                mask_condition = self.mask_processor.preprocess(mask_image)
####                masked_image = image * (mask_condition < 0.5)
####                mask, masked_image_latents = self.prepare_mask_latents(
####                mask_condition, masked_image,
####                num_images_per_prompt,
####                prompt_embeds.dtype, device, generator )                
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 6a. Create tensor stating which controlnets to keep
        #is this necessary? why not use start/end directly?
        if self.controlnet != None:
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float((i+1) / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(self.controlnet, SD3ControlNetModel) else keeps)


        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if doDiffDiff and float((i+1) / self._num_timesteps) <= self._mask_cutoff:# and i > 0 :
                    tmask = mask >= float((i+1) / self._num_timesteps)
                    init_latents_proper = self.scheduler.scale_noise(image_latents, torch.tensor([t]), noise)
                    latents = (init_latents_proper * ~tmask) + (latents * tmask)

                if doInPaint and float((i+1) / self._num_timesteps) <= self._mask_cutoff:
                    init_latents_proper = self.scheduler.scale_noise(image_latents, torch.tensor([t]), noise)
                    latents = (init_latents_proper * (1 - mask)) + (latents * mask)


                if float((i+1) / len(timesteps)) > self._guidance_cutoff and self._guidance_scale != 1.0:
                    self._guidance_scale = 1.0
                    prompt_embeds = prompt_embeds[1]
                    pooled_prompt_embeds = pooled_prompt_embeds[1]
                    if self.controlnet != None:
                        controlnet_pooled_projections = controlnet_pooled_projections[1]
                        control_image = control_image[1].unsqueeze(0)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                ####    would be used by real inpainting model
####                if doInPaint:
####                    if num_channels_transformer == 33:
####                        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                if self.controlnet != None:
                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    # controlnet(s) inference
                    control_block_samples = self.controlnet(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=controlnet_pooled_projections,
                        joint_attention_kwargs=None,#self.joint_attention_kwargs,   #only check 'scale', default set to 1.0 - but scale used by LoRAs
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        return_dict=False,
                    )[0]
                else:
                    control_block_samples = None

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_samples,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if doInPaint:
            if mask_image is not None and 1.0 <= self._mask_cutoff:
                latents = (image_latents * (1 - mask)) + (latents * mask)

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
