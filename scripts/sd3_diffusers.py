#### THIS IS THE main BRANCH - deletes models after use / keep loaded now optional

#todo: check VRAM, different paths
#   low <= 6GB enable_sequential_model_offload() on pipe
#   medium 8-10/12? as is
#   high 16         - everything fully to GPU while running, CLIPs + transformer to cpu after use (T5 stay GPU)
#   very high 24+   - everything to GPU, noUnload (lock setting?)

class SD3Storage:
    ModuleReload = False
    lastSeed = -1
    galleryIndex = 0
    combined_positive = None
    combined_negative = None
    clipskip = 0
    redoEmbeds = True
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    lora = None
    lora_scale = 1.0
    LFO = False

    teT5 = None
    teCG = None
    teCL = None
    lastModel = None
    lastControlNet = None
    pipe = None
    loadedLora = False

    locked = False     #   for preventing changes to the following volatile state while generating
    noUnload = False
    useCL = True
    useCG = True
    useT5 = False
    ZN = False
    i2iAllSteps = False
    sharpNoise = False


import gc
import gradio
import math
import numpy
import os
import torch
import torchvision.transforms.functional as TF
try:
    import reload
    SD3Storage.ModuleReload = True
except:
    SD3Storage.ModuleReload = False

##   from webui
from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste

##   diffusers / transformers necessary imports
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, T5ForConditionalGeneration
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor

##  for Florence-2, including workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

##   my stuff
import customStylesListSD3 as styles
import scripts.SD3_pipeline as pipeline


# modules/processing.py
def create_infotext(model, positive_prompt, negative_prompt, guidance_scale, guidance_rescale, guidance_cutoff, shift, clipskip, steps, seed, width, height, loraSettings, controlNetSettings):
    generation_params = {
        "Size"          :   f"{width}x{height}",
        "Seed"          :   seed,
        "Steps"         :   steps,
        "CFG"           :   f"{guidance_scale} ({guidance_rescale}) [{guidance_cutoff}]",
        "Shift"         :   f"{shift}",
        "Clip skip"     :   f"{clipskip}",
        "LoRA"          :   loraSettings,
        "controlNet"    :   controlNetSettings,
        "CLIP-L"        :   '✓' if SD3Storage.useCL else '✗',
        "CLIP-G"        :   '✓' if SD3Storage.useCG else '✗',
        "T5"            :   '✓' if SD3Storage.useT5 else '✗', #2713, 2717
        "zero negative" :   '✓' if SD3Storage.ZN else '✗',
    }
#add loras list and scales

    prompt_text = f"Prompt: {positive_prompt}\n"
    prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    noise_text = f"\nInitial noise: {SD3Storage.noiseRGBA}" if SD3Storage.noiseRGBA[3] != 0.0 else ""

    return f"Model: StableDiffusion3m {model}\n{prompt_text}{generation_params_text}{noise_text}"

def predict(model, positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, guidance_cutoff, shift, clipskip, 
            num_steps, sampling_seed, num_images, style, i2iSource, i2iDenoise, maskType, maskSource, maskBlur, maskCutOff, 
            controlNet, controlNetImage, controlNetStrength, controlNetStart, controlNetEnd):
    try:
        with open('huggingface_access_token.txt', 'r') as file:
            access_token = file.read().strip()
    except:
        print ("SD3: couldn't load 'huggingface_access_token.txt' from the webui directory. Will not be able to download models. Local cache will work.")
        access_token = 0

    torch.set_grad_enabled(False)
    
    localFilesOnly = SD3Storage.LFO
    
    # do I care about catching this?
#    if SD3Storage.CL == False and SD3Storage.CG == False and SD3Storage.T5 == False:
        
    if controlNet != 0 and controlNetImage != None and controlNetStrength > 0.0:
        controlNetImage = controlNetImage.resize((width, height))
        useControlNet = ['InstantX/SD3-Controlnet-Canny', 'InstantX/SD3-Controlnet-Pose', 'InstantX/SD3-Controlnet-Tile'][controlNet-1]
    else:
        controlNetStrength = 0.0
        useControlNet = None

    ####    check img2img
    if i2iSource == None:
        maskType = 0
        i2iDenoise = 1
    if maskSource == None:
        maskType = 0
    if SD3Storage.i2iAllSteps == True:
        num_steps = int(num_steps / i2iDenoise)
        
    match maskType:
        case 0:     #   'none'
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0
        case 1:     #   'image'
            maskSource = maskSource['image']
        case 2:     #   'drawn'
            maskSource = maskSource['mask']
        case _:
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0

    if maskBlur > 0:
        maskSource = TF.gaussian_blur(maskSource, 1+2*maskBlur)
        
    ####    end check img2img

    #   triple prompt, automatic support, no longer needs button to enable
    def promptSplit (prompt):
        split_prompt = prompt.split('|')
        c = len(split_prompt)
        prompt_1 = split_prompt[0].strip()
        if c == 1:
            prompt_2 = prompt_1
            prompt_3 = prompt_1
        elif c == 2:
            if SD3Storage.useT5 == True:
                prompt_2 = prompt_1
                prompt_3 = split_prompt[1].strip()
            else:
                prompt_2 = split_prompt[1].strip()
                prompt_3 = ''
        elif c >= 3:
            prompt_2 = split_prompt[1].strip()
            prompt_3 = split_prompt[2].strip()
        return prompt_1, prompt_2, prompt_3

    positive_prompt_1, positive_prompt_2, positive_prompt_3 = promptSplit (positive_prompt)
    negative_prompt_1, negative_prompt_2, negative_prompt_3 = promptSplit (negative_prompt)

    if style != None:
        for s in style:
            k = 0;
            while styles.styles_list[k][0] != s:
                k += 1
            if "{prompt}" in styles.styles_list[k][1]:
                positive_prompt_1 = styles.styles_list[k][1].replace("{prompt}", positive_prompt_1)
                positive_prompt_2 = styles.styles_list[k][1].replace("{prompt}", positive_prompt_2)
                positive_prompt_3 = styles.styles_list[k][1].replace("{prompt}", positive_prompt_3)
            else:
                positive_prompt_1 += styles.styles_list[k][1]
                positive_prompt_2 += styles.styles_list[k][1]
                positive_prompt_3 += styles.styles_list[k][1]
            
    combined_positive = positive_prompt_1 + " | \n" + positive_prompt_2 + " | \n" + positive_prompt_3
    combined_negative = negative_prompt_1 + " | \n" + negative_prompt_2 + " | \n" + negative_prompt_3

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    SD3Storage.lastSeed = fixed_seed

    source = "stabilityai/stable-diffusion-3-medium-diffusers"

    useCachedEmbeds = (SD3Storage.combined_positive == combined_positive and
                       SD3Storage.combined_negative == combined_negative and
                       SD3Storage.redoEmbeds == False and
                       SD3Storage.clipskip == clipskip)
    #   also shouldn't cache if change model, but how to check if new model has own CLIPs?
    #   maybe just WON'T FIX, to keep it simple

    if useCachedEmbeds:
        print ("SD3: Skipping text encoders and tokenizers.")
    else:
        ####    start T5 text encoder
        if SD3Storage.useT5 == True:
            tokenizer = T5TokenizerFast.from_pretrained(
                source, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                subfolder='tokenizer_3',
                torch_dtype=torch.float16,
                max_length=512,
                use_auth_token=access_token,
            )

            input_ids = tokenizer(
                [positive_prompt_3, negative_prompt_3],          padding=True, max_length=512, truncation=True,
                add_special_tokens=True,    return_tensors="pt",
            ).input_ids

            # positive_input_ids = input_ids[0:1]
            # negative_input_ids = input_ids[1:]

            del tokenizer

            if SD3Storage.teT5 == None:             #   model not loaded
                if SD3Storage.noUnload == True:     #   will keep model loaded
                    device_map = {  #   how to find which blocks are most important? if any?
                        'shared': 0,
                        'encoder.embed_tokens': 0,
                        'encoder.block.0': 0,   'encoder.block.1': 0,   'encoder.block.2': 0,   'encoder.block.3': 0, 
                        'encoder.block.4': 'cpu',   'encoder.block.5': 'cpu',   'encoder.block.6': 'cpu',   'encoder.block.7': 'cpu', 
                        'encoder.block.8': 'cpu',   'encoder.block.9': 'cpu',   'encoder.block.10': 'cpu',  'encoder.block.11': 'cpu', 
                        'encoder.block.12': 'cpu',  'encoder.block.13': 'cpu',  'encoder.block.14': 'cpu',  'encoder.block.15': 'cpu', 
                        'encoder.block.16': 'cpu',  'encoder.block.17': 'cpu',  'encoder.block.18': 'cpu',  'encoder.block.19': 'cpu', 
                        'encoder.block.20': 'cpu',  'encoder.block.21': 'cpu',  'encoder.block.22': 'cpu',  'encoder.block.23': 'cpu', 
                        'encoder.final_layer_norm': 0, 
                        'encoder.dropout': 0
                    }
                else:                               #   will delete model after use
                    device_map = 'auto'

                print ("SD3: loading T5 ...", end="\r", flush=True)
                SD3Storage.teT5  = T5EncoderModel.from_pretrained(
                    source, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                    subfolder='text_encoder_3',
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    use_auth_token=access_token,
                )
            #   if model loaded, then switch off noUnload, loaded model still used (could alter device_map?: model.hf_device_map)
            #   not a major concern anyway

            print ("SD3: encoding prompt (T5) ...", end="\r", flush=True)
            embeds_3 = SD3Storage.teT5(input_ids.to('cuda'))[0]
            positive_embeds_3 = embeds_3[0].unsqueeze(0)
            if SD3Storage.ZN == True:
                negative_embeds_3 = torch.zeros_like(positive_embeds_3)
            else:
                negative_embeds_3 = embeds_3[1].unsqueeze(0)
                
            del input_ids, embeds_3
            
            # positive_embeds_3 = SD3Storage.teT5(positive_input_ids)[0]

            # if SD3Storage.ZN == True:
              # negative_embeds_3 = torch.zeros((1, 512, 4096),    device='cuda', dtype=torch.float16, )
                # negative_embeds_3 = torch.zeros_like(positive_embeds_3)
            # else:
                # negative_embeds_3 = SD3Storage.teT5(negative_input_ids)[0]

            if SD3Storage.noUnload == False:
                SD3Storage.teT5 = None
            print ("SD3: encoding prompt (T5) ... done")
        else:
            #dim 1 (512) is tokenizer max length from config; dim 2 (4096) is transformer joint_attention_dim from its config
            #why max_length - it's not used, so make it small (or None)
            positive_embeds_3 = torch.zeros((1, 1, 4096),    device='cuda', dtype=torch.float16, )
            negative_embeds_3 = torch.zeros((1, 1, 4096),    device='cuda', dtype=torch.float16, )
        ####    end T5

        ####    start CLIP-G
        if SD3Storage.useCG == True:
            tokenizer = CLIPTokenizer.from_pretrained(
                source, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                subfolder='tokenizer',
                torch_dtype=torch.float16,
                use_auth_token=access_token,
            )

            input_ids = tokenizer(
                [positive_prompt_1, negative_prompt_1],          padding='max_length', max_length=77, truncation=True,
                return_tensors="pt",
            ).input_ids

            positive_input_ids = input_ids[0:1]
            negative_input_ids = input_ids[1:]

            del tokenizer
            
            #   check if custom model has trained CLIPs
            if model != SD3Storage.lastModel:
                if model == '(base)':
                    SD3Storage.teCG = None
                else:
                    try:    #   maybe custom model has trained CLIPs - not sure if correct way to load
                        SD3Storage.teCG = CLIPTextModelWithProjection.from_single_file(
                            model, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                            subfolder='text_encoder',
                            torch_dtype=torch.float16,
                            use_auth_token=access_token,
                        )
                    except:
                        SD3Storage.teCG = None
            if SD3Storage.teCG == None:             #   model not loaded, use base
                SD3Storage.teCG = CLIPTextModelWithProjection.from_pretrained(
                    source, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                    subfolder='text_encoder',
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    use_auth_token=access_token,
                )

            SD3Storage.teCG.to('cuda')

            positive_embeds = SD3Storage.teCG(positive_input_ids.to('cuda'), output_hidden_states=True)
            pooled_positive_1 = positive_embeds[0]
            positive_embeds_1 = positive_embeds.hidden_states[-(clipskip + 2)]
            
            if SD3Storage.ZN == True:
                negative_embeds_1 = torch.zeros_like(positive_embeds_1)
                pooled_negative_1 = torch.zeros((1, 768),       device='cuda', dtype=torch.float16, )
            else:
                negative_embeds = SD3Storage.teCG(negative_input_ids.to('cuda'), output_hidden_states=True)
                pooled_negative_1 = negative_embeds[0]
                negative_embeds_1 = negative_embeds.hidden_states[-2]

            if SD3Storage.noUnload == False:
                SD3Storage.teCG = None
            else:
                SD3Storage.teCG.to('cpu')
                
        else:
            positive_embeds_1 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            negative_embeds_1 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            pooled_positive_1 = torch.zeros((1, 768),      device='cuda', dtype=torch.float16, )
            pooled_negative_1 = torch.zeros((1, 768),      device='cuda', dtype=torch.float16, )
        ####    end CLIP-G

        ####    start CLIP-L
        if SD3Storage.useCL == True:
            tokenizer = CLIPTokenizer.from_pretrained(
                source, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                subfolder='tokenizer_2',
                torch_dtype=torch.float16,
                use_auth_token=access_token,
            )
            input_ids = tokenizer(
                [positive_prompt_2, negative_prompt_2],          padding='max_length', max_length=77, truncation=True,
                return_tensors="pt",
            ).input_ids

            positive_input_ids = input_ids[0:1]
            negative_input_ids = input_ids[1:]

            del tokenizer

            #   check if custom model has trained CLIPs
            if model != SD3Storage.lastModel:
                if model == '(base)':
                    SD3Storage.teCL = None
                else:
                    try:    #   maybe custom model has trained CLIPs - not sure if correct way to load
                        SD3Storage.teCL = CLIPTextModelWithProjection.from_single_file(
                            model, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                            subfolder='text_encoder_2',
                            torch_dtype=torch.float16,
                            use_auth_token=access_token,
                        )
                    except:
                        SD3Storage.teCL = None
            if SD3Storage.teCL == None:             #   model not loaded, use base
                SD3Storage.teCL = CLIPTextModelWithProjection.from_pretrained(
                    source, local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                    subfolder='text_encoder_2',
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    use_auth_token=access_token,
                )

            SD3Storage.teCL.to('cuda')

            positive_embeds = SD3Storage.teCL(positive_input_ids.to('cuda'), output_hidden_states=True)
            pooled_positive_2 = positive_embeds[0]
            positive_embeds_2 = positive_embeds.hidden_states[-(clipskip + 2)]
            
            if SD3Storage.ZN == True:
                negative_embeds_2 = torch.zeros_like(positive_embeds_2)
                pooled_negative_2 = torch.zeros((1, 1280),      device='cuda', dtype=torch.float16, )
            else:
                negative_embeds = SD3Storage.teCL(negative_input_ids.to('cuda'), output_hidden_states=True)
                pooled_negative_2 = negative_embeds[0]
                negative_embeds_2 = negative_embeds.hidden_states[-2]

            if SD3Storage.noUnload == False:
                SD3Storage.teCL = None
            else:
                SD3Storage.teCL.to('cpu')

        else:
            positive_embeds_2 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            negative_embeds_2 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            pooled_positive_2 = torch.zeros((1, 1280),     device='cuda', dtype=torch.float16, )
            pooled_negative_2 = torch.zeros((1, 1280),     device='cuda', dtype=torch.float16, )
        ####    end CLIP-L

        #merge
        clip_positive_embeds = torch.cat([positive_embeds_1, positive_embeds_2], dim=-1)
        clip_positive_embeds = torch.nn.functional.pad(clip_positive_embeds, (0, positive_embeds_3.shape[-1] - clip_positive_embeds.shape[-1]) )
        clip_negative_embeds = torch.cat([negative_embeds_1, negative_embeds_2], dim=-1)
        clip_negative_embeds = torch.nn.functional.pad(clip_negative_embeds, (0, negative_embeds_3.shape[-1] - clip_negative_embeds.shape[-1]) )

        positive_embeds = torch.cat([clip_positive_embeds, positive_embeds_3.to('cuda')], dim=-2)
        negative_embeds = torch.cat([clip_negative_embeds, negative_embeds_3.to('cuda')], dim=-2)

        positive_pooled = torch.cat([pooled_positive_1, pooled_positive_2], dim=-1)
        negative_pooled = torch.cat([pooled_negative_1, pooled_negative_2], dim=-1)

        SD3Storage.positive_embeds = positive_embeds.to('cpu')
        SD3Storage.negative_embeds = negative_embeds.to('cpu')
        SD3Storage.positive_pooled = positive_pooled.to('cpu')
        SD3Storage.negative_pooled = negative_pooled.to('cpu')
        SD3Storage.combined_positive = combined_positive
        SD3Storage.combined_negative = combined_negative
        SD3Storage.clipskip = clipskip
        SD3Storage.redoEmbeds = False

        del positive_embeds, negative_embeds, positive_pooled, negative_pooled
        del clip_positive_embeds, clip_negative_embeds
        del pooled_positive_1, pooled_positive_2, pooled_negative_1, pooled_negative_2
        del positive_embeds_1, positive_embeds_2, positive_embeds_3
        del negative_embeds_1, negative_embeds_2, negative_embeds_3

        gc.collect()
        torch.cuda.empty_cache()

    ####    end useCachedEmbeds

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(source,
                                                                subfolder='scheduler', local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                                                                shift=shift,
                                                                token=access_token,
                                                                )
    if useControlNet:
        if useControlNet != SD3Storage.lastControlNet:
            controlnet=SD3ControlNetModel.from_pretrained(
                useControlNet, cache_dir=".//models//diffusers//", torch_dtype=torch.float16)
    else:
        controlnet = None

    if SD3Storage.pipe == None:
        if model == '(base)':
            SD3Storage.pipe = pipeline.SD3Pipeline_DoE_combined.from_pretrained(
                source,
                local_files_only=localFilesOnly, cache_dir=".//models//diffusers//",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,                
                use_safetensors=True,
                scheduler=scheduler,
                token=access_token,
                controlnet=controlnet
            )
        else:
            customModel = './/models//diffusers//SD3Custom//' + model + '.safetensors'
            SD3Storage.pipe = pipeline.SD3Pipeline_DoE_combined.from_pretrained(
                source,
                local_files_only=True, cache_dir=".//models//diffusers//",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,                
                use_safetensors=True,
                transformer=SD3Transformer2DModel.from_single_file(customModel, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16),
                scheduler=scheduler,
                token=access_token,
                controlnet=controlnet
            )
        SD3Storage.lastModel = model
#        if SD3Storage.noUnload:        #   use for low VRAM only
#            SD3Storage.pipe.enable_sequential_cpu_offload()
#        else:
#            SD3Storage.pipe.to('cuda')

        SD3Storage.pipe.enable_model_cpu_offload()
        SD3Storage.pipe.vae.to(memory_format=torch.channels_last)
    else:
        SD3Storage.pipe.scheduler = scheduler
        if SD3Storage.noUnload:
            if useControlNet != SD3Storage.lastControlNet:
                SD3Storage.pipe.controlnet = controlnet
                SD3Storage.lastControlNet = useControlNet

    del scheduler, controlnet

    if model != SD3Storage.lastModel:
        del SD3Storage.pipe.transformer
        if model == '(base)':
            SD3Storage.pipe.transformer=SD3Transformer2DModel.from_pretrained(
                source,
                subfolder='transformer',
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            )
        else:
            customModel = './/models//diffusers//SD3Custom//' + model + '.safetensors'
            SD3Storage.pipe.transformer=SD3Transformer2DModel.from_single_file(
                customModel,
                local_files_only=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            
        SD3Storage.model = model
#        if SD3Storage.noUnload:    #for very low VRAM only
#            SD3Storage.pipe.enable_sequential_cpu_offload()
        SD3Storage.pipe.enable_model_cpu_offload()

    SD3Storage.pipe.transformer.to(memory_format=torch.channels_last)

    shape = (
        num_images,
        SD3Storage.pipe.transformer.config.in_channels,
        int(height) // SD3Storage.pipe.vae_scale_factor,
        int(width) // SD3Storage.pipe.vae_scale_factor,
    )

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    latents = randn_tensor(shape, generator=generator).to('cuda').to(torch.float16)

    if SD3Storage.sharpNoise:
        minDim = 1 + 2*(min(latents.size(2), latents.size(3)) // 4)
        for b in range(len(latents)):
            blurred = TF.gaussian_blur(latents[b], minDim)
            latents[b] = 1.05*latents[b] - 0.05*blurred

    #regen the generator to minimise differences between single/batch - might still be different - batch processing could use different pytorch kernels
    del generator
    generator = torch.Generator(device='cpu').manual_seed(14641)

    #   colour the initial noise
    if SD3Storage.noiseRGBA[3] != 0.0:
        nr = SD3Storage.noiseRGBA[0] ** 0.5
        ng = SD3Storage.noiseRGBA[1] ** 0.5
        nb = SD3Storage.noiseRGBA[2] ** 0.5

        imageR = torch.tensor(numpy.full((8,8), (nr), dtype=numpy.float32))
        imageG = torch.tensor(numpy.full((8,8), (ng), dtype=numpy.float32))
        imageB = torch.tensor(numpy.full((8,8), (nb), dtype=numpy.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)

        image = SD3Storage.pipe.image_processor.preprocess(image).to('cuda').to(torch.float16)
        image_latents = (SD3Storage.pipe.vae.encode(image).latent_dist.sample(generator) - SD3Storage.pipe.vae.config.shift_factor) * SD3Storage.pipe.vae.config.scaling_factor

        image_latents = image_latents.repeat(num_images, 1, latents.size(2), latents.size(3))

        for b in range(len(latents)):
            for c in range(4):
                latents[b][c] -= latents[b][c].mean()

        torch.lerp (latents, image_latents, SD3Storage.noiseRGBA[3], out=latents)

        del imageR, imageG, imageB, image, image_latents
    #   end: colour the initial noise


#   load in LoRA, weight passed to pipe
    if SD3Storage.lora and SD3Storage.lora != "(None)" and SD3Storage.lora_scale != 0.0:
        lorafile = ".//models/diffusers//SD3Lora//" + SD3Storage.lora + ".safetensors"
        try:
            SD3Storage.pipe.load_lora_weights(lorafile, local_files_only=True, adapter_name=SD3Storage.lora)
            SD3Storage.loadedLora = True
#            pipe.set_adapters(SD3Storage.lora, adapter_weights=SD3Storage.lora_scale)    #.set_adapters doesn't exist so no easy multiple LoRAs and weights
        except:
            print ("Failed: LoRA: " + lorafile)
            #   no reason to abort, just carry on without LoRA

#adapter_weight_scales = { "unet": { "down": 1, "mid": 0, "up": 0} }
#pipe.set_adapters("pixel", adapter_weight_scales)
#pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])

#    print (pipe.scheduler.compatibles)

    SD3Storage.pipe.transformer.to(memory_format=torch.channels_last)
    SD3Storage.pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = SD3Storage.pipe(
            num_inference_steps             = num_steps,
            guidance_scale                  = guidance_scale,
            guidance_rescale                = guidance_rescale,
            guidance_cutoff                 = guidance_cutoff,
            prompt_embeds                   = SD3Storage.positive_embeds.to('cuda'),
            negative_prompt_embeds          = SD3Storage.negative_embeds.to('cuda'),
            pooled_prompt_embeds            = SD3Storage.positive_pooled.to('cuda'),
            negative_pooled_prompt_embeds   = SD3Storage.negative_pooled.to('cuda'),
            num_images_per_prompt           = num_images,
            generator                       = generator,
            latents                         = latents,

            image                           = i2iSource,
            strength                        = i2iDenoise,
            mask_image                      = maskSource,
            mask_cutoff                     = maskCutOff,

            control_image                   = controlNetImage, 
            controlnet_conditioning_scale   = controlNetStrength,  
            control_guidance_start          = controlNetStart,
            control_guidance_end            = controlNetEnd,
            
            joint_attention_kwargs          = {"scale": SD3Storage.lora_scale }
        )
        del controlNetImage, i2iSource

    del generator, latents

    if SD3Storage.noUnload:
        if SD3Storage.loadedLora == True:
            SD3Storage.pipe.unload_lora_weights()
            SD3Storage.loadedLora = False
        SD3Storage.pipe.transformer.to('cpu')
#        SD3Storage.pipe.controlnet.to('cpu')
    else:
        SD3Storage.pipe.transformer = None
        SD3Storage.pipe.controlnet = None

    gc.collect()
    torch.cuda.empty_cache()

#    SD3Storage.pipe.vae.enable_slicing()       #   tiling works once only?

    if SD3Storage.lora != "(None)" and SD3Storage.lora_scale != 0.0:
        loraSettings = SD3Storage.lora + f" ({SD3Storage.lora_scale})"
    else:
        loraSettings = None

    if useControlNet != None:
        useControlNet += f" strength: {controlNetStrength}; step range: {controlNetStart}-{controlNetEnd}"

    result = []
    total = len(output)
    for i in range (total):
        print (f'SD3: VAE: {i+1} of {total}', end='\r', flush=True)
        info=create_infotext(
            model, combined_positive, combined_negative, 
            guidance_scale, guidance_rescale, guidance_cutoff, shift, clipskip, num_steps, 
            fixed_seed + i, 
            width, height,
            loraSettings,
            useControlNet)      #   doing this for every image when only change is fixed_seed

        #   manually handling the VAE prevents hitting shared memory on 8GB
        latent = (output[i:i+1]) / SD3Storage.pipe.vae.config.scaling_factor
        latent = latent + SD3Storage.pipe.vae.config.shift_factor
        image = SD3Storage.pipe.vae.decode(latent, return_dict=False)[0]
        image = SD3Storage.pipe.image_processor.postprocess(image, output_type='pil')[0]

        result.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed + i,
            combined_positive,
            opts.samples_format,
            info
        )
    print ('SD3: VAE: done  ')

    if not SD3Storage.noUnload:
        SD3Storage.pipe = None
    
    del output
    gc.collect()
    torch.cuda.empty_cache()

    SD3Storage.locked = False
    return gradio.Button.update(value='Generate', variant='primary', interactive=True), gradio.Button.update(interactive=True), result


def on_ui_tabs():
    if SD3Storage.ModuleReload:
        reload(styles)
        reload(pipeline)

    def buildLoRAList ():
        loras = ["(None)"]
        
        import glob
        customLoRA = glob.glob(".\models\diffusers\SD3Lora\*.safetensors")

        for i in customLoRA:
            filename = i.split('\\')[-1]
            loras.append(filename[0:-12])

        return loras
    def buildModelList ():
        models = ["(base)"]
        
        import glob
        customModel = glob.glob(".\models\diffusers\SD3Custom\*.safetensors")

        for i in customModel:
            filename = i.split('\\')[-1]
            models.append(filename[0:-12])

        return models

    loras = buildLoRAList ()
    models = buildModelList ()

    def refreshLoRAs ():
        loras = buildLoRAList ()
        return gradio.Dropdown.update(choices=loras)
    def refreshModels ():
        models = buildModelList ()
        return gradio.Dropdown.update(choices=models)
   
    def getGalleryIndex (evt: gradio.SelectData):
        SD3Storage.galleryIndex = evt.index

    def reuseLastSeed ():
        return SD3Storage.lastSeed + SD3Storage.galleryIndex
        
    def randomSeed ():
        return -1

    def i2iSetDimensions (image, w, h):
        if image is not None:
            w = 32 * (image.size[0] // 32)
            h = 32 * (image.size[1] // 32)
        return [w, h]

    def i2iImageFromGallery (gallery):
        try:
            newImage = gallery[SD3Storage.galleryIndex][0]['name'].split('?')
            return newImage[0]
        except:
            return None

    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports
    def i2iMakeCaptions (image, originalPrompt):
        if image == None:
            return originalPrompt

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', 
                                                         attn_implementation="sdpa", 
                                                         torch_dtype=torch.float16, 
                                                         cache_dir=".//models//diffusers//", 
                                                         trust_remote_code=True).to('cuda')
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  cache_dir=".//models//diffusers//", 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image, return_tensors="pt")
            inputs.to('cuda').to(torch.float16)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            del inputs
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            del generated_ids
            parsed_answer = processor.post_process_generation(generated_text, task=p, image_size=(image.width, image.height))
            del generated_text
            print (parsed_answer)
            result += parsed_answer[p]
            del parsed_answer
            if p != prompts[-1]:
                result += ' | \n'

        del model, processor

        if SD3Storage.captionToPrompt:
            return result
        else:
            return originalPrompt
    def toggleC2P ():
        SD3Storage.captionToPrompt ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.captionToPrompt])
    def toggleLFO ():
        SD3Storage.LFO ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.LFO])

    #   these are volatile state, should not be changed during generation
    def toggleNU ():
        if not SD3Storage.locked:
            SD3Storage.noUnload ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.noUnload])
    def unloadM ():
        if not SD3Storage.locked:
            SD3Storage.teT5 = None
            SD3Storage.teCG = None
            SD3Storage.teCL = None
            SD3Storage.pipe = None
            SD3Storage.lastModel = None
            SD3Storage.lastControlNet = None
            gc.collect()
            torch.cuda.empty_cache()
        else:
            gradio.Info('Unable to unload models while using them.')

    def toggleCL ():
        if not SD3Storage.locked:
            SD3Storage.redoEmbeds = True
            SD3Storage.useCL ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.useCL])
    def toggleCG ():
        if not SD3Storage.locked:
            SD3Storage.redoEmbeds = True
            SD3Storage.useCG ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.useCG])
    def toggleT5 ():
        if not SD3Storage.locked:
            SD3Storage.redoEmbeds = True
            SD3Storage.useT5 ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.useT5])
    def toggleZN ():
        if not SD3Storage.locked:
            SD3Storage.redoEmbeds = True
            SD3Storage.ZN ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.ZN])
    def toggleAS ():
        if not SD3Storage.locked:
            SD3Storage.i2iAllSteps ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD3Storage.i2iAllSteps])
    def toggleSP ():
        if not SD3Storage.locked:
            return gradio.Button.update(variant='primary')
    def superPrompt (prompt, seed):
        tokenizer = getattr (shared, 'SuperPrompt_tokenizer', None)
        superprompt = getattr (shared, 'SuperPrompt_model', None)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(
                'roborovski/superprompt-v1',
                cache_dir='.//models//diffusers//',
            )
            shared.SuperPrompt_tokenizer = tokenizer
        if superprompt is None:
            superprompt = T5ForConditionalGeneration.from_pretrained(
                'roborovski/superprompt-v1',
                cache_dir='.//models//diffusers//',
                device_map='auto',
                torch_dtype=torch.float16
            )
            shared.SuperPrompt_model = superprompt
            print("SuperPrompt-v1 model loaded successfully.")
            if torch.cuda.is_available():
                superprompt.to('cuda')
 
        torch.manual_seed(get_fixed_seed(seed))
        device = superprompt.device
        systemprompt1 = "Expand the following prompt to add more detail: "
        
        input_ids = tokenizer(systemprompt1 + prompt, return_tensors="pt").input_ids.to(device)
        outputs = superprompt.generate(input_ids, max_new_tokens=256, repetition_penalty=1.2, do_sample=True)
        dirty_text = tokenizer.decode(outputs[0])
        result = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        return gradio.Button.update(variant='secondary'), result

    resolutionList = [
        (1536, 672),    (1344, 768),    (1248, 832),    (1120, 896),
        (1200, 1200),   (1024, 1024),
        (896, 1120),    (832, 1248),    (768, 1344),    (672, 1536)
    ]

    def updateWH (idx, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        if idx < len(resolutionList):
            return None, resolutionList[idx][0], resolutionList[idx][1]
        return None, w, h

    def randomString ():
        import random
        import string
        alphanumeric_string = ''
        for i in range(8):
            alphanumeric_string += ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            if i < 7:
                alphanumeric_string += ' '
        return alphanumeric_string

    def toggleGenerate (R, G, B, A, lora, scale):
        SD3Storage.noiseRGBA = [R, G, B, A]
        SD3Storage.lora = lora
        SD3Storage.lora_scale = scale# if lora != "(None)" else 1.0
        SD3Storage.locked = True
        return gradio.Button.update(value='...', variant='secondary', interactive=False), gradio.Button.update(interactive=False)


    def parsePrompt (positive, negative, width, height, seed, steps, CFG, CFGrescale, CFGcutoff, shift, nr, ng, nb, ns, loraName, loraScale):
        p = positive.split('\n')
        lineCount = len(p)

        negative = ''
        
        if "Prompt" != p[0] and "Prompt: " != p[0][0:8]:               #   civitAI style special case
            positive = p[0]
            l = 1
            while (l < lineCount) and not (p[l][0:17] == "Negative prompt: " or p[l][0:7] == "Steps: " or p[l][0:6] == "Size: "):
                if p[l] != '':
                    positive += '\n' + p[l]
                l += 1
        
        for l in range(lineCount):
            if "Prompt" == p[l][0:6]:
                if ": " == p[l][6:8]:                                   #   mine
                    positive = str(p[l][8:])
                    c = 1
                elif "Prompt" == p[l] and (l+1 < lineCount):            #   webUI
                    positive = p[l+1]
                    c = 2
                else:
                    continue

                while (l+c < lineCount) and not (p[l+c][0:10] == "Negative: " or p[l+c][0:15] == "Negative Prompt" or p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        positive += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Negative" == p[l][0:8]:
                if ": " == p[l][8:10]:                                  #   mine
                    negative = str(p[l][10:])
                    c = 1
                elif " prompt: " == p[l][8:17]:                         #   civitAI
                    negative = str(p[l][17:])
                    c = 1
                elif " Prompt" == p[l][8:15] and (l+1 < lineCount):     #   webUI
                    negative = p[l+1]
                    c = 2
                else:
                    continue
                
                while (l+c < lineCount) and not (p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        negative += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Initial noise: " == str(p[l][0:15]):
                noiseRGBA = str(p[l][16:-1]).split(',')
                nr = float(noiseRGBA[0])
                ng = float(noiseRGBA[1])
                nb = float(noiseRGBA[2])
                ns = float(noiseRGBA[3])
            else:
                params = p[l].split(',')
                for k in range(len(params)):
                    pairs = params[k].strip().split(' ')        #split on ':' instead?
                    match pairs[0]:
                        case "Size:":
                            size = pairs[1].split('x')
                            width = int(size[0])
                            height = int(size[1])
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            steps = int(steps[0])
                        case "Steps:":
                            steps = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                CFG = float(pairs[2])
                        case "CFG:":
                            CFG = float(pairs[1])
                            if len(pairs) == 4:
                                CFGrescale = float(pairs[2].strip('\(\)'))
                                CFGcutoff = float(pairs[3].strip('\[\]'))
                        case "Shift:":
                            shift = float(pairs[1])
                        case "width:":
                            width = float(pairs[1])
                        case "height:":
                            height = float(pairs[1])
                        case "LoRA:":
                            if len(pairs) == 3:
                                loraName = pairs[1]
                                loraScale = float(pairs[2].strip('\(\)'))
                            
                        #clipskip?
        return positive, negative, width, height, seed, steps, CFG, CFGrescale, CFGcutoff, shift, nr, ng, nb, ns, loraName, loraScale

    def style2prompt (prompt, style):
        splitPrompt = prompt.split('|')
        newPrompt = ''
        for p in splitPrompt:
            subprompt = p.strip()
            for s in style:
                #get index from value, working around possible gradio bug
                k = 0;
                while styles.styles_list[k][0] != s:
                    k += 1
                if "{prompt}" in styles.styles_list[k][1]:
                    subprompt = styles.styles_list[k][1].replace("{prompt}", subprompt)
                else:
                    subprompt += styles.styles_list[k][1]
            newPrompt += subprompt
            if p != splitPrompt[-1]:
                newPrompt += ' |\n'
        return newPrompt, []


    def refreshStyles (style):
        if SD3Storage.ModuleReload:
            reload(styles)
        
            newList = [x[0] for x in styles.styles_list]
            newStyle = []
        
            for s in style:
                if s in newList:
                    newStyle.append(s)

            return gradio.Dropdown.update(choices=newList, value=newStyle)
        else:
            return gradio.Dropdown.update(value=style)
            

    def toggleSharp ():
        if not SD3Storage.locked:
            SD3Storage.sharpNoise ^= True
        return gradio.Button.update(value=['s', 'S'][SD3Storage.sharpNoise],
                                variant=['secondary', 'primary'][SD3Storage.sharpNoise])

    def maskFromImage (image):
        if image:
            return image, 'drawn'
        else:
            return None, 'none'

    with gradio.Blocks() as sd3_block:
        with ResizeHandleRow():
            with gradio.Column():
#                    LFO = ToolButton(value='lfo', variant='secondary', tooltip='local files only')

                with gradio.Row():
                    model = gradio.Dropdown(models, label='Model', value='(base)', type='value')
                    refreshM = ToolButton(value='\U0001f504')
                    nouse0 = ToolButton(value="️|", variant='tertiary', tooltip='', interactive=False)
                    CL = ToolButton(value='CL', variant='primary',   tooltip='use CLIP-L text encoder')
                    CG = ToolButton(value='CG', variant='primary',   tooltip='use CLIP-G text encoder')
                    T5 = ToolButton(value='T5', variant='secondary', tooltip='use T5 text encoder')
                    nouseC = ToolButton(value="️|", variant='tertiary', tooltip='', interactive=False)
                    ZN = ToolButton(value='ZN', variant='secondary', tooltip='zero out negative embeds')
                    nouseB = ToolButton(value="️|", variant='tertiary', tooltip='', interactive=False)
                    clipskip = gradio.Number(label='CLIP skip', minimum=0, maximum=8, step=1, value=0, precision=0, scale=0)

                with gradio.Row():
                    positive_prompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here ...', default='', lines=1.01)
                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")
                    SP = ToolButton(value='ꌗ', variant='secondary', tooltip='zero out negative embeds')
                with gradio.Row():
                    negative_prompt = gradio.Textbox(label='Negative', placeholder='', lines=1.01)
                    randNeg = ToolButton(value='rng', variant='secondary', tooltip='random negative')

                with gradio.Row():
                    style = gradio.Dropdown([x[0] for x in styles.styles_list], label='Style', value=None, type='value', multiselect=True)
                    strfh = ToolButton(value="🔄", variant='secondary', tooltip='reload styles')
                    st2pr = ToolButton(value="📋", variant='secondary', tooltip='add style to prompt')
#make infotext from all settings, send to clipboard?

                with gradio.Row():
                    width = gradio.Slider(label='Width', minimum=512, maximum=2048, step=32, value=1024, elem_id='StableDiffusion3_width')
                    swapper = ToolButton(value='\U000021C4')
                    height = gradio.Slider(label='Height', minimum=512, maximum=2048, step=32, value=1024, elem_id='StableDiffusion3_height')
                    dims = gradio.Dropdown([f'{i} \u00D7 {j}' for i,j in resolutionList],
                                        label='Quickset', type='index', scale=0)

                with gradio.Row():
                    guidance_scale = gradio.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=5, scale=1)
                    CFGrescale = gradio.Slider(label='rescale CFG', minimum=0.00, maximum=1.0, step=0.01, value=0.0, precision=0.01, scale=1)
                    CFGcutoff = gradio.Slider(label='CFG cutoff step', minimum=0.00, maximum=1.0, step=0.01, value=1.0, precision=0.01, scale=1)
                    shift = gradio.Slider(label='Shift', minimum=1.0, maximum=8.0, step=0.1, value=3.0, scale=1)
                with gradio.Row(equal_height=True):
                    steps = gradio.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                    sampling_seed = gradio.Number(label='Seed', value=-1, precision=0, scale=0)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gradio.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gradio.Row(equal_height=True):
                    lora = gradio.Dropdown([x for x in loras], label='LoRA (place in models/diffusers/SD3Lora)', value="(None)", type='value', multiselect=False, scale=1)
                    refreshL = ToolButton(value='\U0001f504')
                    scale = gradio.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01, scale=1)

                with gradio.Accordion(label='the colour of noise', open=False):
                    with gradio.Row():
                        initialNoiseR = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gradio.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')
                        sharpNoise = ToolButton(value="s", variant='secondary', tooltip='Sharpen initial noise')

                with gradio.Accordion(label='ControlNet', open=False):
                    with gradio.Row():
                        CNSource = gradio.Image(label='control image', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gradio.Column():
                            CNMethod = gradio.Dropdown(['(None)', 'canny', 'pose', 'tile'], label='method', value='(None)', type='index', multiselect=False, scale=1)
                            CNStrength = gradio.Slider(label='Strength', minimum=0.00, maximum=1.0, step=0.01, value=0.8)
                            CNStart = gradio.Slider(label='Start step', minimum=0.00, maximum=1.0, step=0.01, value=0.0)
                            CNEnd = gradio.Slider(label='End step', minimum=0.00, maximum=1.0, step=0.01, value=0.8)

                with gradio.Accordion(label='image to image', open=False):
                    with gradio.Row():
                        i2iSource = gradio.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        maskSource = gradio.Image(label='source mask', sources=['upload'], type='pil', interactive=True, show_download_button=False, tool='sketch', image_mode='RGB', brush_color='#F0F0F0')#opts.img2img_inpaint_mask_brush_color)
                    with gradio.Row():
                        with gradio.Column():
                            with gradio.Row():
                                i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                AS = ToolButton(value='AS')
                            with gradio.Row():
                                i2iFromGallery = gradio.Button(value='Get gallery image')
                                i2iSetWH = gradio.Button(value='Set size from image')
                            with gradio.Row():
                                i2iCaption = gradio.Button(value='Caption image (Florence-2)', scale=6)
                                toPrompt = ToolButton(value='P', variant='secondary')

                        with gradio.Column():
                            maskType = gradio.Dropdown(['none', 'image', 'drawn'], value='none', label='Mask', type='index')
                            maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                            maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=25, step=1, value=0)
                            maskCopy = gradio.Button(value='use i2i source as template')

                with gradio.Row():
                    noUnload = gradio.Button(value='keep models loaded', variant='primary' if SD3Storage.noUnload else 'secondary', tooltip='noUnload', scale=1)
                    unloadModels = gradio.Button(value='unload models', tooltip='force unload of models', scale=1)

                ctrls = [model, positive_prompt, negative_prompt, width, height, guidance_scale, CFGrescale, CFGcutoff, shift, clipskip, steps, sampling_seed, batch_size, style, i2iSource, i2iDenoise, maskType, maskSource, maskBlur, maskCut, CNMethod, CNSource, CNStrength, CNStart, CNEnd]
                parseable = [positive_prompt, negative_prompt, width, height, sampling_seed, steps, guidance_scale, CFGrescale, CFGcutoff, shift, initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale]

            with gradio.Column():
                generate_button = gradio.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gradio.Gallery(label='Output', height="80vh",
                                            show_label=False, object_fit='contain', visible=True, columns=1, preview=True)
#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks, alt text does

                with gradio.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=positive_prompt,
                        source_image_component=output_gallery,
                    ))
        noUnload.click(toggleNU, inputs=[], outputs=noUnload)
        unloadModels.click(unloadM, inputs=[], outputs=[], show_progress=True)
                    
        SP.click(toggleSP, inputs=[], outputs=SP)
        SP.click(superPrompt, inputs=[positive_prompt, sampling_seed], outputs=[SP, positive_prompt])
        maskCopy.click(fn=maskFromImage, inputs=[i2iSource], outputs=[maskSource, maskType])
        sharpNoise.click(toggleSharp, inputs=[], outputs=sharpNoise)
        strfh.click(refreshStyles, inputs=[style], outputs=[style])
        st2pr.click(style2prompt, inputs=[positive_prompt, style], outputs=[positive_prompt, style])
        parse.click(parsePrompt, inputs=parseable, outputs=parseable, show_progress=False)
        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        refreshM.click(refreshModels, inputs=[], outputs=[model])
        refreshL.click(refreshLoRAs, inputs=[], outputs=[lora])
        CL.click(toggleCL, inputs=[], outputs=CL)
        CG.click(toggleCG, inputs=[], outputs=CG)
        T5.click(toggleT5, inputs=[], outputs=T5)
        ZN.click(toggleZN, inputs=[], outputs=ZN)
        AS.click(toggleAS, inputs=[], outputs=AS)
#        LFO.click(toggleLFO, inputs=[], outputs=LFO)
        swapper.click(fn=None, _js="function(){switchWidthHeight('StableDiffusion3')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        randNeg.click(randomString, inputs=[], outputs=[negative_prompt])

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])
        i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])#outputs=[positive_prompt]
        toPrompt.click(toggleC2P, inputs=[], outputs=[toPrompt])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, SP, output_gallery])
        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale], outputs=[generate_button, SP])

    return [(sd3_block, "StableDiffusion3", "sd3")]

script_callbacks.on_ui_tabs(on_ui_tabs)

