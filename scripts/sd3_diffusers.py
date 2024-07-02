import math
import torch
import gc
import json
import numpy as np
import os

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste
import gradio as gr

from PIL import Image
#workaround for unnecessary flash_attn requirement for Florence-2
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 


#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(True)


#   when control pipeline released, aim to cobine all three into
#   current image2image use of latents input is moronic

import customStylesListSD3 as styles

class SD3Storage:
    lastSeed = -1
    galleryIndex = 0
    combined_positive = None
    combined_negative = None
    positive_embeds = None
    negative_embeds = None
    positive_pooled = None
    negative_pooled = None
    clipskip = 0
    CL = True
    CG = True
    T5 = False
    ZN = False
    i2iAllSteps = False
    redoEmbeds = True
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    lora = None
    lora_scale = 1.0

#from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline#, StableDiffusion3ControlNetPipeline
from diffusers import FlowMatchEulerDiscreteScheduler

from scripts.SD3_pipeline import SD3Pipeline_DoE_combined
from diffusers.models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel

from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.utils.torch_utils import randn_tensor

import argparse
import pathlib
from pathlib import Path
import sys

current_file_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_file_path))


# modules/infotext_utils.py
def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

# modules/processing.py
def create_infotext(positive_prompt, negative_prompt, guidance_scale, guidance_rescale, guidance_cutoff, shift, clipskip, steps, seed, width, height, controlNetSettings, state):
    generation_params = {
        "CLIP-L":       '✓' if state[0] else '✗',
        "CLIP-G":       '✓' if state[1] else '✗',
        "T5":           '✓' if state[2] else '✗', #2713, 2717
        "zero negative":'✓' if state[3] else '✗',
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Steps": steps,
        "CFG": f"{guidance_scale} ({guidance_rescale}) [/ {guidance_cutoff}]",
        "Shift": f"{shift}",
        "Clip skip": f"{clipskip}",
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
        "controlNet": controlNetSettings,
    }
#add loras list and scales

    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
    noise_text = f"\nInitial noise: {SD3Storage.noiseRGBA}" if SD3Storage.noiseRGBA[3] != 0.0 else ""

    return f"Model: StableDiffusion3\n{prompt_text}{generation_params_text}{noise_text}"

def predict(positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, guidance_cutoff, shift, clipskip, 
            num_steps, sampling_seed, num_images, style, i2iSource, i2iDenoise, maskSource, maskCutOff, 
            controlNet, controlNetImage, controlNetStrength, controlNetStart, controlNetEnd, 
            *args):

    try:
        with open('huggingface_access_token.txt', 'r') as file:
            access_token = file.read().strip()
    except:
        print ("SD3: couldn't load 'huggingface_access_token.txt' from the webui directory. Will not be able to download models. Local cache will work.")
        access_token = 0

    torch.set_grad_enabled(False)
    
    # do I care about catching this?
#    if SD3Storage.CL == False and SD3Storage.CG == False and SD3Storage.T5 == False:
    
    volatileState = [SD3Storage.CL, SD3Storage.CG, SD3Storage.T5, SD3Storage.ZN]
    
    if controlNet != 0 and controlNetImage != None and controlNetStrength > 0.0:
        controlNetImage = controlNetImage.resize((width, height))
        useControlNet = ['InstantX/SD3-Controlnet-Canny', 'InstantX/SD3-Controlnet-Pose', 'InstantX/SD3-Controlnet-Tile'][controlNet-1]
    else:
        controlNetStrength = 0.0
        useControlNet = None

    if i2iSource != None:
        if SD3Storage.i2iAllSteps == True:
            num_steps = int(num_steps / i2iDenoise)

        i2iSource = i2iSource.resize((width, height))
    else:
        i2iDenoise = 1.0
        maskSource = None

    if maskSource != None:
        maskSource = maskSource.resize((int(width/8), int(height/8)))
    else:
        maskCutOff = 1.0

    #   triple prompt, automatic support, no longer needs button to enable
    split_positive = positive_prompt.split('|')
    pc = len(split_positive)
    if pc == 1:
        positive_prompt_1 = split_positive[0].strip()
        positive_prompt_2 = positive_prompt_1
        positive_prompt_3 = positive_prompt_1
    elif pc == 2:
        if SD3Storage.T5 == True:
            positive_prompt_1 = split_positive[0].strip()
            positive_prompt_2 = positive_prompt_1
            positive_prompt_3 = split_positive[1].strip()
        else:
            positive_prompt_1 = split_positive[0].strip()
            positive_prompt_2 = split_positive[1].strip()
            positive_prompt_3 = ''
    elif pc >= 3:
        positive_prompt_1 = split_positive[0].strip()
        positive_prompt_2 = split_positive[1].strip()
        positive_prompt_3 = split_positive[2].strip()
        
    split_negative = negative_prompt.split('|')
    nc = len(split_negative)
    if nc == 1:
        negative_prompt_1 = split_negative[0].strip()
        negative_prompt_2 = negative_prompt_1
        negative_prompt_3 = negative_prompt_1
    elif nc == 2:
        if SD3Storage.T5 == True:
            negative_prompt_1 = split_negative[0].strip()
            negative_prompt_2 = negative_prompt_1
            negative_prompt_3 = split_negative[1].strip()
        else:
            negative_prompt_1 = split_negative[0].strip()
            negative_prompt_2 = split_negative[1].strip()
            negative_prompt_3 = ''
    elif nc >= 3:
        negative_prompt_1 = split_negative[0].strip()
        negative_prompt_2 = split_negative[1].strip()
        negative_prompt_3 = split_negative[2].strip()

    if style != 0:
        positive_prompt_1 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_1)
        positive_prompt_2 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_2)
        positive_prompt_3 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_3)
        negative_prompt_1 = styles.styles_list[style][2] + negative_prompt_1
        negative_prompt_2 = styles.styles_list[style][2] + negative_prompt_2
        negative_prompt_3 = styles.styles_list[style][2] + negative_prompt_3

    combined_positive = positive_prompt_1 + " |\n"
    combined_positive += ("[repeat 1]" if positive_prompt_2 == positive_prompt_1 else positive_prompt_2) + " |\n"
    combined_positive += ("[repeat 1]" if positive_prompt_3 == positive_prompt_1 else ("[repeat 2]" if positive_prompt_3 == positive_prompt_2 else positive_prompt_3))
    combined_negative = negative_prompt_1 + " |\n" + negative_prompt_2 + " |\n" + negative_prompt_3

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    SD3Storage.lastSeed = fixed_seed

    source = "stabilityai/stable-diffusion-3-medium-diffusers"

    useCachedEmbeds = (SD3Storage.combined_positive == combined_positive and
                       SD3Storage.combined_negative == combined_negative and
                       SD3Storage.redoEmbeds == False and
                       SD3Storage.clipskip == clipskip)
    if useCachedEmbeds:
        print ("Skipping text encoders and tokenizers.")
    else:
        #do the T5, if enabled
        if SD3Storage.T5 == True:
            tokenizer = T5TokenizerFast.from_pretrained(
                source, local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='tokenizer_3',
                torch_dtype=torch.float16,
                max_length=512,
                token=access_token,
                )

            text_inputs = tokenizer(
                positive_prompt_3,          padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True,    return_tensors="pt", )
            positive_input_ids = text_inputs.input_ids

            if SD3Storage.ZN != True:
                text_inputs = tokenizer(
                    negative_prompt_3,          padding="max_length", max_length=512, truncation=True,
                    add_special_tokens=True,    return_tensors="pt", )
                negative_input_ids = text_inputs.input_ids

            del tokenizer, text_inputs

            text_encoder = T5EncoderModel.from_pretrained(
                source, local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='text_encoder_3',
                torch_dtype=torch.float16,
                device_map='auto',
                token=access_token,
               )
            
            positive_embeds_3 = text_encoder(positive_input_ids)[0]

            if SD3Storage.ZN == True:
                negative_embeds_3 = torch.zeros((1, 512, 4096),    device='cpu', dtype=torch.float16, )
            else:
                negative_embeds_3 = text_encoder(negative_input_ids)[0]

            del text_encoder
        else:
            #512 is tokenizer max length from config; 4096 is transformer joint_attention_dim from its config
            positive_embeds_3 = torch.zeros((1, 512, 4096),    device='cpu', dtype=torch.float16, )
            negative_embeds_3 = torch.zeros((1, 512, 4096),    device='cpu', dtype=torch.float16, )
            #end: T5

    #   do first CLIP
        if SD3Storage.CL == True:
            tokenizer = CLIPTokenizer.from_pretrained(
                source, local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='tokenizer',
                torch_dtype=torch.float16,
                token=access_token,
                )

            text_inputs = tokenizer(
                positive_prompt_1,         padding="max_length",  max_length=77,  truncation=True,
                return_tensors="pt", )
            positive_input_ids = text_inputs.input_ids

            if SD3Storage.ZN != True:
                text_inputs = tokenizer(
                    negative_prompt_1,         padding="max_length",  max_length=77,  truncation=True,
                    return_tensors="pt", )
                negative_input_ids = text_inputs.input_ids

            del tokenizer, text_inputs

            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                source, local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='text_encoder',
                torch_dtype=torch.float16,
                token=access_token,
                )
            text_encoder.to('cuda')

            positive_embeds = text_encoder(positive_input_ids.to('cuda'), output_hidden_states=True)
            pooled_positive_embeds_1 = positive_embeds[0]
            positive_embeds_1 = positive_embeds.hidden_states[-(clipskip + 2)]
            del positive_embeds
            if SD3Storage.ZN == True:
                negative_embeds_1 = torch.zeros((1, 77, 4096),     device='cuda', dtype=torch.float16, )
                pooled_negative_embeds_1 = torch.zeros((1, 768),   device='cuda', dtype=torch.float16, )
            else:
                negative_embeds = text_encoder(negative_input_ids.to('cuda'), output_hidden_states=True)
                pooled_negative_embeds_1 = negative_embeds[0]
                negative_embeds_1 = negative_embeds.hidden_states[-2]
                del negative_embeds
            del text_encoder

        else:
            #77 is tokenizer max length from config; 4096 is transformer joint_attention_dim from its config
            positive_embeds_1 = torch.zeros((1, 77, 4096),     device='cuda', dtype=torch.float16, )
            negative_embeds_1 = torch.zeros((1, 77, 4096),     device='cuda', dtype=torch.float16, )
            pooled_positive_embeds_1 = torch.zeros((1, 768),   device='cuda', dtype=torch.float16, )
            pooled_negative_embeds_1 = torch.zeros((1, 768),   device='cuda', dtype=torch.float16, )

    #   do second CLIP
        if SD3Storage.CG == True:
            tokenizer = CLIPTokenizer.from_pretrained(
                source, local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='tokenizer_2',
                torch_dtype=torch.float16,
                token=access_token,
                )

            text_inputs = tokenizer(
                positive_prompt_2,         padding="max_length",  max_length=77,  truncation=True,
                return_tensors="pt", )
            positive_input_ids = text_inputs.input_ids

            if SD3Storage.ZN != True:
                text_inputs = tokenizer(
                    negative_prompt_2,         padding="max_length",  max_length=77,  truncation=True,
                    return_tensors="pt", )
                negative_input_ids = text_inputs.input_ids

            del tokenizer, text_inputs

            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                source, local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='text_encoder_2',
                torch_dtype=torch.float16,
                token=access_token,
                )
            text_encoder.to('cuda')

            positive_embeds = text_encoder(positive_input_ids.to('cuda'), output_hidden_states=True)
            pooled_positive_embeds_2 = positive_embeds[0]
            positive_embeds_2 = positive_embeds.hidden_states[-(clipskip + 2)]
            del positive_embeds
            if SD3Storage.ZN == True:
                negative_embeds_2 = torch.zeros((1, 77, 4096),     device='cuda', dtype=torch.float16, )
                pooled_negative_embeds_2 = torch.zeros((1, 1280),  device='cuda', dtype=torch.float16, )
            else:
                negative_embeds = text_encoder(negative_input_ids.to('cuda'), output_hidden_states=True)
                pooled_negative_embeds_2 = negative_embeds[0]
                negative_embeds_2 = negative_embeds.hidden_states[-2]
                del negative_embeds
            del text_encoder


        else:
            #77 is tokenizer max length from config; 4096 is transformer joint_attention_dim from its config
            positive_embeds_2 = torch.zeros((1, 77, 4096),     device='cuda', dtype=torch.float16, )
            negative_embeds_2 = torch.zeros((1, 77, 4096),     device='cuda', dtype=torch.float16, )
            pooled_positive_embeds_2 = torch.zeros((1, 1280),  device='cuda', dtype=torch.float16, )
            pooled_negative_embeds_2 = torch.zeros((1, 1280),  device='cuda', dtype=torch.float16, )

        #merge
        clip_positive_embeds = torch.cat([positive_embeds_1, positive_embeds_2], dim=-1)
        clip_positive_embeds = torch.nn.functional.pad(clip_positive_embeds, (0, positive_embeds_3.shape[-1] - clip_positive_embeds.shape[-1]) )
        clip_negative_embeds = torch.cat([negative_embeds_1, negative_embeds_2], dim=-1)
        clip_negative_embeds = torch.nn.functional.pad(clip_negative_embeds, (0, negative_embeds_3.shape[-1] - clip_negative_embeds.shape[-1]) )

        positive_embeds = torch.cat([clip_positive_embeds, positive_embeds_3.to('cuda')], dim=-2)
        negative_embeds = torch.cat([clip_negative_embeds, negative_embeds_3.to('cuda')], dim=-2)

        positive_pooled = torch.cat([pooled_positive_embeds_1, pooled_positive_embeds_2], dim=-1)
        negative_pooled = torch.cat([pooled_negative_embeds_1, pooled_negative_embeds_2], dim=-1)

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
        del pooled_positive_embeds_1, pooled_positive_embeds_2, pooled_negative_embeds_1, pooled_negative_embeds_2
        del positive_embeds_1, positive_embeds_2, positive_embeds_3
        del negative_embeds_1, negative_embeds_2, negative_embeds_3

        gc.collect()
        torch.cuda.empty_cache()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(source,
                                                                subfolder='scheduler', local_files_only=False, cache_dir=".//models//diffusers//",
                                                                shift=shift,
                                                                token=access_token,)

   
    pipe = SD3Pipeline_DoE_combined.from_pretrained(
        source,
        local_files_only=False, cache_dir=".//models//diffusers//",
        torch_dtype=torch.float16,
                            text_encoder  =None,
        tokenizer_2=None,   text_encoder_2=None,
        tokenizer_3=None,   text_encoder_3=None,
        scheduler=scheduler,
        token=access_token,
        controlnet=SD3ControlNetModel.from_pretrained(useControlNet, cache_dir=".//models//diffusers//", torch_dtype=torch.float16) if useControlNet else None
    )
    del scheduler

    pipe.to('cuda')
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()       #tiling works once only?

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    shape = (
        num_images,
        pipe.transformer.config.in_channels,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )

    latents = randn_tensor(shape, generator=generator, dtype=torch.float16).to('cuda').to(torch.float16)
    #   colour the initial noise
    if SD3Storage.noiseRGBA[3] != 0.0:
        nr = SD3Storage.noiseRGBA[0] ** 0.5
        ng = SD3Storage.noiseRGBA[1] ** 0.5
        nb = SD3Storage.noiseRGBA[2] ** 0.5

        imageR = torch.tensor(np.full((8,8), (nr), dtype=np.float32))
        imageG = torch.tensor(np.full((8,8), (ng), dtype=np.float32))
        imageB = torch.tensor(np.full((8,8), (nb), dtype=np.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)

        image = pipe.image_processor.preprocess(image).to('cuda').to(torch.float16)
        image_latents = (pipe.vae.encode(image).latent_dist.sample(generator) - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        image_latents = image_latents.repeat(num_images, 1, 1, 1)

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
            pipe.load_lora_weights(lorafile, local_files_only=True, adapter_name=SD3Storage.lora)
#            pipe.set_adapters(SD3Storage.lora, adapter_weights=SD3Storage.lora_scale)    #.set_adapters doesn't exist so no easy multiple LoRAs and weights
        except:
            print ("Failed: LoRA: " + lorafile)
            return gr.Button.update(value='Generate', variant='primary', interactive=True), None

#adapter_weight_scales = { "unet": { "down": 1, "mid": 0, "up": 0} }
#pipe.set_adapters("pixel", adapter_weight_scales)
#pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])

#    print (pipe.scheduler.compatibles)

    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = pipe(
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            guidance_cutoff=guidance_cutoff,
            prompt_embeds=SD3Storage.positive_embeds.to('cuda'),
            negative_prompt_embeds=SD3Storage.negative_embeds.to('cuda'),
            pooled_prompt_embeds=SD3Storage.positive_pooled.to('cuda'),
            negative_pooled_prompt_embeds=SD3Storage.negative_pooled.to('cuda'),
            num_images_per_prompt=num_images,
            output_type="pil",
            generator=generator,
            latents=latents,

            image=i2iSource,
            mask_image = maskSource,
            strength=i2iDenoise,

            control_image=controlNetImage, 
            controlnet_conditioning_scale=controlNetStrength,  
            control_guidance_start=controlNetStart,
            control_guidance_end=controlNetEnd,
            
            mask_cutoff = maskCutOff,

            joint_attention_kwargs={"scale": SD3Storage.lora_scale }
        ).images
        del controlNetImage, i2iSource

    del pipe, generator, latents

    gc.collect()
    torch.cuda.empty_cache()

    if useControlNet != None:
        useControlNet += f" strength: {controlNetStrength}, step range: {controlNetStart}-{controlNetEnd}"

    result = []
    for image in output:
        info=create_infotext(
            combined_positive, combined_negative, 
            guidance_scale, guidance_rescale, guidance_cutoff, shift, clipskip, num_steps, 
            fixed_seed, 
            width, height, 
            useControlNet, volatileState)

        result.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed,
            combined_positive,
            opts.samples_format,
            info
        )
        fixed_seed += 1

    del output
    gc.collect()
    torch.cuda.empty_cache()

    return gr.Button.update(value='Generate', variant='primary', interactive=True), result


def on_ui_tabs():

    def buildLoRAList ():
        loras = ["(None)"]
        
        import glob
        customLoRA = glob.glob(".\models\diffusers\SD3Lora\*.safetensors")

        for i in customLoRA:
            filename = i.split('\\')[-1]
            loras.append(filename[0:-12])

        return loras

    loras = buildLoRAList ()

    def refreshLoRAs ():
        loras = buildLoRAList ()
        return gr.Dropdown.update(choices=loras)
   
    def getGalleryIndex (evt: gr.SelectData):
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
                                                         torch_dtype=torch.float32, 
                                                         cache_dir=".//models//diffusers//", 
                                                         trust_remote_code=True)
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  cache_dir=".//models//diffusers//", 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image, return_tensors="pt")
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
            if p != prompts[-1]:
                result += ' | \n'
            del parsed_answer

        del model, processor

        if SD3Storage.captionToPrompt:
            return result
        else:
            return originalPrompt


    def toggleCL ():
        SD3Storage.redoEmbeds = True
        SD3Storage.CL ^= True
        return gr.Button.update(variant=['secondary', 'primary'][SD3Storage.CL])
    def toggleCG ():
        SD3Storage.redoEmbeds = True
        SD3Storage.CG ^= True
        return gr.Button.update(variant=['secondary', 'primary'][SD3Storage.CG])
    def toggleT5 ():
        SD3Storage.redoEmbeds = True
        SD3Storage.T5 ^= True
        return gr.Button.update(variant=['secondary', 'primary'][SD3Storage.T5])
    def toggleZN ():
        SD3Storage.redoEmbeds = True
        SD3Storage.ZN ^= True
        return gr.Button.update(variant=['secondary', 'primary'][SD3Storage.ZN])


    def toggleAS ():
        SD3Storage.i2iAllSteps ^= True
        return gr.Button.update(variant=['secondary', 'primary'][SD3Storage.i2iAllSteps])
    def toggleC2P ():
        SD3Storage.captionToPrompt ^= True
        return gr.Button.update(variant=['secondary', 'primary'][SD3Storage.captionToPrompt])

    def updateWH (idx, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        if idx == 0:
            return None, 1536, 672
        if idx == 1:
            return None, 1344, 768
        if idx == 2:
            return None, 1248, 832
        if idx == 3:
            return None, 1120, 896
        if idx == 4:
            return None, 1024, 1024
        if idx == 5:
            return None, 896, 1120
        if idx == 6:
            return None, 832, 1248
        if idx == 7:
            return None, 768, 1344
        if idx == 8:
            return None, 672, 1536
        return None, w, h

    def randomString ():
        import random
        import string
        alphanumeric_string = ''
        for i in range(8):
            alphanumeric_string += ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ' '
        return alphanumeric_string

    def toggleGenerate (R, G, B, A, lora, scale):
        SD3Storage.noiseRGBA = [R, G, B, A]
        SD3Storage.lora = lora
        SD3Storage.lora_scale = scale# if lora != "(None)" else 1.0
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    with gr.Blocks() as sd3_block:
        with ResizeHandleRow():
            with gr.Column():
                with gr.Row():
                    CL = ToolButton(value='CL', variant='primary',   tooltip='use CLIP-L text encoder for positive')
                    CG = ToolButton(value='CG', variant='primary',   tooltip='use CLIP-G text encoder for positive')
                    T5 = ToolButton(value='T5', variant='secondary', tooltip='use T5 text encoder for positive')
                    ZN = ToolButton(value='ZN', variant='secondary', tooltip='zero out negative embeds')
                positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here ...', default='', lines=1.01)
                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=1.01)
                    randNeg = ToolButton(value='rng', variant='secondary', tooltip='random negative')
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value='(None)', type='index', scale=0)

                with gr.Row():
                    width = gr.Slider(label='Width', minimum=512, maximum=2048, step=32, value=1024, elem_id='StableDiffusion3_width')
                    swapper = ToolButton(value='\U000021C4')
                    height = gr.Slider(label='Height', minimum=512, maximum=2048, step=32, value=1024, elem_id='StableDiffusion3_height')
                    dims = gr.Dropdown(['21:9 — 1568 \u00D7 672', '16:9 — 1344 \u00D7 768', '3:2 — 1248 \u00D7 832', '5:4 — 1120 \u00D7 896', 
                                        '1:1 — 1024 \u00D7 1024',
                                        '4:5 — 896 \u00D7 1120', '2:3 — 832 \u00D7 1248', '9:16 — 768 \u00D7 1344', '9:21 — 672 \u00D7 1568'],
                                        label='Quickset', type='index', scale=0)

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=5, scale=2)
                    CFGrescale = gr.Slider(label='rescale CFG', minimum=0.00, maximum=1.0, step=0.01, value=0.0, precision=0.01, scale=1)
                    CFGcutoff = gr.Slider(label='CFG cutoff step', minimum=0.00, maximum=1.0, step=0.01, value=1.0, precision=0.01, scale=1)
                    shift = gr.Slider(label='Shift', minimum=1.0, maximum=8.0, step=0.1, value=3.0, scale=1)
                with gr.Row():
                    steps = gr.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                    clipskip = gr.Slider(label='Clip skip', minimum=0, maximum=8, step=1, value=0, scale=1) #use webUI setting instead?
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=0)
                    random = ToolButton(value='\U0001f3b2\ufe0f')
                    reuseSeed = ToolButton(value='\u267b\ufe0f')
                    batch_size = gr.Number(label='Batch size', minimum=1, maximum=9, value=1, precision=1, scale=0)

                with gr.Row(equal_height=True):
                    lora = gr.Dropdown([x for x in loras], label='LoRA', value="(None)", type='value', multiselect=False, scale=1)
                    refresh = ToolButton(value='\U0001f504')
                    scale = gr.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01)

                with gr.Accordion(label='the colour of noise', open=False):
                    with gr.Row():
                        initialNoiseR = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gr.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')

                with gr.Accordion(label='ControlNet', open=False):
                    with gr.Row():
                        CNSource = gr.Image(label='control image', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            CNMethod = gr.Dropdown(['(None)', 'canny', 'pose', 'tile'], label='method', value='(None)', type='index', multiselect=False, scale=1)
                            CNStrength = gr.Slider(label='Strength', minimum=0.00, maximum=1.0, step=0.01, value=0.8)
                            CNStart = gr.Slider(label='Start step', minimum=0.00, maximum=1.0, step=0.01, value=0.0)
                            CNEnd = gr.Slider(label='End step', minimum=0.00, maximum=1.0, step=0.01, value=0.8)

                with gr.Accordion(label='image to image', open=False):
                    with gr.Row():
                        i2iSource = gr.Image(label='source image', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        maskSource = gr.Image(label='source mask', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            with gr.Row():
                                i2iDenoise = gr.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                AS = ToolButton(value='AS')
 
                            i2iSetWH = gr.Button(value='Set Width / Height from image')
                            i2iFromGallery = gr.Button(value='Get image from gallery')
                            with gr.Row():
                                i2iCaption = gr.Button(value='Caption this image (Florence-2)', scale=9)
                                toPrompt = ToolButton(value='P', variant='secondary')
                            maskCut = gr.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)

                ctrls = [positive_prompt, negative_prompt, width, height, guidance_scale, CFGrescale, CFGcutoff, shift, clipskip, steps, sampling_seed,
                         batch_size, style, i2iSource, i2iDenoise, maskSource, maskCut, CNMethod, CNSource, CNStrength, CNStart, CNEnd]

            with gr.Column():
                generate_button = gr.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gr.Gallery(label='Output', height="80vh",
                                            show_label=False, object_fit='contain', visible=True, columns=3, preview=True)
#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks, alt text does

                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=positive_prompt,
                        source_image_component=output_gallery,
                    ))


        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        refresh.click(refreshLoRAs, inputs=[], outputs=[lora])
        CL.click(toggleCL, inputs=[], outputs=CL)
        CG.click(toggleCG, inputs=[], outputs=CG)
        T5.click(toggleT5, inputs=[], outputs=T5)
        ZN.click(toggleZN, inputs=[], outputs=ZN)
        AS.click(toggleAS, inputs=[], outputs=AS)
        swapper.click(fn=None, _js="function(){switchWidthHeight('StableDiffusion3')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        randNeg.click(randomString, inputs=[], outputs=[negative_prompt])

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])
        i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])#outputs=[positive_prompt]
        toPrompt.click(toggleC2P, inputs=[], outputs=[toPrompt])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, output_gallery])
        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale], outputs=[generate_button])

    return [(sd3_block, "StableDiffusion3", "sd3")]

script_callbacks.on_ui_tabs(on_ui_tabs)

