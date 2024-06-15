import math
import torch
import gc
import json
import numpy as np

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste
import gradio as gr

from PIL import Image

#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(True)


import customStylesListSD3 as styles

class SD3Storage:
    lastSeed = -1
    combined_positive = None
    combined_negative = None
    positive_embeds = None
    negative_embeds = None
    positive_pooled = None
    negative_pooled = None
    clipskip = 0
    T5 = False
    i2iAllSteps = False
    redoEmbeds = True

from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
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
def create_infotext(positive_prompt, negative_prompt, guidance_scale, clipskip, steps, seed, width, height):
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Steps": steps,
        "CFG": f"{guidance_scale}",
        "Clip skip": f"{clipskip}",
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None
    }
#add loras list and scales

    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    return f"Model: StableDiffusion3\n{prompt_text}{generation_params_text}"

def predict(positive_prompt, negative_prompt, width, height, guidance_scale, clipskip, 
            num_steps, sampling_seed, num_images, style, i2iSource, i2iDenoise, *args):

    torch.set_grad_enabled(False)

    if i2iSource != None:
        if i2iDenoise < (num_steps + 1) / 1000:
            i2iDenoise = (num_steps + 1) / 1000
        if SD3Storage.i2iAllSteps == True:
            num_steps = int(num_steps / i2iDenoise)

    #   triple prompt, automatic support, no longer needs button to enable
    split_positive = positive_prompt.split('|')
    pc = len(split_positive)
    if pc == 1:
        positive_prompt_1 = positive_prompt
        positive_prompt_2 = positive_prompt
        positive_prompt_3 = positive_prompt
    elif pc == 2:
        if SD3Storage.T5 == True:
            positive_prompt_1 = split_positive[0].strip()
            positive_prompt_2 = split_positive[0].strip()
            positive_prompt_3 = split_positive[1].strip()
        else:
            positive_prompt_1 = split_positive[0].strip()
            positive_prompt_2 = split_positive[1].strip()
    elif pc >= 3:
        positive_prompt_1 = split_positive[0].strip()
        positive_prompt_2 = split_positive[1].strip()
        positive_prompt_3 = split_positive[2].strip()
        
    split_negative = negative_prompt.split('|')
    nc = len(split_negative)
    if nc == 1:
        negative_prompt_1 = negative_prompt
        negative_prompt_2 = negative_prompt
        negative_prompt_3 = negative_prompt
    elif nc == 2:
        if SD3Storage.T5 == True:
            negative_prompt_1 = split_negative[0].strip()
            negative_prompt_2 = split_negative[0].strip()
            negative_prompt_3 = split_negative[1].strip()
        else:
            negative_prompt_1 = split_negative[0].strip()
            negative_prompt_2 = split_negative[1].strip()
    elif nc >= 3:
        negative_prompt_1 = split_negative[0].strip()
        negative_prompt_2 = split_negative[1].strip()
        negative_prompt_3 = split_negative[2].strip()

    if style != 0:  #better to rebuild stored prompt from _1,_2,_3 so random changes at end /whitespace effect nothing
        positive_prompt_1 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_1)
        positive_prompt_2 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_2)
        positive_prompt_3 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_3)
        negative_prompt_1 = styles.styles_list[style][2] + negative_prompt_1
        negative_prompt_2 = styles.styles_list[style][2] + negative_prompt_2
        negative_prompt_3 = styles.styles_list[style][2] + negative_prompt_3

    combined_positive = positive_prompt_1 + " |\n" + positive_prompt_2 + " |\n" + positive_prompt_3
    combined_negative = negative_prompt_1 + " |\n" + negative_prompt_2 + " |\n" + negative_prompt_3

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    SD3Storage.lastSeed = fixed_seed

    source = "stabilityai/stable-diffusion-3-medium-diffusers"
    with open('huggingface_access_token.txt', 'r') as file:
        access_token = file.read().rstrip()

# 3 positives, 3 negatives (1 of each necessary)

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
                torch_dtype=torch.float16, )

            text_inputs = tokenizer(
                positive_prompt_3,          padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True,    return_tensors="pt", )
            positive_input_ids = text_inputs.input_ids

            text_inputs = tokenizer(
                negative_prompt_3,          padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True,    return_tensors="pt", )
            negative_input_ids = text_inputs.input_ids

            del tokenizer

            text_encoder = T5EncoderModel.from_pretrained(
                source, local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='text_encoder_3',
                torch_dtype=torch.float16,
                device_map='auto'
                )
            
            positive_embeds = text_encoder(positive_input_ids)[0]
            _, seq_len, _ = positive_embeds.shape
            positive_embeds = positive_embeds.repeat(1, num_images, 1)
            positive_embeds_3 = positive_embeds.view(num_images, seq_len, -1)

            negative_embeds = text_encoder(negative_input_ids)[0]
            _, seq_len, _ = negative_embeds.shape
            negative_embeds = negative_embeds.repeat(1, num_images, 1)
            negative_embeds_3 = negative_embeds.view(num_images, seq_len, -1)

            del text_encoder
        else:
            #512 is tokenizer max length from config; 4096 is transformer joint_attention_dim from its config
            positive_embeds_3 = torch.zeros((num_images, 512, 4096),
                                            device='cpu', dtype=torch.float16, )
            negative_embeds_3 = torch.zeros((num_images, 512, 4096),
                                            device='cpu', dtype=torch.float16, )
            #end: T5

    #   do first CLIP
        tokenizer = CLIPTokenizer.from_pretrained(
            source, local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='tokenizer',
            torch_dtype=torch.float16, )

        text_inputs = tokenizer(
            positive_prompt_1,         padding="max_length",  max_length=77,  truncation=True,
            return_tensors="pt", )
        positive_input_ids = text_inputs.input_ids

        text_inputs = tokenizer(
            negative_prompt_1,         padding="max_length",  max_length=77,  truncation=True,
            return_tensors="pt", )
        negative_input_ids = text_inputs.input_ids

        del tokenizer

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            source, local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='text_encoder',
            torch_dtype=torch.float16,
            )
        text_encoder.to('cuda')

        positive_embeds = text_encoder(positive_input_ids.to('cuda'), output_hidden_states=True)
        negative_embeds = text_encoder(negative_input_ids.to('cuda'), output_hidden_states=True)
        del text_encoder

        pooled_positive_embeds = positive_embeds[0]
        pooled_negative_embeds = negative_embeds[0]

        if clipskip == 0:
            positive_embeds = positive_embeds.hidden_states[-2]
        else:
            positive_embeds = positive_embeds.hidden_states[-(clipskip + 2)]

        negative_embeds = negative_embeds.hidden_states[-2]

        _, seq_len, _ = positive_embeds.shape
        positive_embeds = positive_embeds.repeat(1, num_images, 1)
        positive_embeds_1 = positive_embeds.view(num_images, seq_len, -1)
        pooled_positive_embeds = pooled_positive_embeds.repeat(1, num_images, 1)
        pooled_positive_embeds_1 = pooled_positive_embeds.view(num_images, -1)

        _, seq_len, _ = negative_embeds.shape
        negative_embeds = negative_embeds.repeat(1, num_images, 1)
        negative_embeds_1 = negative_embeds.view(num_images, seq_len, -1)
        pooled_negative_embeds = pooled_negative_embeds.repeat(1, num_images, 1)
        pooled_negative_embeds_1 = pooled_negative_embeds.view(num_images, -1)

    #   do second CLIP
        tokenizer = CLIPTokenizer.from_pretrained(
            source, local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='tokenizer_2',
            torch_dtype=torch.float16, )

        text_inputs = tokenizer(
            positive_prompt_2,         padding="max_length",  max_length=77,  truncation=True,
            return_tensors="pt", )
        positive_input_ids = text_inputs.input_ids

        text_inputs = tokenizer(
            negative_prompt_2,         padding="max_length",  max_length=77,  truncation=True,
            return_tensors="pt", )
        negative_input_ids = text_inputs.input_ids

        del tokenizer

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            source, local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='text_encoder_2',
            torch_dtype=torch.float16,
            )
        text_encoder.to('cuda')

        positive_embeds = text_encoder(positive_input_ids.to('cuda'), output_hidden_states=True)
        negative_embeds = text_encoder(negative_input_ids.to('cuda'), output_hidden_states=True)
        del text_encoder

        pooled_positive_embeds = positive_embeds[0]
        pooled_negative_embeds = negative_embeds[0]

        if clipskip is None:
            positive_embeds = positive_embeds.hidden_states[-2]
        else:
            positive_embeds = positive_embeds.hidden_states[-(clipskip + 2)]

        negative_embeds = negative_embeds.hidden_states[-2]

        _, seq_len, _ = positive_embeds.shape
        positive_embeds = positive_embeds.repeat(1, num_images, 1)
        positive_embeds_2 = positive_embeds.view(num_images, seq_len, -1)
        pooled_positive_embeds = pooled_positive_embeds.repeat(1, num_images, 1)
        pooled_positive_embeds_2 = pooled_positive_embeds.view(num_images, -1)

        _, seq_len, _ = negative_embeds.shape
        negative_embeds = negative_embeds.repeat(1, num_images, 1)
        negative_embeds_2 = negative_embeds.view(num_images, seq_len, -1)
        pooled_negative_embeds = pooled_negative_embeds.repeat(1, num_images, 1)
        pooled_negative_embeds_2 = pooled_negative_embeds.view(num_images, -1)

        #merge
        clip_positive_embeds = torch.cat([positive_embeds_1, positive_embeds_2], dim=-1)
        clip_negative_embeds = torch.cat([negative_embeds_1, negative_embeds_2], dim=-1)

        clip_positive_embeds = torch.nn.functional.pad(clip_positive_embeds, (0, positive_embeds_3.shape[-1] - clip_positive_embeds.shape[-1]) )
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
        del pooled_positive_embeds_1, pooled_positive_embeds_2, pooled_negative_embeds_1, pooled_negative_embeds_2
        del positive_embeds_1, positive_embeds_2, positive_embeds_3
        del negative_embeds_1, negative_embeds_2, negative_embeds_3

        gc.collect()
        torch.cuda.empty_cache()

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]


    if i2iSource == None:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            source,
            local_files_only=False, cache_dir=".//models//diffusers//",
            torch_dtype=torch.float16,
            tokenizer  =None,   text_encoder  =None,
            tokenizer_2=None,   text_encoder_2=None,
            tokenizer_3=None,   text_encoder_3=None,
            token=access_token
            )
    else:
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            source,
            local_files_only=False, cache_dir=".//models//diffusers//",
            torch_dtype=torch.float16,
                                text_encoder  =None,
            tokenizer_2=None,   text_encoder_2=None,
            tokenizer_3=None,   text_encoder_3=None,
            token=access_token
            )

    pipe.to('cuda')
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()       #tiling works once only?

#   load in LoRA, weight passed to pipe

    if SD3Storage.lora != "(None)":
        lorafile = ".//models/diffusers//SD3Lora//" + SD3Storage.lora + ".safetensors"
        try:
            pipe.load_lora_weights(lorafile, local_files_only=True, adapter_name=SD3Storage.lora)
#            pipe.set_adapters(SD3Storage.lora, adapter_weights=0.5)    #.set_adapters doesn't exist so no easy multiple LoRAs and weights
        except:
            print ("Failed: LoRA: " + lorafile)
            return gr.Button.update(value='Generate', variant='primary', interactive=True), None

#adapter_weight_scales = { "unet": { "down": 1, "mid": 0, "up": 0} }
#pipe.set_adapters("pixel", adapter_weight_scales)
#pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
        pipe.transformer.set_adapters(SD3Storage.lora, adapter_weights=0.5)    #.set_adapters doesn't exist so no easy multiple LoRAs and weights

#   i2i may require default FlowMatchEulerDiscreteScheduler

    with torch.inference_mode():
        if i2iSource == None:
            output = pipe(
                prompt=None,
                negative_prompt=None, 
                num_inference_steps=num_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,

                prompt_embeds=SD3Storage.positive_embeds.to('cuda'),
                negative_prompt_embeds=SD3Storage.negative_embeds.to('cuda'),
                pooled_prompt_embeds=SD3Storage.positive_pooled.to('cuda'),
                negative_pooled_prompt_embeds=SD3Storage.negative_pooled.to('cuda'),
         
                output_type="pil",
                generator=generator,
                joint_attention_kwargs={"scale": SD3Storage.lora_scale }
            ).images
        else:
            i2iSource = i2iSource.resize((width, height))
            output = pipe(
                prompt=None,
                negative_prompt=None, 
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,

                prompt_embeds=SD3Storage.positive_embeds.to('cuda'),
                negative_prompt_embeds=SD3Storage.negative_embeds.to('cuda'),
                pooled_prompt_embeds=SD3Storage.positive_pooled.to('cuda'),
                negative_pooled_prompt_embeds=SD3Storage.negative_pooled.to('cuda'),
         
                output_type="pil",
                generator=generator,

                image=i2iSource,
                strength=i2iDenoise,
                joint_attention_kwargs={"scale": SD3Storage.lora_scale }
            ).images

    del pipe, generator

    gc.collect()
    torch.cuda.empty_cache()

    result = []
    for image in output:
        info=create_infotext(
            combined_positive, combined_negative,
            guidance_scale, clipskip, num_steps, 
            fixed_seed, 
            width, height, )

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

    def toggleT5 ():
        SD3Storage.redoEmbeds = True
        if SD3Storage.T5 == False:
            SD3Storage.T5 = True
            return gr.Button.update(variant='primary')
        else:
            SD3Storage.T5 = False
            return gr.Button.update(variant='secondary')

    def toggleAS ():
        if SD3Storage.i2iAllSteps == False:
            SD3Storage.i2iAllSteps = True
            return gr.Button.update(variant='primary')
        else:
            SD3Storage.i2iAllSteps = False
            return gr.Button.update(variant='secondary')

    def toggleGenerate (lora, scale):
        SD3Storage.lora = lora
        SD3Storage.lora_scale = scale# if lora != "(None)" else 1.0
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    with gr.Blocks() as sd3_block:
        with ResizeHandleRow():
            with gr.Column():
                with gr.Row():
                    positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here ...', default='', lines=1.1)
##                    scheduler = gr.Dropdown(["default",
##                                             "DDPM",
##                                             "DEIS",
##                                             "DPM++ 2M",
##                                             "DPM++ 2M SDE",
##                                             "DPM",
##                                             "DPM SDE",
##                                             "Euler",
##                                             "SA-solver",
##                                             "UniPC",
##                                             ],
##                        label='Sampler', value="default", type='value', scale=0)
                    T5 = ToolButton(value='T5', variant='secondary', tooltip='use T5 text encoder')

                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=1.1)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value='(None)', type='index', scale=0)

                with gr.Row(equal_height=True):
                    lora = gr.Dropdown([x for x in loras], label='LoRA', value="(None)", type='value', multiselect=False, scale=1)
                    refresh = ToolButton(value='\U0001f504')
                    scale = gr.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01)

                with gr.Row():
                    width = gr.Slider(label='Width', minimum=512, maximum=2048, step=32, value=1024, elem_id='StableDiffusion3_width')
                    swapper = ToolButton(value='\U000021C5')
                    height = gr.Slider(label='Height', minimum=512, maximum=2048, step=32, value=1024, elem_id='StableDiffusion3_height')

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=5, scale=2)
                    clipskip = gr.Slider(label='Clip skip', minimum=0, maximum=8, step=1, value=0, scale=2)
                with gr.Row():
                    steps = gr.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value='\U0001f3b2\ufe0f')
                    reuseSeed = ToolButton(value='\u267b\ufe0f')
                    batch_size = gr.Number(label='Batch size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gr.Accordion(label='image to image', open=False):
                    with gr.Row():
                        i2iSource = gr.Image(label='image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            with gr.Row():
                                i2iDenoise = gr.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                AS = ToolButton(value="AS")
 
                            i2iSetWH = gr.Button(value='Set Width / Height from image')
                            i2iFromGallery = gr.Button(value='Get image from gallery')

                ctrls = [positive_prompt, negative_prompt, width, height, guidance_scale, clipskip, steps, sampling_seed,
                         batch_size, style, i2iSource, i2iDenoise]

            with gr.Column():
                generate_button = gr.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gr.Gallery(label='Output', height=shared.opts.gallery_height or None,
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


        refresh.click(refreshLoRAs, inputs=[], outputs=[lora])
        T5.click(toggleT5, inputs=[], outputs=T5)
        AS.click(toggleAS, inputs=[], outputs=AS)
        swapper.click(fn=None, _js="function(){switchWidthHeight('StableDiffusion3')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(toggleGenerate, inputs=[lora, scale], outputs=[generate_button])
        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, output_gallery])

    return [(sd3_block, "StableDiffusion3", "sd3")]

script_callbacks.on_ui_tabs(on_ui_tabs)

