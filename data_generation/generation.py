from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from diffusers import AutoPipelineForText2Image
import torch
import os
import json
import random
from pathlib import Path
from itertools import product

## Load pipeline
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda:0")

## Generation
folder_path = "/linxindisk/AttrSyn/datasets/synthetic/cub-photo/sdxl-diversity-huge"
subfolders = sorted([f.name for f in Path(folder_path).iterdir() if f.is_dir()])

## Consider behavior, background, style
# fine_grained_class_prompts = json.load(open('class_desc_by_wiki.json', 'r'))
behavior_prompts = json.load(open('class_behavior.json'))
background_prompts = json.load(open('class_background.json'))
# style_prompts = ["oil painting", "watercolor painting", "cartoon painting", "gouache painting", "simple drawing"]
style_prompts = ["portrait photo", "minimalistic photo", "close-up detail photo", "candid photo", "night photo"]

for subfolder in subfolders[:20]:
    class_name = " ".join(subfolder.split(".")[1].split("_")).lower()
    class_prompt = "a {} bird,\n".format(class_name)

    subfolder_path = os.path.join(folder_path, subfolder)

    diversities = product(behavior_prompts[class_name], background_prompts[class_name], style_prompts)

    for i, diversity in enumerate(diversities):
        for j in range(10):
            behavior_prompt, background_prompt, style_prompt = diversity
            prompt = class_prompt + behavior_prompt + ",\n" + background_prompt + ",\n" + style_prompt
            image  = pipe(prompt=prompt).images[0].resize((512,512))
            image.save(os.path.join(subfolder_path, "./{:03d}-{:02d}.png".format(i+1, j+1)))