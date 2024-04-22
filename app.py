import torch
import gradio as gr
from PIL import Image
import random
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline
)
import tempfile
import time

# Define your CSS style for the application
css = """
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: #f4f4f9;
    color: #333;
}

h1, h2, h3 {
    color: #5d5d5d;
}

button {
    background-color: #0066A2;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
}

input, textarea, select {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border: 1px solid #ddd;
    border-radius: 5px;
    width: calc(100% - 22px);
}

.column {
    float: left;
    width: 50%;
    padding: 20px;
}

.row:after {
    content: "";
    display: table;
    clear: both;
}

@media (max-width: 600px) {
    .column {
        width: 100%;
    }
}
"""

# Initialize your models (assuming models are already downloaded and available locally)
vae = AutoencoderKL.from_pretrained("/path/to/sd-vae-ft-mse", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("/path/to/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)
main_pipe = StableDiffusionControlNetPipeline(
    vae=vae,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")

image_pipe = StableDiffusionImg2ImgPipeline(**main_pipe.components).to("cuda")

def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size
    new_dimension = min(width, height)
    left = (width - new_dimension) / 2
    top = (height - new_dimension) / 2
    right = (width + new_dimension) / 2
    bottom = (height + new_dimension) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)
    return img

def inference(control_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, control_guidance_start, control_guidance_end, upscaler_strength, seed, sampler):
    control_image = center_crop_resize(control_image)
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)

    output = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=50,
        output_type="numpy"
    )
    return output.images[0], seed

# Setting up the Gradio interface
with gr.Blocks(css=css) as app:
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Input Illusion")
            prompt = gr.Textbox(label="Prompt")
            neg_prompt = gr.Textbox(label="Negative Prompt")
            guidance_scale = gr.Slider(minimum=0, maximum=20, label="Guidance Scale", value=7)
            controlnet_scale = gr.Slider(minimum=0, maximum=10, label="ControlNet Scale", value=1)
            start_control = gr.Slider(minimum=0, maximum=1, step=0.1, label="Start Control")
            end_control = gr.Slider(minimum=0, maximum=1, step=0.1, label="End Control")
            upscaler_str = gr.Slider(minimum=0, maximum=1, step=0.1, label="Upscaler Strength")
            seed = gr.Number(label="Seed", value=-1)
            sampler = gr.Dropdown(choices=["Euler", "DPM++ Karras SDE"], label="Sampler")
            submit_btn = gr.Button("Run")
        with gr.Column():
            output_img = gr.Image(label="Output Image")
            output_seed = gr.Label()

    submit_btn.click(
        inference,
        inputs=[img_input, prompt, neg_prompt, guidance_scale, controlnet_scale, start_control, end_control, upscaler_str, seed, sampler],
        outputs=[output_img, output_seed]
    )

if __name__ == "__main__":
    app.launch(host="0.0.0.0", port=7860, enable_queue=True)
