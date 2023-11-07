import numpy as np
import gradio as gr
from attacks.iadvdm_webui import init, attack, update_args_with_config
import os
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps

def process_image(eps, max_training_step, device, mode, data_path, class_path, output_path):
    config = (eps, max_training_step, device, mode, data_path, class_path, output_path)

    print("Loading ...")
    funcs, args = init(config=config)

    print('Ready to attack...')
    attack(funcs=funcs, args=args)

if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Image("MIST_logo.png", show_label=False)
            with gr.Row():
                with gr.Column():
                    eps = gr.Slider(0, 16, step=2, value=8, label='Strength',
                                    info="Larger strength results in stronger but more visible defense.")
                    max_training_step = gr.Slider(1, 20, step=1, value=5, label='Steps',
                                      info="Larger training steps results in stronger defense.")
                    device = gr.Radio(["CPU", "GPU"], value="CPU", label="Device",
                                    info="If you do not have good GPUs with your PC, choose 'CPU'.")
                    mode = gr.Radio(["Mode 1", "Mode 2"], value="Mode 1", label="Mode",
                                    info="Two modes both work with different visualization.")
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to your images")
                    class_path = gr.Textbox(label="Class Path", lines=1, placeholder="Path to the comparison images")
                    output_path = gr.Textbox(label="Output Path", lines=1, placeholder="Path to store the outputs")
                    inputs = [eps, max_training_step, device, mode, data_path, class_path, output_path]
                    image_button = gr.Button("Mist")
            image_button.click(process_image, inputs=inputs)

    demo.queue().launch(share=False)