import numpy as np
import gradio as gr
from attacks.ita_webui import init, attack, update_args_with_config
import os
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps

def process_image(eps, max_training_step, device, mode, data_path, class_path, output_path, model_path,\
                  max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, lora_rank):
    config = (eps, max_training_step, device, mode, data_path, class_path, output_path, model_path, \
              max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, lora_rank)

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
                    device = gr.Radio(["cpu", "gpu"], value="cpu", label="Device",
                                    info="If you do not have good GPUs with your PC, choose 'CPU'.")
                    mode = gr.Radio(["Mode 1", "Mode 2"], value="Mode 1", label="Mode",
                                    info="Two modes both work with different visualization.")
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to your images")
                    class_path = gr.Textbox(label="Class Path", lines=1, placeholder="Path to the comparison images")
                    output_path = gr.Textbox(label="Output Path", lines=1, placeholder="Path to store the outputs")
                    model_path = gr.Textbox(label="Target Model Path", lines=1, placeholder="Path to the target model")

                    with gr.Accordion("Professional Setups", open=False):
                        max_f_train_steps = gr.Slider(1, 20, step=1, value=1, label='Steps',
                                      info="Training steps of LoRA")
                        max_adv_train_steps = gr.Slider(0, 200, step=10, value=50, label='Steps',
                                      info="Training steps of LoRA")
                        lora_lr = gr.Number(label="The learning rate of LoRA", default=1e-4, float=True)
                        pgd_lr = gr.Number(label="The learning rate of PGD", default=5e-3, float=True)
                        lora_rank = gr.Slider(4, 20, step=4, value=4, label='LoRA Ranks',
                                      info="Ranks of LoRA")
                        


                    inputs = [eps, max_training_step, device, mode, data_path, class_path, output_path, model_path, \
                              max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, lora_rank]
                    image_button = gr.Button("Mist")

                    
            image_button.click(process_image, inputs=inputs)

    demo.queue().launch(share=False)