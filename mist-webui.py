import gradio as gr
from attacks.ita_diffusers_version import update_args_with_config, main

'''
    TODO: 
    - SDXL
    - model changing
''' 


def process_image(eps, device, precision, mode, model_type, original_resolution, data_path, output_path, model_path, \
              prompt, max_f_train_steps, max_train_steps, max_adv_train_steps, lora_lr, pgd_lr, rank):

    config = (eps, device, precision, mode, model_type, original_resolution, data_path, output_path, model_path, \
              prompt, max_f_train_steps, max_train_steps, max_adv_train_steps, lora_lr, pgd_lr, rank)
    args = None
    args = update_args_with_config(args, config)
    main(args)

if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Image("MIST_logo.png", show_label=False)
            with gr.Row():
                with gr.Column():
                    eps = gr.Slider(0, 32, step=1, value=10, label='Strength',
                                    info="Larger strength results in stronger but more visible defense.")
                    device = gr.Radio(["cpu", "gpu"], value="cpu", label="Device",
                                    info="If you do not have good GPUs on your PC, choose 'CPU'.")
                    precision = gr.Radio(["float16", "bfloat16"], value="bfloat16", label="Precision",
                                    info="Precision used in computing")
                    mode = gr.Radio(["Mode 1", "Mode 2"], value="Mode 1", label="Mode",
                                    info="Two modes both work with different visualization.")
                    model_type = gr.Radio(["Stable Diffusion", "SDXL"], value="Stable Diffusion", label="Target Model",
                                    info="Model used by imaginary copyright infringers")
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to your images")
                    output_path = gr.Textbox(label="Output Path", lines=1, placeholder="Path to store the outputs")
                    model_path = gr.Textbox(label="Target Model Path", lines=1, placeholder="Path to the target model")
                    prompt = gr.Textbox(label="Prompt", lines=1, placeholder="Describe your images")

                    with gr.Accordion("Professional Setups", open=False):
                        original_resolution = gr.Checkbox(label="Class Data Training")
                        max_f_train_steps = gr.Slider(1, 20, step=1, value=5, label='Epochs',
                                      info="Training epochs of Mist-V2")
                        max_train_steps = gr.Slider(0, 200, step=10, value=50, label='LoRA Steps',
                                      info="Training steps of LoRA in one epoch")
                        max_adv_train_steps = gr.Slider(0, 200, step=10, value=50, label='Attacking Steps',
                                      info="Training steps of attacking in one epoch")
                        lora_lr = gr.Number(label="The learning rate of LoRA", value=0.0005)
                        pgd_lr = gr.Number(label="The learning rate of PGD", value=0.005)
                        rank = gr.Slider(4, 20, step=4, value=4, label='LoRA Ranks',
                                      info="Ranks of LoRA")

                    inputs = [eps, device, precision, mode, model_type, original_resolution, data_path, \
                              output_path, model_path, prompt, max_f_train_steps, max_train_steps, max_adv_train_steps, lora_lr, pgd_lr, rank]
                    
                    image_button = gr.Button("Mist")

                    
            image_button.click(process_image, inputs=inputs)

    demo.queue().launch(share=False)