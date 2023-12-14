import gradio as gr
from attacks.mist import update_args_with_config, main

'''
    TODO: 
    - SDXL
    - model changing
''' 


def process_image(eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
        class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
            rank, prior_loss_weight, fused_weight, constraint_mode, lpips_bound, lpips_weight):

    config = (eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
        class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
            rank, prior_loss_weight, fused_weight, constraint_mode, lpips_bound, lpips_weight)
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
                    device = gr.Radio(["cpu", "gpu"], value="gpu", label="Device",
                                    info="If you do not have good GPUs on your PC, choose 'CPU'.")
                    # precision = gr.Radio(["float16", "bfloat16"], value="bfloat16", label="Precision",
                    #                 info="Precision used in computing")
                    resize = gr.Checkbox(value=True, label="Resizing the output image to the original resolution")
                    mode = gr.Radio(["Mode 1", "Mode 2", "Mode 3"], value="Mode 1", label="Mode",
                                    info="Two modes both work with different visualization.")
                    # model_type = gr.Radio(["Stable Diffusion", "SDXL"], value="Stable Diffusion", label="Target Model",
                    #                 info="Model used by imaginary copyright infringers")
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to your images")
                    output_path = gr.Textbox(label="Output Path", lines=1, placeholder="Path to store the outputs")
                    model_path = gr.Textbox(label="Target Model Path", lines=1, placeholder="Path to the target model")
                    class_path = gr.Textbox(label="Path to place contrast images ", lines=1, placeholder="Path to the target model")
                    prompt = gr.Textbox(label="Prompt", lines=1, placeholder="Describe your images")

                    with gr.Accordion("Professional Setups", open=False):
                        class_prompt = gr.Textbox(label="Class prompt", lines=1, placeholder="Prompt for contrast images.")
                        max_train_steps = gr.Slider(1, 20, step=1, value=5, label='Epochs',
                                      info="Training epochs of Mist-V2")
                        max_f_train_steps = gr.Slider(0, 30, step=1, value=10, label='LoRA Steps',
                                      info="Training steps of LoRA in one epoch")
                        max_adv_train_steps = gr.Slider(0, 100, step=5, value=30, label='Attacking Steps',
                                      info="Training steps of attacking in one epoch")
                        lora_lr = gr.Number(minimum=0.0, maximum=1.0, label="The learning rate of LoRA", value=0.0001)
                        pgd_lr = gr.Number(minimum=0.0, maximum=1.0, label="The learning rate of PGD", value=0.005)
                        rank = gr.Slider(4, 32, step=4, value=4, label='LoRA Ranks',
                                      info="Ranks of LoRA (Bigger ranks need better GPUs)")
                        prior_loss_weight = gr.Number(minimum=0.0, maximum=1.0, label="The weight of prior loss", value=0.1)
                        fused_weight = gr.Number(minimum=0.0, maximum=1.0, label="The weight of vae loss", value=0.00001)
                        constraint_mode = gr.Radio(["Epsilon", "LPIPS"], value="Epsilon", label="Constraint Mode",
                                    info="The mode to constraint the watermark")
                        lpips_bound = gr.Number(minimum=0.0, maximum=0.2, label="The LPIPS bound", value=0.1)
                        lpips_weight = gr.Number(minimum=0.0, maximum=1.0, label="The weight of LPIPI constraint", value=0.5)

                    # inputs = [eps, device, precision, mode, model_type, original_resolution, data_path, \
                    #           output_path, model_path, prompt, max_f_train_steps, max_train_steps, max_adv_train_steps, lora_lr, pgd_lr, rank]
                    
                    inputs = [eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
        class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
            rank, prior_loss_weight, fused_weight, constraint_mode, lpips_bound, lpips_weight]
                    

                    image_button = gr.Button("Mist")

                    
            image_button.click(process_image, inputs=inputs)

    demo.queue().launch(share=False)