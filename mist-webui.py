import gradio as gr
from attacks.mist import update_args_with_config, main
import subprocess

'''
    TODO: 
    - SDXL
    - model changing
''' 


# def process_image(eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
#         class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
#             rank, prior_loss_weight, fused_weight, constraint_mode, lpips_bound, lpips_weight, num_process):

#     config = (eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
#         class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
#             rank, prior_loss_weight, fused_weight, constraint_mode, lpips_bound, lpips_weight, num_process)
#     args = None
#     args = update_args_with_config(args, config)
#     main(args)

def run_mist(eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
            class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
            rank, prior_loss_weight, fused_weight, mixed_precision):

    # Initialize command list
    command = ["accelerate", "launch"]
    if device == 'cpu':
        command.append("--cpu")      

    # Append each parameter to the command list
    command.append("attacks/mist_dev.py")
    command.append("--pgd_eps")
    command.append(str(eps))
    if device == 'gpu':
        command.append("--cuda")
        command.append("--low_vram_mode")
    if resize == True:
        command.append("--resize")
    command.append("--mode")
    if mode == 'Mode 1':
        mode = 'lunet'
    elif mode == 'Mode 2':
        mode = 'fused'
    elif mode == 'Mode 3':
        mode = 'anti-db'
    command.append(mode)

    command.append("--instance_data_dir")
    command.append(data_path)
    command.append("--output_dir")
    command.append(output_path)
    command.append("--pretrained_model_name_or_path")
    command.append(model_path)
    command.append("--class_data_dir")
    command.append(class_path)
    command.append("--instance_prompt")
    command.append(prompt)
    command.append("--class_prompt")
    command.append(class_prompt)
    command.append("--max_train_steps")
    command.append(str(max_train_steps))
    command.append("--max_f_train_steps")
    command.append(str(max_f_train_steps))
    command.append("--max_adv_train_steps")
    command.append(str(max_adv_train_steps))
    command.append("--learning_rate")
    command.append(str(lora_lr))
    command.append("--pgd_alpha")
    command.append(str(pgd_lr))
    command.append("--lora_rank")
    command.append(str(rank))
    command.append("--prior_loss_weight")
    command.append(str(prior_loss_weight))
    command.append("--fused_weight")
    command.append(str(fused_weight))
    command.append("--mixed_precision")
    command.append(mixed_precision)
    

    # Execute the command
    subprocess.run(command)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Image("MIST_logo.png", show_label=False)
            with gr.Row():
                with gr.Column():
                    eps = gr.Slider(0, 32, step=1, value=12, label='Strength',
                                    info="Larger strength results in stronger but more visible defense.")
                    device = gr.Radio(["cpu", "gpu"], value="gpu", label="Device",
                                    info="If you do not have good GPUs on your PC, choose 'CPU'.")
                    resize = gr.Checkbox(value=False, label="Resizing the output image to the original resolution")
                    mode = gr.Radio(["Mode 1", "Mode 2", "Mode 3"], value="Mode 1", label="Mode",
                                    info="Two modes both work with different visualization.")
                    mixed_precision = gr.Radio(["fp16", "bf16"], value="bf16", label="Precision",
                                    info="bf16: 30 series GPU, fp16: else")
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
                        
                    
                    inputs = [eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
                        class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
                        rank, prior_loss_weight, fused_weight, mixed_precision]
                    

                    image_button = gr.Button("Mist")

                    
            # image_button.click(process_image, inputs=inputs)
            image_button.click(run_mist, inputs=inputs)

    demo.queue().launch(share=False)