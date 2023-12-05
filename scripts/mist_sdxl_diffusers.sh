accelerate launch attacks/ita_diffusers_version.py \
 --model_type sdxl \
 --cuda \
 --low_vram_mode \
 --pretrained_model_name_or_path stable-diffusion/sdxl-base-1.0/ \
 --instance_data_dir data/training/painting \
 --output_dir output/mist/sdxl/painting \
 --instance_prompt "a painting of in the style of sks, high quality, masterpiece" \
 --gradient_checkpointing \
 --pre_compute_text_embeddings \
 --low_vram_mode \
 --mixed_precision bf16 \
 --max_f_train_steps 5 \
 --max_train_steps 50 \
 --num_train_epochs 1 \
 --max_adv_train_steps 50 \
 --pgd_eps 0.05