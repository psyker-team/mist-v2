accelerate launch attacks/ita.py \
 --cuda --low_vram_mode \
 --instance_data_dir_for_adversarial data/training/painting \
 --output_dir output/mist/sd1-5/painting \
 --instance_prompt "a painting in the style of sks, high quality, masterpiece" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --pgd_eps 0.05 \
 --max_f_train_steps 60


 accelerate launch attacks/ita.py \
 --cuda --low_vram_mode \
 --instance_data_dir_for_adversarial data/training/painting \
 --output_dir output/mist/sd1-5/null \
 --instance_prompt "a painting in the style of sks, high quality, masterpiece" \
 --mixed_precision bf16 \
 --max_train_steps 1 \
 --checkpointing_iterations 1 \
 --pgd_eps 0.05 \
 --max_f_train_steps 0

 accelerate launch attacks/ita_diffusers_version.py \
 --model_type sd \
 --cuda \
 --low_vram_mode \
 --pretrained_model_name_or_path stable-diffusion/stable-diffusion-1-5/ \
 --instance_data_dir data/training/painting \
 --output_dir output/mist/sd1-5/painting \
 --instance_prompt "a painting of in the style of sks, high quality, masterpiece" \
 --gradient_checkpointing \
 --pre_compute_text_embeddings \
 --low_vram_mode \
 --mixed_precision bf16 \
 --max_f_train_steps 1 \
 --max_train_steps 1 \
 --num_train_epochs 1 \
 --max_adv_train_steps 1 \
 --pgd_eps 0.03

 accelerate launch attacks/ita_diffusers_version.py \
 --model_type sd \
 --cuda \
 --low_vram_mode \
 --pretrained_model_name_or_path stable-diffusion/stable-diffusion-1-5/ \
 --instance_data_dir data/training/painting \
 --output_dir output/mist/sd1-5/painting \
 --instance_prompt "a painting of in the style of sks, high quality, masterpiece" \
 --gradient_checkpointing \
 --low_vram_mode \
 --mixed_precision bf16 \
 --max_f_train_steps 5 \
 --max_train_steps 50 \
 --num_train_epochs 1 \
 --max_adv_train_steps 50 \
 --pgd_eps 0.04 \
 --original_resolution


  accelerate launch attacks/ita_diffusers_version.py \
 --model_type sd \
 --cuda \
 --low_vram_mode \
 --pretrained_model_name_or_path stable-diffusion/stable-diffusion-1-5/ \
 --instance_data_dir data/training/kent \
 --output_dir output/mist/sd1-5/kent \
 --instance_prompt "a photo of a animate person in the style of sks, high quality, masterpiece" \
 --gradient_checkpointing \
 --low_vram_mode \
 --mixed_precision bf16 \
 --max_f_train_steps 5 \
 --max_train_steps 50 \
 --num_train_epochs 1 \
 --max_adv_train_steps 50 \
 --pgd_eps 0.04 \
 --original_resolution



