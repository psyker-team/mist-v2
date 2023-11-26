accelerate launch attacks/ita.py \
 --cuda --low_vram_mode \
 --instance_data_dir_for_adversarial data/training \
 --output_dir output/sd1-5/ \
 --class_data_dir data/class \
 --instance_prompt "a photo of a sks person, high quality, masterpiece" \
 --class_prompt "a photo of a person, high quality, masterpiece" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1