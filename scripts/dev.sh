accelerate launch attacks/mist.py \
 --cuda --low_vram_mode \
 --instance_data_dir data/training/portrait/ \
 --output_dir output/mist/sd1-5/portrait \
 --class_data_dir data/class \
 --instance_prompt "a photo of a sks person, high quality, masterpiece" \
 --class_prompt "a photo of a person, high quality, masterpiece" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --pgd_eps 0.04


 accelerate launch attacks/mist.py \
 --cuda --low_vram_mode \
 --instance_data_dir data/training/painting/ \
 --output_dir output/mist/sd1-5/painting \
 --class_data_dir data/class \
 --instance_prompt "a painting of a sks person, high quality, masterpiece" \
 --class_prompt "a painting of a person, high quality, masterpiece" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --resize \
 --pgd_eps 0.04


 


