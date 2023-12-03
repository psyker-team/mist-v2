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