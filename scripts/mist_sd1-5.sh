export MODEL_NAME="stable-diffusion/stable-diffusion-1-5"
export INSTANCE_DIR="data/training/kent/"
export OUTPUT_DIR="output/mist/sd1-5/kent"
export CLASS_DIR="data/class"
 
accelerate launch attacks/mist.py \
 --cuda --low_vram_mode --resize \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir=$OUTPUT_DIR \
 --class_data_dir=$CLASS_DIR\
 --instance_prompt "an animated painting of a sks person, high quality, masterpiece" \
 --class_prompt "an animated painting of a person, high quality, masterpiece" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --pgd_eps 0.04