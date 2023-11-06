# mist-v2

## Installation

This repository is based on [PyTorch](https://pytorch.org/). Specifically, please make sure your environment satisfies `python==3.10` and `cuda >= 11.6`. We do not guarantee the code's performance on other versions.

If you're a conda user, simply run:
```bash
conda create -n ldm python=3.10
conda activate ldm
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
You may change the cuda and pytorch version according to your own environment. However, we strongly recommend to use pytorch with version no less than `1.13.1`.

Then, install the other requirements:
```bash
pip install -r requirements.txt
```

## Usage

To run a default attack, simply run:

```bash
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial $DATA_DIR --output_dir $OUTPUT_DIR --class_data_dir $CLASS_DATA_DIR --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
```

If class data is not generated yet, the program will automatically generate it under the directory `$CLASS_DATA_DIR`. The generated class data will be used in the following attacks. You may change the prompt according to your own data.

The default command in the WebUI is:

```bash
accelerate launch --cpu attacks/iadvdm_all_device.py --instance_data_dir_for_adversarial data/training --output_dir output/ --class_data_dir data/class --instance_prompt "a painting in the style of sks, high quality, masterpiece" --class_prompt "a painting, high quality, masterpiece" --mixed_precision bf16 --max_train_steps 3 --checkpointing_iterations 1
```
