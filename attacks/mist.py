import argparse
import copy
import hashlib
import itertools
import logging
import os
import sys
import gc
from pathlib import Path
from colorama import Fore, Style, init,Back
import random, time
'''some system level settings'''
init(autoreset=True)
sys.path.insert(0, sys.path[0]+"/../")
import lpips

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torch import autograd
from typing import Optional, Tuple
import pynvml
# from utils import print_tensor

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from attacks.utils import LatentAttack

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use gpu for attack",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        "-p",
        type=str,
        default="./stable-diffusion/stable-diffusion-1-5",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="",
        required=False,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default="",
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a picture",
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a picture",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=True,
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=0.1,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=50,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the perturbed data is stored",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_false",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train surogate model.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=30,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--checkpointing_iterations",
        type=int,
        default=5,
        help=("Save a checkpoint of the training state every X iterations."),
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--low_vram_mode",
        action="store_false",
        help="Whether or not to use low vram mode.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=5e-3,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=float(8.0/255.0),
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--lpips_bound",
        type=float,
        default=0.1,
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--lpips_weight",
        type=float,
        default=0.5,
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--fused_weight",
        type=float,
        default=1e-5,
        help="The decay of alpha and eps when applying pre_attack",
    )
    parser.add_argument(
        "--target_image_path",
        default="data/MIST.png",
        help="target image for attacking",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['lunet','fused', 'anti-db'],
        default='lunet',
        help="The mode of attack",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        choices=['eps','lpips'],
        default='eps',
        help="The constraint of attack",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help=("File path for text encoder lora to resume training."),
    )
    parser.add_argument(
        "--resize",
        action='store_true',
        required=False,
        help="Should images be resized to --resolution after attacking?",
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    if args.output_dir != "":
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir,exist_ok=True)
            print(Back.BLUE+Fore.GREEN+'create output dir: {}'.format(args.output_dir))
    return args


class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
        self,
        instance_images_tensor,
        prompts,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        
        self.instance_images_tensor = instance_images_tensor
        self.instance_prompts = prompts
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            # self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        instance_prompt = self.instance_prompts[index % self.num_instance_images]
        if instance_prompt == None:
            instance_prompt = self.instance_prompt
        instance_prompt = \
            'masterpiece,best quality,extremely detailed CG unity 8k wallpaper,illustration,cinematic lighting,beautiful detailed glow' + instance_prompt
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # load images & prompts
    images, prompts = [], []
    num_image = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            file_path = os.path.join(data_dir, filename)
            images.append(Image.open(file_path).convert("RGB"))
            num_image += 1

            prompt_name = filename[:-3] + 'txt'
            prompt_path = os.path.join(data_dir, prompt_name)
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as file:
                    text_string = file.read()
                    prompts.append(text_string)
                    print("==load image {} from {}, prompt: {}==".format(num_image-1, file_path, text_string))
            else:
                prompts.append(None)
                print("==load image {} from {}, prompt: None, args.instance_prompt used==".format(num_image-1, file_path))

    # load sizes
    sizes = [img.size for img in images]

    # preprocess images
    images = [image_transforms(img) for img in images]
    images = torch.stack(images)
    print("==tensor shape: {}==".format(images.shape))

    return images, prompts, sizes


def train_one_epoch(
    args,
    accelerator,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    prompts, 
    weight_dtype=torch.bfloat16,
):
    # prepare training data
    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        prompts,
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )

    device = accelerator.device

    # prepare models & inject lora layers
    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    vae.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    if args.low_vram_mode:
        set_use_memory_efficient_attention_xformers(unet,True)

    # this is only done at the first epoch
    unet_lora_params, _ = inject_trainable_lora(
        unet, r=args.lora_rank, loras=args.resume_unet
    )
    if args.train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=["CLIPAttention"],
            r=args.lora_rank,
        )
        # for _up, _down in extract_lora_ups_down(
        #     text_encoder, target_replace_module=["CLIPAttention"]
        # ):
        #     print("Before training: text encoder First Layer lora up", _up.weight.data)
        #     print(
        #         "Before training: text encoder First Layer lora down", _down.weight.data
        #     )
        #     break
    
    # build the optimizer
    optimizer_class = torch.optim.AdamW

    text_lr = (
        args.learning_rate
        if args.learning_rate_text is None
        else args.learning_rate_text
    )

    params_to_optimize = (
        [
            {
                "params": itertools.chain(*unet_lora_params), 
                "lr": args.learning_rate},
            {
                "params": itertools.chain(*text_encoder_lora_params),
                "lr": text_lr,
            },
        ]
        if args.train_text_encoder
        else itertools.chain(*unet_lora_params)
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # begin training
    for step in range(args.max_f_train_steps):
        unet.train()
        text_encoder.train()

        random.seed(time.time())
        instance_idx = random.randint(0, len(train_dataset)-1)
        step_data = train_dataset[instance_idx]
        pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]])
        #print("pixel_values shape: {}".format(pixel_values.shape))
        input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)
        for k in range(pixel_values.shape[0]):
            #calculate loss of instance and class seperately
            pixel_value = pixel_values[k, :].unsqueeze(0).to(device, dtype=weight_dtype)
            latents = vae.encode(pixel_value).latent_dist.sample().detach().clone()
            latents = latents * vae.config.scaling_factor
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # encode text
            input_id = input_ids[k, :].unsqueeze(0)
            encode_hidden_states = text_encoder(input_id)[0]
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            model_pred= unet(noisy_latents, timesteps, encode_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            if k == 1:
                # calculate loss of class(prior)
                loss *= args.prior_loss_weight
            loss.backward()
            if k == 1:
                print(f"==loss - image index {instance_idx}, loss: {loss.detach().item() / args.prior_loss_weight}, prior")
            else:
                print(f"==loss - image index {instance_idx}, loss: {loss.detach().item()}, instance")
                
        params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters())
                    if args.train_text_encoder
                    else unet.parameters()
                )
        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()
    
    return [unet, text_encoder]



def pgd_attack(
    args,
    accelerator,
    models,
    tokenizer,
    noise_scheduler:DDIMScheduler,
    vae:AutoencoderKL,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    target_tensor: torch.Tensor,
    weight_dtype = torch.bfloat16,
):
    """Return new perturbed data"""

    num_steps = args.max_adv_train_steps

    unet, text_encoder = models
    device = accelerator.device
    if args.constraint == 'lpips':
        lpips_vgg = lpips.LPIPS(net='vgg')

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    if args.low_vram_mode:
        unet.set_use_memory_efficient_attention_xformers(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    data_tensor = data_tensor.detach().clone()
    num_image = len(data_tensor)
    image_list = []
    tbar = tqdm(range(num_image))
    tbar.set_description("PGD attack")
    for id in range(num_image):
        tbar.update(1)
        perturbed_image = data_tensor[id, :].unsqueeze(0)
        perturbed_image.requires_grad = True
        original_image = original_images[id, :].unsqueeze(0)
        input_ids = tokenizer(
            args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        input_ids = input_ids.to(device)
        for step in range(num_steps):
            perturbed_image.requires_grad = False
            with torch.no_grad():
                latents = vae.encode(perturbed_image.to(device, dtype=weight_dtype)).latent_dist.mean
            #offload vae
            latents = latents.detach().clone()
            latents.requires_grad = True
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            unet.zero_grad()
            text_encoder.zero_grad()
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # target-shift loss
            if target_tensor is not None:
                if args.mode != 'anti-db':
                    loss = - F.mse_loss(model_pred, target_tensor)
                    # fused mode
                    if args.mode == 'fused':
                        latent_attack = LatentAttack()
                        loss = loss - 1e2 * latent_attack(latents, target_tensor=target_tensor)            

            loss = loss / args.gradient_accumulation_steps
            grads = autograd.grad(loss, latents)[0].detach().clone()
            # now loss is backproped to latents
            #print('grads: {}'.format(grads))
            #do forward on vae again
            perturbed_image.requires_grad = True
            gc_latents = vae.encode(perturbed_image.to(device, dtype=weight_dtype)).latent_dist.mean
            gc_latents.backward(gradient=grads)
            
            if step % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                
                if args.constraint == 'eps':
                    alpha = args.pgd_alpha
                    adv_images = perturbed_image + alpha * perturbed_image.grad.sign()

                    # hard constraint
                    eps = args.pgd_eps
                    eta = torch.clamp(adv_images - original_image, min=-eps, max=+eps)
                    perturbed_image = torch.clamp(original_image + eta, min=-1, max=+1).detach_()
                    perturbed_image.requires_grad = True
                elif args.constraint == 'lpips':
                    # compute reg loss
                    lpips_distance = lpips_vgg(perturbed_image, original_image)
                    reg_loss = args.lpips_weight * torch.max(lpips_distance - args.lpips_bound, 0)[0].squeeze()
                    reg_loss.backward()

                    alpha = args.pgd_alpha
                    adv_images = perturbed_image + alpha * perturbed_image.grad.sign()

                    eta = adv_images - original_image
                    perturbed_image = torch.clamp(original_image + eta, min=-1, max=+1).detach_()
                    perturbed_image.requires_grad = True
                else:
                    raise NotImplementedError
                    
            #print(f"PGD loss - step {step}, loss: {loss.detach().item()}")

        image_list.append(perturbed_image.detach().clone().squeeze(0))
    outputs = torch.stack(image_list)


    return outputs
    
def main(args):
    if args.cuda:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_free = mem_info.free  / float(1073741824)
            if mem_free < 5.5:
                raise NotImplementedError("Your GPU memory is not enough for running Mist on GPU. Please try CPU mode.")
        except:
            raise NotImplementedError("No GPU found in GPU mode. Please try CPU mode.")


    logging_dir = Path(args.output_dir, args.logging_dir)

    if not args.cuda:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir,
            cpu=True
        )
    else:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if args.cuda:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    print("==precision: {}==".format(weight_dtype))

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # add by lora
    unet.requires_grad_(False)
    # end: added by lora

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if not args.cuda:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        ).cuda()
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.encoder.training = True
    vae.encoder.gradient_checkpointing = True

    #print info about train_text_encoder
    
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    perturbed_data, prompts, data_sizes = load_data(
        args.instance_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)


    target_latent_tensor = None
    if args.target_image_path is not None and args.target_image_path != "":
        # print(Style.BRIGHT+Back.BLUE+Fore.GREEN+'load target image from {}'.format(args.target_image_path))
        target_image_path = Path(args.target_image_path)
        assert target_image_path.is_file(), f"Target image path {target_image_path} does not exist"

        target_image = Image.open(target_image_path).convert("RGB").resize((args.resolution, args.resolution))
        target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)
        if args.cuda:
            target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=weight_dtype) / 127.5 - 1.0
        else:
            target_image_tensor = torch.from_numpy(target_image).to(dtype=weight_dtype) / 127.5 - 1.0
        target_latent_tensor = (
            vae.encode(target_image_tensor).latent_dist.sample().to(dtype=weight_dtype) * vae.config.scaling_factor
        )
        target_image_tensor = target_image_tensor.to('cpu')
        del target_image_tensor
        #target_latent_tensor = target_latent_tensor.repeat(len(perturbed_data), 1, 1, 1).cuda()
    f = [unet, text_encoder]
    for i in range(args.max_train_steps):        
        f_sur = copy.deepcopy(f)
        perturbed_data = pgd_attack(
            args,
            accelerator,
            f_sur,
            tokenizer,
            noise_scheduler,
            vae,
            perturbed_data,
            original_data,
            target_latent_tensor,
            weight_dtype,
        )
        del f_sur
        if args.cuda:
            gc.collect()
        f = train_one_epoch(
            args,
            accelerator,
            f,
            tokenizer,
            noise_scheduler,
            vae,
            perturbed_data,
            prompts,
            weight_dtype,
        )
        
        for model in f:
            if model != None:
                model.to('cpu')
        
        if args.cuda:
            gc.collect()
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("=======Epoch {} ends! Memory cost: {}======".format(i, mem_info.used / float(1073741824)))
        else:
            print("=======Epoch {} ends!======".format(i))

        if (i + 1) % args.max_train_steps == 0:
            save_folder = f"{args.output_dir}"
            os.makedirs(save_folder, exist_ok=True)
            noised_imgs = perturbed_data.detach().cpu()
            origin_imgs = original_data.detach().cpu()
            img_names = []
            for filename in os.listdir(args.instance_data_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    img_names.append(str(filename))
            for img_pixel, ori_img_pixel, img_name, img_size in zip(noised_imgs, origin_imgs, img_names, data_sizes):
                save_path = os.path.join(save_folder, f"{i+1}_noise_{img_name}")
                if not args.resize:
                    Image.fromarray(
                        (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    ).save(save_path)
                else:
                    ori_img_path = os.path.join(args.instance_data_dir, img_name)
                    ori_img = np.array(Image.open(ori_img_path).convert("RGB"))

                    ori_img_duzzy = np.array(Image.fromarray(
                        (ori_img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    ).resize(img_size), dtype=np.int32)
                    perturbed_img_duzzy = np.array(Image.fromarray(
                        (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    ).resize(img_size), dtype=np.int32)
                    
                    perturbation = perturbed_img_duzzy - ori_img_duzzy
                    assert perturbation.shape == ori_img.shape

                    perturbed_img =  (ori_img + perturbation).clip(0, 255).astype(np.uint8)
                    # print("perturbation: {}, ori: {}, res: {}".format(
                    #     perturbed_img_duzzy[:2, :2, :], ori_img_duzzy[:2, :2, :], perturbed_img_duzzy[:2, :2, :]))
                    Image.fromarray(perturbed_img).save(save_path)


                print(f"==Saved misted image to {save_path}, size: {img_size}==")
            # print(f"Saved noise at step {i+1} to {save_folder}")
            del noised_imgs

def update_args_with_config(args, config):
    '''
        Update the default augments in args with config assigned by users
        args list:
            eps: 
            max train epoch:
            data path:
            class path:
            output path:
            device: 
                gpu normal,
                gpu low vram,
                cpu,
            mode:
                lunet, full
    '''

    args = parse_args()
    eps, device, mode, resize, data_path, output_path, model_path, class_path, prompt, \
        class_prompt, max_train_steps, max_f_train_steps, max_adv_train_steps, lora_lr, pgd_lr, \
            rank, prior_loss_weight, fused_weight, constraint_mode, lpips_bound, lpips_weight = config
    args.pgd_eps = float(eps)/255.0
    if device == 'cpu':
        args.cuda, args.low_vram_mode = False, False
    else:
        args.cuda, args.low_vram_mode = True, True
    # if precision == 'bfloat16':
    #     args.mixed_precision = 'bf16'
    # else:
    #     args.mixed_precision = 'fp16'
    if mode == 'Mode 1':
        args.mode = 'lunet'
    elif mode == 'Mode 2':
        args.mode = 'fused'
    elif mode == 'Mode 3':
        args.mode = 'anti-db'
    if resize:
        args.resize = True

    assert os.path.exists(data_path) and os.path.exists(output_path)
    args.instance_data_dir = data_path
    args.output_dir = output_path
    args.pretrained_model_name_or_path = model_path
    args.class_data_dir = class_path
    args.instance_prompt = prompt

    args.class_prompt = class_prompt
    args.max_train_steps = max_train_steps
    args.max_f_train_steps = max_f_train_steps
    args.max_adv_train_steps = max_adv_train_steps
    args.learning_rate = lora_lr
    args.pgd_alpha = pgd_lr
    args.rank = rank
    args.prior_loss_weight = prior_loss_weight
    args.fused_weight = fused_weight

    if constraint_mode == 'LPIPS':
        args.constraint = 'lpips'
    else:
        args.constraint = 'eps'
    args.lpips_bound = lpips_bound
    args.lpips_weight = lpips_weight

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

