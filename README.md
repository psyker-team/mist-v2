<p align="center">
<br>
<!-- <img  src="mist_logo.png"> -->
<img  src="assets/MIST_V2_LOGO.png">
<br>
</p>


[![project page](https://img.shields.io/badge/homepage-psyker--team.io-blue.svg)](https://psyker-team.github.io/index_en.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k5tLNsWTTAkOlkl5d9llf93bJ6csvMuZ?usp=sharing)
[![One-click Package](https://img.shields.io/badge/-Google_Drive-1A73E8.svg?style=flat&logo=Google-Drive&logoColor=white)](https://drive.google.com/drive/folders/1vg8oK2BUOla5adaJcFYx5QMq0-MoP8kk?usp=drive_link)

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2310.04687-red.svg)](https://arxiv.org/abs/2310.04687) -->
<!-- 
[![document](https://img.shields.io/badge/document-passing-light_green.svg)](https://arxiv.org/abs/2310.04687)
-->
<!-- 
### [project page](https://mist-project.github.io) | [arxiv](https://arxiv.org/abs/2310.04687) | [document](https://arxiv.org/abs/2310.04687) -->

<!-- #region -->
<!-- <p align="center">
<img  src="effect_show.png">
</p> -->
<!-- #endregion -->
<!-- 
> Mist adds watermarks to images, making them unrecognizable and unusable for AI-for-Art models that try to mimic them. -->

<!-- #region -->
<p align="center">
<img  src="assets/user_2.jpg">
</p>
<!-- <p align="center">
<img  src="user_case_2.png">
</p> -->
<!-- #endregion -->

> Mist's Effects in User Cases. **The first row:** Lora generation from source images.
**The second row:** Lora generation from Mist-treated samples. Mist V2 significantly disrupts the output of the generation, effectively protecting artists' images. Used images are from anonymous artists. All rights reserved. 
<!-- #region -->
<!-- <p align="center">
<img  src="robustness.png">
</p> -->
<!-- #endregion -->

<!-- > Robustness of Mist against image preprocessing. -->

<!-- ## News

**2022/12/11**: Mist V2 released.  -->


## What is Mist

Mist is a powerful image preprocessing tool designed for the purpose of protecting the style and content of
images from being mimicked by state-of-the-art AI-for-Art applications. By adding watermarks to the images, Mist renders them unrecognizable and inmitable for the
models employed by AI-for-Art applications. Attempts by AI-for-Art applications to mimic these Misted images
will be ineffective, and the output image of such mimicry will be scrambled and unusable as artwork.


<p align="center">
<img  src="assets/effect_show.png">
</p>

**Updates of version 2.1.0**: 
- Enhanced protection against AI-for-Art applications like Lora and SDEdit
- Imperceptible noise.
- 3-5 minutes processing with only 6GB of GPU memory in most cases. CPU processing supported.
- Resilience against denoising methods.


 <!-- For more details, refer to our [documentation](https://arxiv.org/abs/2310.04687). -->



## Quick Start

### For end users (artists, photographers, designers,...)

We provide two approaches for you to deploy Mist-v2:

- **Free version for local deployment**: If your system is Windows and it has an Nvidia GPU with more than 6GB VRAM, you can download our free version pack (i.e. no need for installation, runnable after downloading) from [Google Drive](https://drive.google.com/drive/folders/1vg8oK2BUOla5adaJcFYx5QMq0-MoP8kk?usp=drive_link). This is the safest and most flexible way to deploy and run Mist-v2.
We provide a [guideline](docs/Handbook-Free-version.md) about how to deploy and run.

- **Colab Notebook**: If your system is MacOS or you do not own proper Nvidia GPUs, you can run Mist with our [Colab Notebook](https://colab.research.google.com/drive/1k5tLNsWTTAkOlkl5d9llf93bJ6csvMuZ?usp=sharing) on free GPU resources provided by Google (Thank you Google). The Notebook is **self-instructed**. 

### For developers

If you want to build Mist-v2 from source code, we also provide the instruction as follows:

<details><summary> (click-to-expand)  
 </summary>

#### Environment

**Preliminaries:** To run this repository, please have [Anaconda](https://pytorch.org/) installed in your work station. The GPU version of Mist requires a NVIDIA GPU in [Ampere](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) or more advanced architecture with more than 6GB VRAM. You can also try the CPU version 
in a moderate running speed.

Clone this repository to your local and get into the repository root:

```bash
git clone https://github.com/mist-project/mist-v2.git
cd mist-v2
```

Then, run the following commands in the root of the repository to install the environment:

```bash
conda create -n mist-v2 python=3.10
conda activate mist-v2
pip install -r requirements.txt
```

#### Usage

Run Mist V2 in the default setup on GPU:
```bash
accelerate launch attacks/mist.py --cuda --low_vram_mode --instance_data_dir $INSTANCE_DIR --output_dir $OUTPUT_DIR --class_data_dir $CLASS_DATA_DIR --instance_prompt $PROMPT --class_prompt $CLASS_PROMPT --mixed_precision bf16
```

Run Mist V2 in the default setup on CPU:
```bash
accelerate launch attacks/mist.py --instance_data_dir $INSTANCE_DIR --output_dir $OUTPUT_DIR --class_data_dir $CLASS_DATA_DIR --instance_prompt $PROMPT --class_prompt $CLASS_PROMPT --mixed_precision bf16
```

 Detailed demonstration of the parameters:

| Parameter       | Explanation                                                                                |
| --------------- | ------------------------------------------------------------------------------------------ |
| $INSTANCE_DIR   | Directory of  input clean images. The goal is to add adversarial noise to them.            |
| $OUTPUT_DIR     | Directory for output adversarial examples (misted images).                                 |
| $CLASS_DATA_DIR | Directory  for class data in prior preserved training of Dreambooth, required to be empty. |
| $PROMPT         | Prompt that describes the input clean images, used to perturb the images.                  |
| $CLASS_PROMPT   | Prompt used to generate class data, recommended to be similar to $PROMPT.                  |

Here is a case command to run Mist V2 on GPU:

```bash
accelerate launch attacks/mist.py --cuda --low_vram_mode --instance_data_dir data/training --output_dir output/ --class_data_dir data/class --instance_prompt "a photo of a misted person, high quality, masterpiece" --class_prompt "a photo of a person, high quality, masterpiece" --mixed_precision bf16
```

The aforementioned GUI can be also booted by running:

```bash
python mist-webui.py
```

#### Evaluation

This repo provides a simple pipeline to evaluate the output adversarial examples. 

Basically, this pipeline trains a LoRA on the adversarial examples and samples images with the LoRA. 
Note that our adversarial examples may induce LoRA to output images with NSFW contents 
(for example, chaotic texture). As stated, this is to prevent LoRA training on unauthorized image data. To evaluate the effectiveness of our method, we disable the safety checker in the LoRA sampling script. Following is the instruction to run the pipeline.

First, train a LoRA on the output adversarial examples. 

```bash
accelerate launch eval/train_dreambooth_lora_15.py --instance_data_dir=$LORA_INPUT_DIR --output_dir=$LORA_OUTPUT_DIR --class_data_dir=$LORA_CLASS_DIR --instance_prompt $LORA_PROMPT --class_prompt $LORA_CLASS_PROMPT --resolution=512 --train_batch_size=1 --learning_rate=1e-4 --scale_lr --max_train_steps=2000
```

Detailed demonstration of the parameters:  


| Parameter          | Explanation                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| $LORA_INPUT_DIR    | Directory of  training data (adversarial examples), staying the same as $OUTPUT_DIR in the previous table. |
| $LORA_OUTPUT_DIR   | Directory to store the trained LoRA.                                                                       |
| $LORA_CLASS_DIR    | Directory  for class data in prior preserved training of Dreambooth, required to be empty.                 |
| $LORA_PROMPT       | Prompt that describes the training data, used to train the LoRA.                                           |
| $LORA_CLASS_PROMPT | Prompt used to generate class data, recommended to be related to $LORA_PROMPT.                             |


Next, open the `eval/sample_lora_15.ipynb` and run the first block. After that, change the value of the variable `LORA_OUTPUT_DIR` to be the previous `$LORA_OUTPUT_DIR` when training the LoRA. 

```Python
from lora_diffusion import tune_lora_scale, patch_pipe
torch.manual_seed(time.time())

# The directory of LoRA
LORA_OUTPUT_DIR = [The value of $LORA_OUTPUT_DIR]
...
```

Finally, run the second block to see the output and evaluate the performance of Mist.
</details>

## TODO

- [x] Mist-v2 release (12/15/2023)
- [x] Mist-v2 free version (1/13/2024)
- [x] Mist-v2 Colab Notebook (1/18/2024)
- [ ] Mist-v2 WebUI: Misting images online, open to everyone with human artists verification)
  - [ ] Appling for free GPU resources
  - [ ] Implementing the web service


## Contribute & Contact

Mist is an open-source project and we sincerely welcome contributions. Apart from Mist, we are broadly interested in the ethics, copyright, and trustworthy concerns of new-generation AIGC. If you have good ideas on these topics, feel free to contact us through our e-mail: [mist202304@gmail.com](mist202304@gmail.com). 


## A Glimpse to Methodology

Our paper is still in progress so that we provide a glimpse to our methodology here. Mist V2 works by adversarially attacking generative diffusion models. Basically, the attacking is an optimization over the following objective:

$$ \underset{x'}{min} \mathbb{E} {(z_0', \epsilon,t)}  \Vert \epsilon_\theta(z'_t(z'_0,\epsilon),t)-z_0^T\Vert^2_2, \Vert x'-x\Vert\leq\zeta$$

<details><summary> (click-to-expand) We demonstrate the notation in the following table. </summary>


| Variable          | Explanation                                                      |
| ----------------- | ---------------------------------------------------------------- |
| $x$ / $x'$        | The clean image / The adversarial example                        |
| $t$               | Time step in the diffusion model.                                |
| $z'_0$            | The latent variable of $x'$ in the 0th time step                 |
| $\epsilon$        | A standard Gaussian noise                                        |
| $z_0^T$           | The latent variable of a target image $x^T$ in the 0th time step |
| $\epsilon_\theta$ | The noise predictor (U-Net) in the diffusion model               |
| $\zeta$           | The budget of adversarial noise                                  |

</details>


Intuitively, Mist-v2 guides the gradient predicted by the diffusion model to a fixed and directed error. When finetuned on the Misted
images, the model tries to fix this error by adding a fixed counteracting bias to its prediction. This bias will be adopted as parts of
the pattern learned by finetuning. The finetuned model will also add the fixed bias in their sampling process, resulting in chaotic texture in the output images. 


## License

This project is licensed under the Apache-2.0 license. Additionally, we forbid any unauthorized commercial use. Mist series will be permanently free and open-sourced. Currently, we do not cooperate with any person or organization for commercial purpose.

 
## Citation

Our paper is still in progress. If you find this repo useful, please keep an eye to our updates. We will release the paper as it is available for public release.

Additionally, Mist-v2 benefits from the following papers. Their ideas and results inspire us to dive into the mechanism why adversarial attacks work on latent diffusion models. We kindly suggest you to cite them if possible. 

```
@inproceedings{liang2023adversarial,
  title={Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples},
  author={Liang, Chumeng and Wu, Xiaoyu and Hua, Yang and Zhang, Jiaru and Xue, Yiming and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={International Conference on Machine Learning},
  pages={20763--20786},
  year={2023},
  organization={PMLR}
}
```

```
@article{xue2023toward,
  title={Toward effective protection against diffusion based mimicry through score distillation},
  author={Xue, Haotian and Liang, Chumeng and Wu, Xiaoyu and Chen, Yongxin},
  journal={arXiv preprint arXiv:2311.12832},
  year={2023}
}
```

```
@inproceedings{van2023anti,
  title={Anti-DreamBooth: Protecting users from personalized text-to-image synthesis},
  author={Van Le, Thanh and Phung, Hao and Nguyen, Thuan Hoang and Dao, Quan and Tran, Ngoc N and Tran, Anh},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2116--2127},
  year={2023}
}
```

```
@article{liang2023mist,
  title={Mist: Towards Improved Adversarial Examples for Diffusion Models},
  author={Liang, Chumeng and Wu, Xiaoyu},
  journal={arXiv preprint arXiv:2305.12683},
  year={2023}
}
```






