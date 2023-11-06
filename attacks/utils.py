import torch
import torch.nn.functional as F

prompt_dataset = [
    "Portrait of an astronaut in space, detailed starry background, reflective helmet,",
    "Painting of a floating island with giant clock gears, populated with mythical creatures,",
    "Landscape of a Japanese garden in autumn, with a bridge over a koi pond,",
    "Painting representing the sound of jazz music, using vibrant colors and erratic shapes,",
    "Painting of a modern smartphone with classic art pieces appearing on the screen,",
    "Battle scene with futuristic robots and a golden palace in the background,",
    "Scene of a bustling city market with different perspectives of people and stalls,",
    "Scene of a ship sailing in a stormy sea, with dramatic lighting and powerful waves,",
    "Portraint of a female botanist surrounded by exotic plants in a greenhouse,",
    "Painting of an ancient castle at night, with a full moon, gargoyles, and shadows,",
]

style_dataset = [
    "Art Nouveau",
    "Romantic",
    "Cubist",
    "Baroque",
    "Pop Art",
    "Abstract",
    "Impressionist",
    "Surrealist",
    "Renaissance",
    "Pointillism",
]



class attack_mixin:
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unet: torch.nn.Module,
        target_tensor: torch.Tensor,
        noise_scheduler
    ):
        raise NotImplementedError
    
class AdvDM(attack_mixin):
    """
    This attack aims to maximize the training loss of diffusion model
    """
    def __call__(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unet: torch.nn.Module,
        text_encoder: torch.nn.Module,
        input_ids,
        target_tensor: torch.Tensor,
        noise_scheduler
    ):
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
            xtm1_pred = torch.cat(
                [
                    noise_scheduler.step(
                        model_pred[idx : idx + 1],
                        timesteps[idx : idx + 1],
                        noisy_latents[idx : idx + 1],
                    ).prev_sample
                    for idx in range(len(model_pred))
                ]
            )
            xtm1_target = noise_scheduler.add_noise(target_tensor, noise, timesteps - 1)
            loss = loss - F.mse_loss(xtm1_pred, xtm1_target)

        return loss
    
class LatentAttack(attack_mixin):
    """
    This attack aims to minimize the l2 distance between latent and target_tensor
    """
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor=None,
        encoder_hidden_states: torch.Tensor=None,
        unet: torch.nn.Module=None,
        target_tensor: torch.Tensor=None,
        noise_scheduler=None
    ):
        if target_tensor == None:
            raise ValueError("Need a target tensor for pre-attack")
        loss = - F.mse_loss(latents, target_tensor, reduction="mean")
        return loss