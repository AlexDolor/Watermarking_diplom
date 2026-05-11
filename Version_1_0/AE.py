# from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import torch
import torch.nn as nn

class DiffusersVAEWrapper(nn.Module):
    def __init__(self, model_id="stabilityai/sd-vae-ft-mse", device="cuda"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id).to(device)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x):
        # x in [-1,1]
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        z = z * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, z):
        z = z / self.vae.config.scaling_factor
        x_rec = self.vae.decode(z).sample
        return x_rec
    


# =========================
# 4. Интерфейс image VAE
# =========================

class DummyVAE(nn.Module):
    """
    Заглушка. Замени на реальный предобученный VAE.
    Интерфейс:
      encode(x) -> z
      decode(z) -> x_rec
    x: [B,3,H,W] в [-1,1]
    z: [B,C,H',W']
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 4, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)