# from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from nd_vq_vae import NDimVQVAE
import torch
import torch.nn as nn

class DiffusersVAEWrapper(nn.Module):
    def __init__(self, model_id="stabilityai/sd-vae-ft-mse", device="cpu"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id, local_files_only=True).to(device)
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
    
# class WrappedNDimVQVAE(torch.nn.Module):
#     def __init__(self, 
#                 embedding_dim=64,
#                 n_codes=64,
#                 n_dims=3,           # 3D: time, height, width
#                 downsample=(2, 2, 2),
#                 n_hiddens=64,
#                 n_res_layers=2,
#                 codebook_beta=0.25,
#                 input_shape=input_shape,):
#         super().__init__()
#         self.vqvae = NDimVQVAE(
#                 embedding_dim=64,
#                 n_codes=64,
#                 n_dims=3,           # 3D: time, height, width
#                 downsample=(2, 2, 2),
#                 n_hiddens=64,
#                 n_res_layers=2,
#                 codebook_beta=0.10,
#                 input_shape=input_shape
#                 )

#     def encode(self, x):
#         _, _, vq_out = self.vqvae(x)
#         return vq_out["z"]

#     def decode(self, z):
#         return self.vqvae.decoder(z)