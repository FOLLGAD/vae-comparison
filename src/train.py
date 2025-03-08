from torch.utils.data import DataLoader
from einops import rearrange
import torch
from tqdm import tqdm
from data import CIFAR10Dataset
from vae import VAE
import os
from time import time

fingerprint = f"{round(time())}"
outdir = f"checkpoints/run-{fingerprint}"
os.makedirs(outdir, exist_ok=True)


ENABLE_MPS = False
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and ENABLE_MPS
    else "cpu"
)

vae = VAE(in_channels=3, input_shape=(3, 32, 32))
vae.to(device)
vae.train()

optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
β = 1


dataloader = DataLoader(CIFAR10Dataset("train"), shuffle=True, batch_size=32)

for epoch in range(10):
    total_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
        for x, y in pbar:
            x = x.to(device)
            x = rearrange(x, "b w h c -> b c w h")

            y = y.to(device).float()
            recon = vae(x)

            recon_loss = torch.nn.functional.mse_loss(recon, x)
            kl_loss = -0.5 * torch.sum(
                1
                + vae.logvar_layer.weight
                - vae.mu_layer.weight.pow(2)
                - vae.logvar_layer.weight.exp()
            )
            kl_loss /= x.size(0)  # Normalize by batch size

            loss = recon_loss + kl_loss * β
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Recon Loss": recon_loss.item(),
                    "KL Loss": kl_loss.item(),
                }
            )

        loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch}, Loss: {loss}")

        torch.save(vae.state_dict(), f"{outdir}/vae_epoch_{epoch}.pth")
