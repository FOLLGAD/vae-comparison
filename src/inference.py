import torch
from data import CIFAR10Dataset
from torch.utils.data import DataLoader
from vae import VAE
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

ENABLE_MPS = False
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and ENABLE_MPS
    else "cpu"
)

# Load the trained VAE model
vae = VAE(in_channels=3, input_shape=(3, 32, 32))
vae.to(device)
vae.eval()

checkpoint_path = "checkpoints/vae_epoch_9.pth"
vae.load_state_dict(torch.load(checkpoint_path, map_location=device))


dataloader = DataLoader(CIFAR10Dataset("train"), shuffle=True, batch_size=1)


# Function to visualize original and reconstructed images
def visualize_reconstruction(originals, reconstructeds):
    fig, axes = plt.subplots(2, 20)
    for idx, (original, reconstructed) in enumerate(zip(originals, reconstructeds)):
        axes[0, idx].imshow(original)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(reconstructed)
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.show()


# Perform inference on 40 samples
original_images = []
reconstructed_images = []

for i, (image_tensor, _) in enumerate(dataloader):
    if i >= 20:
        break

    image_tensor = image_tensor.to(device)
    image_tensor = rearrange(image_tensor, "1 w h c -> 1 c w h")

    with torch.no_grad():
        reconstructed_tensor = vae(image_tensor)
        reconstructed_image = (
            reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        )

    original_images.append(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    reconstructed_images.append(reconstructed_image)

visualize_reconstruction(original_images, reconstructed_images)
