import gradio as gr
import torch
from vae import VAE
from einops import rearrange
import numpy as np
import sys

# Load the trained VAE model
device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 64  # This should match the latent_dim used in your VAE
vae = VAE(in_channels=3, input_shape=(3, 32, 32), latent_dim=latent_dim)
vae.to(device)
vae.eval()

if len(sys.argv) < 2:
    raise ValueError("Please provide the checkpoint path as a command-line argument.")

checkpoint_path = sys.argv[1]
vae.load_state_dict(torch.load(checkpoint_path, map_location=device))


def reconstruct_image(latent_vector):
    latent_tensor = torch.tensor(latent_vector, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed_tensor = vae.decode(latent_tensor.unsqueeze(0))
        reconstructed_image = (
            reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        )
    return reconstructed_image


def gradio_interface(image):
    image_tensor = (
        torch.tensor(image, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    image_tensor = rearrange(image_tensor, "1 c w h -> 1 c w h") / 255.0

    with torch.no_grad():
        latent_vector = vae.encode(image_tensor).squeeze(0).cpu().numpy()

    return latent_vector


inputs = gr.inputs.Image(shape=(32, 32))
latent_sliders = [
    gr.inputs.Slider(minimum=-3, maximum=3, default=0, label=f"Latent {i}")
    for i in range(latent_dim)
]
outputs = gr.outputs.Image(type="numpy", label="Reconstructed Image")

iface = gr.Interface(
    fn=lambda image, *latent_values: reconstruct_image(np.array(latent_values)),
    inputs=[inputs] + latent_sliders,
    outputs=outputs,
    live=True,
    title="VAE Image Reconstruction",
    description="Upload an image and adjust the latent sliders to generate new images.",
)

iface.launch()
