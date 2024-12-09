import numpy as np
from torch import nn
import torch


class VAE(nn.Module):
    def __init__(self, input_shape, in_channels, latent_dim=16):
        super(VAE, self).__init__()

        size = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(size * 2),
            nn.ReLU(),
            nn.Conv2d(size * 2, size * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(size * 4),
            nn.ReLU(),
            nn.Conv2d(size * 4, size * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(size * 8),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out_size(input_shape)

        self.mu_layer = nn.Linear(conv_out_size, latent_dim)
        self.logvar_layer = nn.Linear(conv_out_size, latent_dim)

        self.predecode = nn.Linear(latent_dim, conv_out_size)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (size * 8, input_shape[1] // 16, input_shape[2] // 16)),
            nn.ConvTranspose2d(size * 8, size * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(size * 4, size * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(size * 2, size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(size, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_func)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def encode(self, x):
        h1 = self.encoder(x)
        mu, logvar = self.mu_layer(h1), self.logvar_layer(h1)

        return self.reparameterize(mu, logvar)

    def decode(self, z):
        h3 = self.predecode(z)
        return self.decoder(h3)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y

    def _get_conv_out_size(self, shape):
        out = self.encoder(torch.zeros(1, *shape))
        self.conv_out_shape = out.size()
        return int(np.prod(self.conv_out_shape))
