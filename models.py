"""
Autoencoders used as feature extractors.

MaterialAE encodes a whole image into a global embedding z in R^256 and
describes the spatial organization of the microstructure. PatchAE encodes
each non overlapping patch into a local embedding z in R^128 and describes
the local texture. Both are trained only on real images.

Both networks share the same layout:

    encoder   stages of Conv2d(kernel 4, stride 2, padding 1) + ReLU,
              each stage halves the height and width
    fc_enc    linear layer from the last feature map to z
    fc_dec    linear layer from z back to the last feature map
    decoder   stages of ConvTranspose2d(kernel 4, stride 2, padding 1) + ReLU,
              with a Sigmoid on the output so pixels stay in [0, 1]

Standard datasets use 128 x 128 images, 32 x 32 patches and four stages
with 32, 64, 128 and 256 channels, so the last feature map is 8 x 8 for
MaterialAE and 2 x 2 for PatchAE. The H13 dataset uses 256 x 256 images,
64 x 64 patches and a fifth stage with 256 channels, which keeps the same
map sizes.
"""

import torch.nn as nn


def encoder(deep):
    """Convolutional encoder, four stages or five if deep."""
    channels = [3, 32, 64, 128, 256] + ([256] if deep else [])
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv2d(c_in, c_out, 4, 2, 1), nn.ReLU()]
    return nn.Sequential(*layers)


def decoder(deep):
    """Mirror of the encoder, ending in 3 channels and a Sigmoid."""
    channels = ([256] if deep else []) + [256, 128, 64, 32]
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.ReLU()]
    layers += [nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """
    zdim   size of the embedding z
    s      side of the last feature map (8 for MaterialAE, 2 for PatchAE)
    deep   five stages instead of four (H13 dataset)
    """

    def __init__(self, zdim, s, deep=False):
        super().__init__()
        self.s = s
        self.enc = encoder(deep)
        self.fc_enc = nn.Linear(256 * s * s, zdim)
        self.fc_dec = nn.Linear(zdim, 256 * s * s)
        self.dec = decoder(deep)

    def encode(self, x):
        """Embedding z of a batch of images or patches."""
        return self.fc_enc(self.enc(x).flatten(1))

    def forward(self, x):
        """Reconstruction of x, used only for training."""
        z = self.fc_dec(self.encode(x))
        return self.dec(z.view(-1, 256, self.s, self.s))


def MaterialAE(deep=False):
    """Global autoencoder, one embedding of size 256 per image."""
    return Autoencoder(256, 8, deep)


def PatchAE(deep=False):
    """Local autoencoder, one embedding of size 128 per patch."""
    return Autoencoder(128, 2, deep)


def extract_patches(x, p):
    """
    Split a batch [B, C, H, W] into non overlapping p x p patches.

    Returns a tensor [B * K, C, p, p] ordered by image, then patch row, then
    patch column, with K = (H / p) * (W / p) = 16 patches per image. The
    channels of each patch stay together.
    """
    B, C, H, W = x.shape
    u = x.unfold(2, p, p).unfold(3, p, p)               # [B, C, rows, cols, p, p]
    return u.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, p, p)
