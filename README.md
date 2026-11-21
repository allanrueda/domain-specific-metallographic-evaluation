# Domain-specific multi-scale evaluation of GAN-generated metallographic images

This repository contains the evaluation protocol of the paper

> A. B. Rueda-Gutiérrez, A. Rodríguez-Molina, M. Aldape-Pérez, O. A. Gómez-Vargas, S. S. Roblero-Aguilar. *Domain-Specific Multi-Scale Metrics for Evaluating GAN-Generated Microstructural Images with Limited Data via Autoencoder Embeddings.* Science and Technology of Advanced Materials: Methods (under review).

Two convolutional autoencoders are trained on real micrographs and used as feature extractors:

- **MaterialAE** gives one global embedding per image (256 values) and describes the spatial organization of the microstructure.
- **PatchAE** gives one local embedding per non-overlapping patch (128 values, 16 patches per image) and describes the local texture.

Real and generated images are compared in both spaces with the normalized Fréchet distance (nFID), the squared MMD with a Gaussian kernel, KID, and k-NN precision and recall. The uncertainty is estimated at the level of independent micrographs, and the scores are calibrated against the distance between two halves of the real images.

## Files

| File | Content |
| --- | --- |
| `models.py` | Autoencoder architectures and patch extraction |
| `metrics.py` | Distances, bootstrap and real vs real reference |
| `evaluate.py` | Complete protocol for one dataset, with all settings at the top |
| `units/<dataset>.json` | Independent unit (original micrograph) of each real image |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

For an NVIDIA GPU, install PyTorch first with the command for your CUDA version from https://pytorch.org/get-started/locally/. Without a GPU, the code runs on the CPU (or on Apple silicon GPUs).

## Data

The images are not included in this repository and are available from the authors on reasonable request. Each dataset is expected as

```
data/<dataset>/real/             real images
data/<dataset>/dcgan/            generated images, one folder per GAN
data/<dataset>/wgan-gp/
data/<dataset>/stylegan2-ada/
```

with `<dataset>` one of `inclusions`, `corrosion`, `boriding` and `h13`. The file names of the real images must match those in `units/<dataset>.json`.

Images derived from the same micrograph are not independent, so each real image is assigned to a unit:

| Dataset | Images | Units | How the units were identified |
| --- | --- | --- | --- |
| inclusions | 490 | 456 | near-duplicate and overlapping regions, by pixel similarity and phase correlation |
| corrosion | 382 | 189 | reflected copies by pixel similarity |
| boriding | 600 | 75 | file name, eight overlapping crops per micrograph |
| h13 | 1000 | 28 | file name, 36 rotations per micrograph |

The 80/20 training/validation split and the bootstrap are made by unit.

## Usage

```
python evaluate.py corrosion
```

For each of the five seeds (42, 1042, 2042, 3042, 4042), the script trains both autoencoders with early stopping, embeds the real and generated images, and computes the scores with their bootstrap distribution. A training whose validation error is not clearly below that of the mean training image is restarted.

## Output

- `results/<dataset>_summary.txt`: the values reported in the paper. These are the scores with their standard deviation over trainings and their bootstrap standard deviation, the reference (Table 12), the ranking stability (Table 13), the ratio to the reference (Table 14), precision and recall (Table 15), and the autoencoder training checks and cost (Tables 7 and 10).
- `results/<dataset>.json`: all values per seed, including the training curves and the bootstrap replicates.

The settings in `evaluate.py` are those of the paper. Results on another GPU can differ in the last digits.

## Citation

If you use this code, please cite the paper above.
