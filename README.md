# Enhancing Generative Capabilities of Diffusion Models

This repository contains the implementation of the thesis "Enhancing Generative Capabilities of Diffusion Models: Evaluating Particle Guidance and Stochastic Sampling" by Julian Kleutgens. The project explores methods to improve the diversity and quality of generated outputs using diffusion models, with applications in sequential data and segmentation tasks.

## Overview

The code is designed to:
- Train and evaluate diffusion models using synthetic datasets (e.g., MovingMNIST).
- Implement stochastic sampling with Particle Guidance (PG) for non-I.I.D. diverse sampling.
- Explore the effects of schedulers, kernels, and hyperparameters on the performance of diffusion models.

Key components include:
- Unconditional generation experiments (e.g., digit diversity in MNIST).
- Conditional generation experiments (e.g., predicting next frames in sequences).
- Implementation of Particle Guidance with customizable kernels.

## Datasets

This repository primarily uses synthetic datasets, including:
1. **MovingMNIST**: A dataset of digits with deterministic and stochastic movement.
2. **Circle Jumping Dataset**: Digits move to predefined directions in a circular arrangement.
3. **Horizontal Movement Dataset**: Digits move left or right with configurable probabilities.

The datasets are generated dynamically using the provided data loaders.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU with NVIDIA drivers (for training and inference).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/juliankleutgens/edm_mnist.git
   cd diffusion-particle-guidance
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install wandb torchvision torchsummary
   ```


## Key Scripts

- `train.py`: Script for training the diffusion model.
- `evaluate.py`: Script for evaluating the trained models.
- `data_loader.py`: Contains dataset generation logic for MovingMNIST and custom datasets.
- `sampler.py`: Implements stochastic sampling with Particle Guidance.
- `model.py`: Defines the architecture for the EDM-inspired diffusion model.

## Results

### Unconditional Results
- The model successfully generated diverse MNIST digits, capturing all modes with optimized settings for Particle Guidance and stochastic sampling.

### Conditional Results
- The model predicted future frames in multimodal cases with controlled diversity using Particle Guidance.

### Metrics
Evaluation metrics include:
- Number of modes discovered (unique outcomes).
- Image quality scores (e.g., L2 distance, IoU).
- Directional accuracy for sequential datasets.

## Reproducing Results

1. Train the model using the provided scripts and configurations.
2. Run the evaluation scripts to generate and analyze results.
3. Use the visualization scripts to plot metrics, heatmaps, and UMAP embeddings.

Example command to reproduce results for Particle Guidance:
```bash
python evaluate.py --task conditional --dataset circle_jumping --pg_factor 1.7 --batch_size 8
```

## References
- [Diffusion Models Overview](https://arxiv.org/abs/2105.05233)
- [EDM Paper](https://arxiv.org/abs/2205.11487)
- [Particle Guidance](https://arxiv.org/abs/2301.10215)

## Contact
For questions or collaborations, feel free to reach out to Julian Kleutgens at `jkleutgens@ethz.ch`.
