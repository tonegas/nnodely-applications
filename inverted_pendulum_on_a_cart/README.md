# Inverted Pendulum on a Cart

This repository contains the code for the inverted pendulum on a cart example, and it includes the Model Structured Neural Networks used in the study.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Contents

- `learned_fir_filter_model.ipynb`: Python notebook regarding the LFIR model.
- `learned_fir_filter_model_noise.ipynb`: Python notebook regarding the LFIR model with noisy inputs.
- `partially_neuralized_model.ipynb`: Python notebook regarding the PNM model.
- `physics_augmented_neural_network.ipynb`: Python notebook regarding the PANN model.
- `physics_based_model.ipynb`: Python notebook regarding the $\text{PBM}_G$ model.

## Data and outputs

The datasets are divided in Training, Validation, and Test respectively in the folders `data/`, `data_val/`, and `data_test/`. Trained model with and without added noise are saved in `saved_models/`.
