# A Road Friction-Aware Anti-Lock Braking System Based on Model-Structured Neural Networks

## Description

This repository contains the code for the paper titled "*A Road Friction-Aware Anti-Lock Braking System Based on Model-Structured Neural Networks*".

## Structure

The repository is structured as follows:

- file `train_MS_NN.py`: contains the code to train the MS-NN models on different road conditions. Our models are implemented using the `nnodely` open-source library, which is available at: https://github.com/tonegas/nnodely
- file `online_friction_estimation.py`: contains the code to test the MS-NN models and perform online friction estimation through robust model selection, by comparing the MSPRT and WTA algorithms.
- folder `trained_models/`: contains the trained MS-NN models, for different road conditions.
- folder `datasets/`: contains the datasets used to train and test the models.

## Requirements

The software requirements are reported in the `requirements.txt` file in the root directory.

## Use

- create a new virtual environment with Python 3.10, activate the environment and install the required packages by running the following command:
```bash
conda create -n <env_name> python=3.10  # replace <env_name> with the desired name
conda activate <env_name>
pip install -r requirements.txt
```
- run the training script `train_MS_NN.py` or the online friction estimation script `online_friction_estimation.py`.

## Repo Owners

- Gastone Pietro Rosati Papini, Mattia Piccinini, Matteo Zumerle
- Contact: gastone.rosatipapini@unitn.it, mattia.piccinini@tum.de, matteo.zumerle@virgilio.it

