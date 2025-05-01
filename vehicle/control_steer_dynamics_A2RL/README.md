# Model-Structured Neural Networks to Control the Steering Dynamics of Autonomous Race Cars

## Description

This repository contains the code for the paper titled "*Model-Structured Neural Networks to Control the Steering Dynamics of Autonomous Race Cars*".

## Structure

The repository is structured as follows:

- file `control_steer_dynamics_A2RL.ipynb`: contains the code of our model-structured neural network `MS-NN-steer`. Our model is implemented using the `nnodely` open-source library, which is available at: https://github.com/tonegas/nnodely
- file `main_sequential_training.py`: contains the code to train the model-structured neural network `MS-NN-steer` multiple times, calling the file `control_steer_dynamics_A2RL_sequential.ipynb`, using different hyperparameters. This is used to find the best hyperparameters for the model, with a grid search.
- folder `A2RL_Data/`: contains the datasets used to train and validate our model.

## Requirements

The software requirements are reported in the `requirements.txt` file in the root directory.

## Use

- create a new virtual environment with Python 3.10, activate the environment and install the required packages by running the following command:
```bash
conda create -n <env_name> python=3.10  # replace <env_name> with the desired name
conda activate <env_name>
pip install -r requirements.txt
```
- run the script `control_steer_dynamics_A2RL.ipynb` to train the `MS-NN-steer`, or the script `main_sequential_training.py` to optimize the hyperparameters of the model. 

## Repo Owners

- Mattia Piccinini, Aniello Mungiello, Gastone Pietro Rosati Papini
- Contact: mattia.piccinini@tum.de, aniello.mungiello@unina.it, gastone.rosatipapini@unitn.it

