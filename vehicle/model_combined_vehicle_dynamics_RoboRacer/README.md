# Model-Structured Neural Networks for Vehicle Dynamics Learning

## Description

This repository contains the code for the paper titled "*Model-Structured Neural Networks for Vehicle Dynamics Learning Near the Limits*".

## Structure

The repository is structured as follows:

- file `Closed_Loop_Dynamics.ipynb`: contains the code of our model-structured neural network `MS-NN-full`
- file `Lateral_Dynamics.ipynb`: contains the code of our model-structured neural network `MS-NN-lat`
- file `Longitudinal_Dynamics.ipynb`: contains the code of our model-structured neural network `MS-NN-long`
- files `Ablation_1_Lateral.ipynb`, `Ablation_2_Lateral.ipynb`, `Ablation_1_Longitudinal.ipynb`: contain the code for the ablation studies on the lateral and longitudinal dynamics
- files `Benchmark_Lateral.ipynb`, `Benchmark_Longitudinal.ipynb`, `General_Purpose_Neural_Networks.ipynb`: contain the code for the benchmark models 
- folder `Dataset/`: contains the datasets used to train and validate our models
- folder `trained_models/`: contains the trained models

All models are implemented using the `nnodely` open-source library, which is available at: https://github.com/tonegas/nnodely

## Requirements

The software requirements are reported in the `requirements.txt` file in the root directory.

## Use

- create a new virtual environment with Python 3.10, activate the environment and install the required packages by running the following command:
```bash
conda create -n <env_name> python=3.10  # replace <env_name> with the desired name
conda activate <env_name>
pip install -r requirements.txt
```
- run the scripts to train the corresponding models

## Repo Owners

- Aniello Mungiello, Mattia Piccinini, Gastone Pietro Rosati Papini
- Contact: aniello.mungiello@unina.it, mattia.piccinini@tum.de, gastone.rosatipapini@unitn.it

