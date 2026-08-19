# Yaw Rate and Yaw Estimation

This repository contains the scripts related to the presented project on **yaw rate and yaw estimation**.

## Project Structure

- **parameters/**
  Vehicle parameters (`params.csv`) and dataset statistics (`data_info.csv`) used to build and normalize the models.

- **telemetries/**
  Preprocessed datasets, split into `training/`, `validation/` and `test/`.

- **build_datasets.py**
  Downloads the raw telemetry data (from *Revs*), preprocesses and downsamples it, and writes the split datasets to `telemetries/` and the vehicle/dataset parameters to `parameters/`.

- **MSNN.py**
  Defines the **Model-Structured Neural Network (MS-NN)**: FIR dynamics combined with a fuzzy local-model understeer correction. Contains three functions: `initialize_model` (network structure), `train_model` (training), and `analyse_model` (inference/evaluation).

- **NN.py**
  Defines the **black-box Neural Network (BB-NN)** baseline: raw sliding-window signals (speed, acceleration, steering), directly concatenated, feeding a plain MLP — no FIR or fuzzy preprocessing. Same three-function structure as `MSNN.py`.

- **main_MSNN.py** / **main_NN.py**
  Entry-point scripts for the MS-NN and BB-NN models respectively. Each loops over `N_MODELS` random seeds, and two flags select the mode:
  - `train = True`: trains `N_MODELS` fresh models (via `initialize_model` + `train_model`) and saves a checkpoint per seed to `trained_models/`.
  - `analyse = True`: loads the `N_MODELS` pretrained checkpoints from `trained_models/` and reports the ensemble yaw / yaw-rate RMSE (mean ± standard error), saving a prediction plot for seed 0.

- **plots.py**
  Loads one trained MS-NN and one trained BB-NN checkpoint and plots their yaw angle / yaw rate predictions against measured test data, saving `Fig/results.pdf`.

- **trained_models/**
  Pretrained model checkpoints (`.json`), one per ensemble member, loaded by the `analyse = True` path of `main_MSNN.py` / `main_NN.py` and by `plots.py`.

- **Fig/**
  Figures and results obtained from the experiments.

## How to Run the Project

1. Run `build_datasets.py` to download and preprocess the raw telemetry into `telemetries/` (skip this step if `telemetries/` is already populated, as delivered here).

2. Run `main_MSNN.py` or `main_NN.py`:
   - Set `train = True` to retrain the ensemble from scratch (saves new checkpoints to `trained_models/`).
   - Set `analyse = True` to load the pretrained checkpoints in `trained_models/` and compute ensemble RMSE statistics.

3. Run `plots.py` to reproduce the prediction comparison plot in `Fig/results.pdf`.

## Notes

- Make sure all required Python dependencies (including `nnodely`) are installed before running the scripts.
