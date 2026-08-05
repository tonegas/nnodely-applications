# Friction Z1

This repository contains the code accompanying the paper on friction modeling for the Z1 robot arm. It includes the data-driven models used in the study, together with inference and visualization scripts.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Contents

- `msnn.py`: trains the MSNN-based model.
- `msnn.py`: trains the regression model.
- `recurrent_blackbox.py`: trains the LSTM baseline.
- `inference.py`: runs inference and saves trajectory predictions.
- `plot_results.py`: compares model predictions against ground truth.
- `run_test.sh`: runs inference for every model and generates the comparison plot.

## Data and outputs

The dataset is stored under `dataset/`, the `cmame/` folders contain the data used in the paper, and trained models are saved in `model/`. Generated results are written to the corresponding `results/` folders.

## Usage

Train the main model:

```bash
python msnn.py
```

Run inference with a saved model:

```bash
python inference.py --model msnn
```

Then plot the results:

```bash
python plot_results.py
```

To run inference for all available models and then plot the comparison in one step:

```bash
bash run_test.sh
```
