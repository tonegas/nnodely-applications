# Double Inverted Pendulum on Cart

`nnodely` application for learning the dynamics of Gymnasium's
`InvertedDoublePendulum-v5` environment.

## Scripts

- `DP_dataset_creation.py`: generates simulation CSV files.
- `nnodely_DP.py`: trains a physics-based model with RK4 integration.
- `DP_black_box.py`: trains a black-box neural model.
- `DP_equation_learner.py`: trains an Equation Learner model.
- `DP_localmodel.py`: trains a fuzzy LocalModel.

Data is stored in `data/data_DP/` and test data in `data/data_DP_test/`.

## Setup

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../../requirements.txt
pip install gymnasium[mujoco] torch scikit-learn tqdm
```

## Run

```bash
python DP_dataset_creation.py
python nnodely_DP.py
python DP_black_box.py
python DP_equation_learner.py
python DP_localmodel.py
```

The training scripts run multiple dataset fractions and random seeds, so they
may require significant time and memory.

## Known path issue

`DP_black_box.py`, `DP_equation_learner.py`, and `DP_localmodel.py` currently
refer to `DIPC/data/...`. Replace those paths with `data/data_DP/...` before
running them, or provide a compatible `DIPC` directory.

## References

- [nnodely](https://github.com/nnodely/nnodely)
- [Gymnasium InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
