import os
from nnodely import nnodely, MPLVisualizer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main(args):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATASET_NAME = "dataset"
    DATASET_FOLDER = os.path.join(ROOT, DATASET_NAME, "test")
    df = pd.read_csv(os.path.join(DATASET_FOLDER, 'mpc_sim_data_0.csv'))

    q = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].to_numpy()
    MODEL_DIR = os.path.join(ROOT, 'model')
    MODEL = args.model
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL)
    SEED = np.random.randint(0, 10000)

    z1_inference = nnodely(visualizer=MPLVisualizer(), workspace=MODEL_DIR, seed=SEED)
    z1_inference.loadModel(name=MODEL, model_folder=MODEL_DIR)
    z1_inference.neuralizeModel(0.01)

    data_struct = ['time', ('q1', 'q1_int'), ('q2', 'q2_int'), ('q3', 'q3_int'), ('q4', 'q4_int'), ('q5', 'q5_int'), ('q6', 'q6_int'),
                ('dq1', 'dq1_int'), ('dq2', 'dq2_int'), ('dq3', 'dq3_int'), ('dq4', 'dq4_int'), ('dq5', 'dq5_int'), ('dq6', 'dq6_int'),
                'ddq1', 'ddq2', 'ddq3', 'ddq4', 'ddq5', 'ddq6',
                'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6']

    z1_inference.loadData(
        name='test',
        source=DATASET_FOLDER,
        format=data_struct,
        skiplines=1
    )

    WINDOW_SIZE = 499
    PREDICTION_HORIZON = 50

    samples_test = z1_inference.getSamples('test', window=WINDOW_SIZE, index=0)

    out_nn_test_set = z1_inference(samples_test, sampled=True, prediction_samples=PREDICTION_HORIZON)

    computed_q = out_nn_test_set['q_out']
    computed_q = np.array(computed_q).squeeze(1)

    np.savez(f"results/numeric/results_{MODEL}.npz", computed_q=computed_q, real_q=q)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for MSNN")
    parser.add_argument('--model', type=str, default='msnn', help='Name of the model to load')
    args = parser.parse_args()
    main(args)