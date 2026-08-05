import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import utils

plt.style.use(utils.get_style())
plt.rcParams.update(utils.get_tex_fonts())
plt.rcParams.update(utils.get_palette(use_deep_cycle=True))
plt.rcParams.update(utils.get_plot_params())

# =========================
# LOAD DATA
# =========================
msnn_data = np.load("results/numeric/results_msnn.npz")
computed_q = msnn_data["computed_q"]
real_q_msnn = msnn_data["real_q"][:-1]
computed_q[:51, :] = real_q_msnn[:51, :] 
regression_q = np.load("results/numeric/results_regression.npz")["computed_q"]
regression_q[:51, :] = real_q_msnn[:51, :]
lstm_data = np.load("results/numeric/results_lstm.npz")
lstm_q = lstm_data["computed_q"][:-1]
real_q_lstm = lstm_data["real_q"][:-1]
print("Shapes debug:")
print(f"MSNN computed_q shape: {computed_q.shape}")
print(f"MSNN real_q shape: {real_q_msnn.shape}")
print(f"Regression computed_q shape: {regression_q.shape}")
print(f"LSTM computed_q shape: {lstm_q.shape}")
print(f"LSTM real_q shape: {real_q_lstm.shape}")
assert np.all(real_q_msnn[0]) == np.all(real_q_lstm[0]), "Ground truth trajectories must be the same for fair comparison"

# =========================
# CONFIG                   
# =========================
SEQ_LEN = 20
dt = 0.01

# =========================
# ALIGNMENT 
# =========================
time = np.arange(lstm_q.shape[0]) * dt

# =========================
# PLOT TRAJECTORIES
# =========================
# MODIFICA QUI: allarga la larghezza e riduci l'altezza
width, height = utils.set_size(1, (3, 2))
fig, axes = plt.subplots(3, 2, figsize=(width * 1, height * 0.65))  # <- larghezza +30%, altezza -35%
axes = axes.flatten()

width = fig.get_size_inches()[0]
lw = utils.linewidth_from_size(fig_width_in=width, base_width=0.5 * width)

for i in range(6):
    ax = axes[i]

    ax.plot(time, computed_q[:, i],
            label="MSNN", linestyle="-.", linewidth=lw)

    ax.plot(time, regression_q[:, i],
            label="Regression", linestyle="--", linewidth=lw)

    ax.plot(time, lstm_q[:, i],
            label="LSTM", linestyle=":", linewidth=lw)
    
    ax.plot(time, real_q_msnn[:, i],
            label="Ground truth", color="black", linewidth=0.2)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.grid(True, which="minor", alpha=0.3)
    ax.grid(True, which="major", alpha=0.7)
    ax.set_xlim(left=0)

    ax.set_ylabel(utils.make_label(f"q_{i+1}", "rad"))

    if i >= 4:
        ax.set_xlabel(utils.make_label("Time", "s"))
    else:
        ax.tick_params(labelbottom=False)

plt.tight_layout()
axes[0].legend()
fig.savefig("results/img/try.pdf", bbox_inches='tight')