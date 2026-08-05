import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import tqdm

# =========================
# CONFIG
# =========================
SEQ_LEN = 20
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
DT = 0.01
HORIZON = 50

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(ROOT, "dataset", "train")
EVAL = False

# =========================
# DATASET
# =========================
class SequenceDataset(Dataset):
    def __init__(self, q, dq, tau, ddq, seq_len=20):
        self.q = q
        self.dq = dq
        self.tau = tau
        self.ddq = ddq
        self.seq_len = seq_len

    def __len__(self):
        return len(self.q) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.q[idx:idx+self.seq_len],
            self.dq[idx:idx+self.seq_len],
            self.tau[idx:idx+self.seq_len],
            self.ddq[idx:idx+self.seq_len]
        )

# =========================
# MODEL
# =========================
class FrictionModelLSTM(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, num_layers=1, output_dim=6):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, q, dq, tau):
        x = torch.cat((q, dq, tau), dim=-1)
        out, _ = self.lstm(x)
        return self.fc(out)

class BBModel(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, output_dim=6, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, q, dq, tau):
        x = torch.cat((q, dq, tau), dim=-1)
        return self.fc(x)



# =========================
# LOAD DATA
# =========================
def load_data(file, stats=None):
    data = pd.read_csv(os.path.join(DATASET_FOLDER, file))

    q = torch.tensor(data[[f'q{i}' for i in range(1,7)]].values, dtype=torch.float32)
    dq = torch.tensor(data[[f'dq{i}' for i in range(1,7)]].values, dtype=torch.float32)
    tau = torch.tensor(data[[f'tau{i}' for i in range(1,7)]].values, dtype=torch.float32)
    ddq = torch.tensor(data[[f'ddq{i}' for i in range(1,7)]].values, dtype=torch.float32)

    if stats is None:
        q_mean, q_std = q.mean(0), q.std(0) + 1e-8
        dq_mean, dq_std = dq.mean(0), dq.std(0) + 1e-8
        tau_mean, tau_std = tau.mean(0), tau.std(0) + 1e-8
        ddq_mean, ddq_std = ddq.mean(0), ddq.std(0) + 1e-8
    else:
        q_mean, q_std = stats['q_mean'], stats['q_std']
        dq_mean, dq_std = stats['dq_mean'], stats['dq_std']
        tau_mean, tau_std = stats['tau_mean'], stats['tau_std']
        ddq_mean, ddq_std = stats['ddq_mean'], stats['ddq_std']

    q = (q - q_mean) / q_std
    dq = (dq - dq_mean) / dq_std
    tau = (tau - tau_mean) / tau_std
    ddq_norm = (ddq - ddq_mean) / ddq_std
    
    stats_out = {
        'q_mean': q_mean, 'q_std': q_std,
        'dq_mean': dq_mean, 'dq_std': dq_std,
        'tau_mean': tau_mean, 'tau_std': tau_std,
        'ddq_mean': ddq_mean, 'ddq_std': ddq_std
    }

    return q, dq, tau, ddq_norm, ddq, stats_out

# =========================
# TRAIN
# =========================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for q, dq, tau, ddq in loader:
        optimizer.zero_grad()
        pred = model(q, dq, tau)
        loss = criterion(pred, ddq)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion): 
    model.eval() 
    total_loss = 0 

    all_preds = []
    all_targets = []

    with torch.no_grad(): 
        for q, dq, tau, ddq in loader: 
            pred = model(q, dq, tau)

            # Score only the last timestep so the evaluation target has seq_len = 1.
            pred = pred[:, -1:, :]
            ddq = ddq[:, -1:, :]

            loss = criterion(pred, ddq) 
            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(ddq.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    ss_res = np.sum((preds - targets) ** 2, axis=(0, 1))
    ss_tot = np.sum((targets - targets.mean(axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))

    r2 = 1 - ss_res / ss_tot

    return r2


# =========================
# ROLLOUT PREDICTION
# =========================
def rollout_prediction(model, q, dq, tau, ddq, stats):
    model.eval()

    q_mean, q_std = stats['q_mean'], stats['q_std']
    dq_mean, dq_std = stats['dq_mean'], stats['dq_std']
    ddq_mean, ddq_std = stats['ddq_mean'], stats['ddq_std']

    preds, gts = [], []
    # add first 20 steps
    for i in range(SEQ_LEN):
        q_init = q[i] * q_std + q_mean
        preds.append(q_init.cpu().numpy())
        gts.append(q_init.cpu().numpy())
    T = q.shape[0]
    i = 0
    HORIZON = 50
    with torch.no_grad():
        while i + HORIZON < T:

            print(i+SEQ_LEN, "/", T)

            q_buf = q[i:i+SEQ_LEN].clone()
            dq_buf = dq[i:i+SEQ_LEN].clone()
            tau_buf = tau[i:i+SEQ_LEN].clone()

            q_init = q_buf[-1] * q_std + q_mean
            preds.append(q_init.cpu().numpy())
            gts.append(q_init.cpu().numpy())
            if i == 0:
                HORIZON = 30
            for h in range(1, HORIZON):
                print("\t", i+SEQ_LEN+h, "/", T)
                pred = model(
                    q_buf.unsqueeze(0),
                    dq_buf.unsqueeze(0),
                    tau_buf.unsqueeze(0)
                )[:, -1, :]

                pred_denorm = pred * ddq_std + ddq_mean

                q_last = q_buf[-1] * q_std + q_mean
                dq_last = dq_buf[-1] * dq_std + dq_mean

                dq_next = dq_last + pred_denorm.squeeze(0) * DT
                q_next = q_last + dq_last * DT + 0.5 * pred_denorm.squeeze(0) * DT**2

                preds.append(q_next.cpu().numpy())

                gt_q = q[i + SEQ_LEN + h] * q_std + q_mean
                gts.append(gt_q.cpu().numpy())

                q_next_norm = (q_next - q_mean) / q_std
                dq_next_norm = (dq_next - dq_mean) / dq_std

                q_buf = torch.cat([q_buf[1:], q_next_norm.unsqueeze(0)], dim=0)
                dq_buf = torch.cat([dq_buf[1:], dq_next_norm.unsqueeze(0)], dim=0)
                tau_buf = torch.cat([tau_buf[1:], tau[i+SEQ_LEN+h].unsqueeze(0)], dim=0)
            i += HORIZON
            HORIZON = 50

    return np.array(preds), np.array(gts)

# =========================
# MAIN
# =========================
nname = "lstm_10_5"
def main():
    # TRAIN DATA
    q, dq, tau, ddq_norm, ddq, stats = load_data('all.csv')

    q_test, dq_test, tau_test, ddq_norm_test, ddq_test, _ = load_data('val.csv')

    train_dataset = SequenceDataset(q, dq, tau, ddq_norm, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FrictionModelLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    losses = []
    if not EVAL:
        print("Training LSTM model...")
        for epoch in tqdm.trange(EPOCHS):
            loss = train(model, train_loader, optimizer, criterion)
            tqdm.tqdm.write(f"Epoch {epoch+1}, Loss: {loss:.6f}")
            losses.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_state = model.state_dict()

        torch.save(best_state, os.path.join(ROOT, "cmame/lstm", f"{nname}.pth"))
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(torch.load(os.path.join(ROOT, "cmame/lstm", f"{nname}.pth")))

    # np.savez("losses_lstm_10_1.npz", losses=losses)

    
    # compute r2 score on test set



    # print("Running horizon rollout...")
    # preds, gts = rollout_prediction(model, q, dq, tau, ddq_norm, stats)



    # print(f"Preds shape: {preds.shape}, GTs shape: {gts.shape}")
    # # evaluate loss on horizon rollout

    # loss = np.mean((preds - gts)**2)
    # print(f"Horizon Rollout MSE Loss: {loss:.6f}")

    r2 = evaluate(model, DataLoader(SequenceDataset(q_test, dq_test, tau_test, ddq_norm_test, 20), batch_size=BATCH_SIZE), criterion)

    max_r2 = np.max(r2)

    # preds = preds.reshape(-1, 6)
    # gts = gts.reshape(-1, 6)

    # os.makedirs("results/numeric", exist_ok=True)
    np.savez(os.path.join(ROOT, "cmame/lstm/results", f"{nname}.npz"), r2=max_r2)

if __name__ == "__main__":
    main()
