import gymnasium as gym
import numpy as np
from numpy import sin, cos
import csv
import os


# 1.  creo l’ambiente di simulazione
env = gym.make("Reacher-v5")
obs, info = env.reset()

# 2. 
l1 = 0.1
l2 = 0.1
sim = 2   # tipo si simu

# 3.  apro il file CSV per salvare i dati
folder_path = os.path.join(os.getcwd(), "dataset", "data")
os.makedirs(folder_path, exist_ok=True)
file_path = os.path.join(folder_path, "reacher_data.csv")

file = open(file_path, "w", newline="")   # overwrite if exists
writer = csv.writer(file, delimiter=";")
#writer.writerow(["step", "l1", "l2", "T1", "T2", "theta1", "theta2", "x_tip", "y_tip", "reward"])
#writer.writerow(["step", "l1", "l2", "T1", "T2", "theta1", "theta2", "x_tip", "y_tip"])


dt = 0.02   # or use env.dt if you know the actual time step
thetadot1_prev = 0
thetadot2_prev = 0



# 4.  eseguo la simulazione per 200 passi
for i in range(2000):
    # --- Step inputs (torques) ---   # definisco le coppie motrici per ogni passo

    if sim == 1:
        T2 =  (.002 * sin(i / 320000)); T1 = (.001 * sin(i / 320000))
        #l1 = 1.0 + 0.5 * sin(i / 500); l2 = 1.0 + 0.5 * cos(i / 500)   # l1 e l2 cambiano con il tempo
    elif sim == 2:  # to define   # da definire
        T2 =  (.002 * sin(i / 80000)); T1 = (.001 * sin(i / 80000))
        #l1 = 1.0 + 0.5 * sin(i / 50); l2 = 1.0 + 0.5 * sin(i / 50)
    elif sim ==3:   
        T2 =  (.001 * sin(i / 300)); T1 = (.001 * cos(i / 200))  
    

    #  creo un array con le due coppie
    action = np.zeros(2, dtype=np.float32)
    action[0] = T1
    action[1] = T2

    #  faccio un passo nella simulazione
    next_obs, reward, terminated, truncated, info = env.step(action)

    #  estraggo i valori 
    cos1 = next_obs[0]
    cos2 = next_obs[1]
    sin1 = next_obs[2]
    sin2 = next_obs[3]
    fingertip_x = next_obs[4] + next_obs[8]
    fingertip_y = next_obs[5] + next_obs[9]
    thetadot1 = next_obs [6]
    thetadot2 = next_obs [7]


        # --- Angular acceleration (finite difference) ---
    if i > 0:
        thetaddot1 = (thetadot1 - thetadot1_prev) / dt
        thetaddot2 = (thetadot2 - thetadot2_prev) / dt
    else:
        thetaddot1 = 0
        thetaddot2 = 0

    # update previous values
    thetadot1_prev = thetadot1
    thetadot2_prev = thetadot2


    #  calcolo gli angoli 
    theta1 = np.arctan2(sin1, cos1)
    theta2 = np.arctan2(sin2, cos2)

    # Fingertip pos(x, y) posizione finale del braccio (coordinate x e y)
    x_tip = (l1 * cos(theta1)) + (l2 * cos(theta1 + theta2))
    y_tip = (l1 * sin(theta1)) + (l2 * sin(theta1 + theta2))

    """"" Round 
    #  salvo i dati nel file CSV
    l1 = round(l1, 4)
    l2 = round(l2, 4)
    T1 = round(T1, 4)
    T2 = round(T2, 4)
    theta1 = round(theta1, 4)
    theta2 = round(theta2, 4)
    x_tip = round(x_tip, 4)
    y_tip = round(y_tip, 4)
    reward = round(reward, 4)
    fingertip_x=round(fingertip_x,4)
    fingertip_y=round(fingertip_y, 4)
    thetadot1=round(thetadot1, 4)
    thetadot2=round(thetadot2, 4)
    """""""""

    writer.writerow([
        i, T1, T2, theta1, theta2,
        fingertip_x, fingertip_y,
        thetadot1, thetadot2,
        thetaddot1, thetaddot2
    ])



#   writer.writerow([i, T1, T2, theta1, theta2, x_tip, y_tip, reward])

# 5.  # chiudo il file e l’ambiente
file.close()
env.close()
