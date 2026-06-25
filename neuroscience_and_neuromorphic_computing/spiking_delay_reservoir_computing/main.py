import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import delay_reservoir_computing as drc

# --------------------------------------------------
# Reservoir configuration
# --------------------------------------------------

reservoir_params = {
    "model": 3,          # 1=LIF,2=FHN,3=Izhikevich,4=HH
    "method": "euler",     # euler or rk4
    "theta": 0.2,
    "N_nodes": 50,
    "regularization": 1e-8,
    #Maybe put here other hyperparameters, such as K_fb and K_inj so we can do hyperparameter optimization
}

# --------------------------------------------------
# Dataset
# --------------------------------------------------

base_dir = Path(__file__).resolve().parent

csv_file = base_dir / "santa_fe_time_series_a2.csv"

df = pd.read_csv(csv_file)
df = np.array(df).flatten()

df = (df - np.min(df))/(np.max(df) - np.min(df))

dataset_size_washout = 20
dataset_size_training = 4000
dataset_size_testing = 4000
jump_dataset = 1000

input_washout = df[:dataset_size_washout]

input_train = df[
    dataset_size_washout:
    dataset_size_washout + dataset_size_training
]

input_test = df[
    jump_dataset +
    dataset_size_washout +
    dataset_size_training:
    jump_dataset +
    dataset_size_washout +
    dataset_size_training +
    dataset_size_testing
]

p = 1

target_train = df[
    dataset_size_washout + p:
    dataset_size_washout + dataset_size_training + p
]

target_test = df[
    jump_dataset +
    dataset_size_washout +
    dataset_size_training + p:
    jump_dataset +
    dataset_size_washout +
    dataset_size_training +
    dataset_size_testing + p
]

#mask_array = 2*np.random.rand(reservoir_params["N_nodes"]) - 1 #Could be a hyerparameter to improve + better if it does not cross 0.
mask_array = 2*np.random.rand(reservoir_params["N_nodes"]) + 1 #Could be a hyerparameter to improve + better if it does not cross 0
X_states_test, Y, NMSE = drc.simulation(
    input_washout,
    input_train,
    input_test,
    target_train,
    target_test,
    mask_array,
    reservoir_params
)


print("NMSE =", NMSE)
print(X_states_test)

# Make sure arrays are flat
Y = np.asarray(Y).flatten()
target_test = np.asarray(target_test).flatten()


# Match lengths just in case
L = min(len(Y), len(target_test))

plt.figure(figsize=(12, 4))

plt.plot(np.arange(L), target_test[:L], label="Target", linewidth=2)
plt.plot(np.arange(L), Y[:L], label="Prediction", linewidth=2)

plt.xlabel("Time index")
plt.ylabel("Value")
plt.title(f"Santa Fe prediction, NMSE = {NMSE:.4f}")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(12, 4))

plt.plot(np.arange(L), X_states_test[:,0], label="1st node", linewidth=2)
plt.plot(np.arange(L), X_states_test[:,1], label="2nd node", linewidth=2)
plt.plot(np.arange(L), X_states_test[:,2], label="3rd node", linewidth=2)

plt.xlabel("Time index")
plt.ylabel("Value")
plt.title(f"Santa Fe prediction, NMSE = {NMSE:.4f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
