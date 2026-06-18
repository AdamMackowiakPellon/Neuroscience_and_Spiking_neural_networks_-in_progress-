import numpy as np
import pandas as pd
from pathlib import Path

import delay_reservoir_computing as drc

# --------------------------------------------------
# Reservoir configuration
# --------------------------------------------------

reservoir_params = {
    "model": 2,          # 1=LIF,2=FHN,3=Izhikevich,4=HH
    "method": "euler",     # euler or rk4
    "theta": 0.2,
    "N_nodes": 500,
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

mask_array = 2*np.random.rand(reservoir_params["N_nodes"]) - 1 #Could be a hyerparameter to improve + better if it does not cross 0.

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

