import numpy as np
from sklearn.linear_model import Ridge

from neuron_models_delay_reservoir import (
    get_neuron_model,
    RK4,
    Euler
)


# ============================================================
# Reservoir stages
# ============================================================

def propagate_dataset(neuron, X,X_buffer,dataset,mask_array,N_nodes,steps_per_node,
                      dt,dt_half,header,limit_header,collect_states=False):
    dim = len(X)

    K1 = np.zeros(dim)
    K2 = np.zeros(dim)
    K3 = np.zeros(dim)
    K4 = np.zeros(dim)

    X_next = np.zeros(dim)

    if collect_states:

        X_states = np.zeros(
            (len(dataset), N_nodes)
        )

    for i, sample in enumerate(dataset):

        for k in range(N_nodes):

            J = mask_array[k] * sample

            neuron.J = J
            spike = 0
            for _ in range(steps_per_node):
                
                X_tau = X_buffer[header]
                X[:] = RK4(neuron, dt, dt_half, X, X_tau, X_next, K1, K2, K3, K4)
                #X[:] = Euler(neuron, header, dt, dt_half, X, X_next, K1, K2, K3, K4)
                X_buffer[header] = X[0]
                header = header + 1
                if header == limit_header:
                    header = int(0)
                
                ##We count number of spikes for each vitual node. This condition is to avoid counting multiple time the same spike
                #if X_buffer[header-2]<neuron.V_th and X_buffer[header-1]>neuron.V_th: 
                #    spike+=1
                
                #We count number of spikes for each vitual node. This condition is to avoid counting multiple time the same spike
                if hasattr(neuron, "V_th"):
                    if X_buffer[header-2]<neuron.V_th and X_buffer[header-1]>neuron.V_th:
                        spike += 1
                        if hasattr(neuron, "threshold"):
                            X[:] = neuron.threshold(X)

            if collect_states:

                # membrane potential
                X_states[i, k] = spike
                #X_states[i, k] = X[0]

    if collect_states:

        return X, header, X_states

    return X, header


# ============================================================
# Main DRC simulation
# ============================================================

def simulation(
    input_washout,
    input_train,
    input_test,
    target_train,
    target_test,
    mask_array,
    reservoir_params
):

    model = reservoir_params["model"]
    method = reservoir_params["method"]

    theta = reservoir_params["theta"]
    N_nodes = reservoir_params["N_nodes"]


    regularization = reservoir_params["regularization"]

    neuron, X0, _, name = get_neuron_model(model)

    print()
    print("Reservoir model:", name)
    print("Integration:", method)

    dt = 0.01
    dt_half = dt*0.5
    header = int(0)

    delay = N_nodes * theta #Here we will use a setup where the time the input remains constant and the delay are the same
    delay_length = int(delay / dt)
    limit_header = delay_length
    X = X0.copy()
    X_buffer = np.full(delay_length, X[0])

    steps_per_node = max(1,int(theta / dt))

    # --------------------------------------------------------
    # Washout
    # --------------------------------------------------------

    print("Washout phase")

    X, header = propagate_dataset(
        neuron,
        X,
        X_buffer,
        input_washout,
        mask_array,
        N_nodes,
        steps_per_node,
        dt,
        dt_half,
        header,
        limit_header,
        collect_states=False,
    )

    # --------------------------------------------------------
    # Training states
    # --------------------------------------------------------

    print("Washout phase")

    X, header, X_states_train = propagate_dataset(
        neuron,
        X,
        X_buffer,
        input_train,
        mask_array,
        N_nodes,
        steps_per_node,
        dt,
        dt_half,
        header,
        limit_header,
        collect_states=True,
    )

    # --------------------------------------------------------
    # Testing states
    # --------------------------------------------------------

    print("Washout phase")

    X, header, X_states_test = propagate_dataset(
        neuron,
        X,
        X_buffer,
        input_test,
        mask_array,
        N_nodes,
        steps_per_node,
        dt,
        dt_half,
        header,
        limit_header,
        collect_states=True,
    )

    # --------------------------------------------------------
    # Ridge regression
    # --------------------------------------------------------

    ridge = Ridge(alpha=regularization, fit_intercept=True)

    ridge.fit(X_states_train, target_train)

    Y = ridge.predict(X_states_test)

    mse = np.mean((Y - target_test) ** 2)

    variance = np.var(target_test)

    NMSE = mse / variance

    print("NMSE =", NMSE)

    return (
        X_states_test,
        Y,
        NMSE
    )