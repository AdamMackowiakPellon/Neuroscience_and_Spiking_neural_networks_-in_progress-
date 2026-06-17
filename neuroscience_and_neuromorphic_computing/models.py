import numpy as np
import scipy as scy
from numba import njit
from sklearn.linear_model import Ridge

# ==============================================================================
# Delay reservoir computing reservoirs to test (maybe also include here another echo state networks)
# ==============================================================================

#Here J is already the injected current*mask. Maybe we should consider separate them.
class LIF():#Complex_valued_oscillator
    def __init__(self, V_rest, RC, I_inj, V_th):
        self.V_rest = V_rest
        self.RC = RC
        self.I_inj = I_inj
        self.V_th = V_th

    def dv_dt(self,K_X,E,N,J,header):
        K_X[0] = 0.5*(1 + 1j*self.alpha)*(self.gain(E[header-1],N) - self.tau_ph**(-1)) * E[header-1] + (self.k_fb / self.tau_in)*E[header]*np.exp(-2*np.pi*1j*self.tau) + (self.k_inj / self.tau_in)*self.E_inj_0*J*np.exp(-2*np.pi*1j*self.diff_freq*time)
        K_X[1] = (self.I_r / 1.6e-19 ) - ( N / self.tau_s ) - self.gain(E[header-1],N) * (abs(E[header-1]))**2
        return K_X
    

"""
It is interesting to mention the implementation of RK4, strictly speaking, is not correct. We are not doing interpolation of E(t+h/2-tau) nor we are updating
the value of time nor the the variables that depend on time, such as the the electric fields that depend on the delays, the mask nor the input for the computation
of k2,k3 and k4. Nevertheless, here we are not interested in the detailed dynamics of the reservoir but on the usage of the reservoir as a high dimensional nonlinear map.
"""
#@njit
def RK4(oscillator, time, header, h_half,h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4, oscillator_type):
    if oscillator_type == 1:
        K_X_1[:]=oscillator.dx_dt(K_X_1,X,N,J,time,header)
        K_X_2[:]=oscillator.dx_dt(K_X_2,X + h_half * K_X_1[0],N + h_half * K_X_1[1],J,time,header)
        K_X_3[:]=oscillator.dx_dt(K_X_3,X + h_half * K_X_2[0],N + h_half * K_X_2[1],J,time,header)
        K_X_4[:]=oscillator.dx_dt(K_X_4,X + h * K_X_3[0],N + h_half * K_X_3[1],J,time,header)
        return X[header-1] + (h/6)*(K_X_1[0] + 2*(K_X_2[0] + K_X_3[0]) + K_X_4[0]),N + (h/6)*(K_X_1[1] + 2*(K_X_2[1] + K_X_3[1]) + K_X_4[1])

def euler(oscillator, time, header, h_half,h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4, oscillator_type):
    if oscillator_type == 1:
        K_X_1[:] = oscillator.dx_dt(K_X_1,X,N,J,time,header)
        return X[header-1] + h*K_X_1[0] ,N + h*K_X_1[1]




# ==============================================================================
# Washout, train and test stage
# ==============================================================================

#@njit
def washout_stage(oscillator, oscillator_type, time, header, limit_header, h, h_half, X,N, K_X_1,K_X_2,K_X_3,K_X_4,
            input_washout, mask_array, N_nodes, steps_per_node):
    input_washout_length = len(input_washout)
    for i in range(input_washout_length):
        input = input_washout[i]
        for k in range(N_nodes):
            mask = mask_array[k]
            J = mask*input
            for _ in range(steps_per_node):
                #X[header], N = RK4(oscillator, time, header, h_half, h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4, oscillator_type)
                X[header], N = euler(oscillator, time, header, h_half, h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4,oscillator_type)
                header = header + 1
                if header == limit_header:
                    header = int(0)
                time = time + h
    return X,N, time, header
                    

#@njit
def train_stage(oscillator, oscillator_type, time, header, limit_header, h, h_half, X,N, K_X_1,K_X_2,K_X_3,K_X_4,
            input_train, X_states_train, mask_array, N_nodes, steps_per_node):
    input_train_length = len(input_train)
    for i in range(input_train_length):
        input = input_train[i]
        for k in range(N_nodes):
            mask = mask_array[k]
            J = mask*input
            for _ in range(steps_per_node):
                #X[header], N = RK4(oscillator, time, header, h_half,h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4, oscillator_type)
                X[header], N = euler(oscillator, time, header, h_half, h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4, oscillator_type)
                header = header + 1
                if header == limit_header:
                    header = int(0)
                time = time + h #I'm aware I'm also doing the same inside RK4 and I know that i could optimize it.
            X_states_train[i,k] = X[header-1]
    return X,N, time, header, X_states_train
    
                    

#@njit
def test_stage(oscillator, oscillator_type, time, header, limit_header, h, h_half, X,N, K_X_1,K_X_2,K_X_3,K_X_4,
            input_test, X_states_test, mask_array, N_nodes, steps_per_node):
    input_test_length = len(input_test)
    for i in range(input_test_length):
        input = input_test[i]
        for k in range(N_nodes):
            mask = mask_array[k]
            J = mask*input
            for _ in range(steps_per_node):
                #X[header], N = RK4(oscillator, time, header, h_half, h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4, oscillator_type)
                X[header], N = euler(oscillator, time, header, h_half, h, X, N, J, K_X_1, K_X_2, K_X_3, K_X_4, oscillator_type)
                header = header + 1
                if header == limit_header:
                    header = int(0)
                time = time + h
            X_states_test[i,k] = X[header-1]
    return X_states_test
    


        
    


    
#Maybe we should include the circular buffer inside the classes? Or maybe as a separate function outside of the classes?
#Maybe we should a program for delay reservoir computing and another one for echo state networks.




"""
Here we will create the X_states_train, X_states_test, X_states_train
"""
def simulation(theta, N_nodes,
                            input_washout, input_train, input_test,
                            target_train, target_test, mask_array, regularization, name, params):
    if name == "lang-kobayashi":
        oscillator = lk(**params)
        X_states_train = np.ones((len(input_train), N_nodes)) + 0j
        X_states_test = np.ones((len(input_test), N_nodes)) + 0j
        X = np.random.randn(int(oscillator.tau / oscillator.h + 0)) + 0j
        oscillator_type = 1
        num_var = 2
        K_X_1 = np.zeros(num_var, dtype= complex)
        K_X_2 = np.zeros(num_var, dtype= complex)
        K_X_3 = np.zeros(num_var, dtype= complex)
        K_X_4 = np.zeros(num_var, dtype= complex)
        N = 1.1*oscillator.N_0
    else:
        raise ValueError("Unknown oscillator")  
    
    steps_per_node = int(theta/oscillator.h)
    h = oscillator.h
    h_half = oscillator.h/2
    time = 0
    header = int(0)
    limit_header = len(X)

    #print(X)
    #print(N)
    X[:], N, time, header = washout_stage(oscillator=oscillator, oscillator_type=oscillator_type, time=time, header=header, limit_header=limit_header, 
                                          h=h, h_half=h_half, X=X, N=N, K_X_1=K_X_1,K_X_2=K_X_2,K_X_3=K_X_3,K_X_4=K_X_4, input_washout=input_washout, 
                                          mask_array=mask_array, N_nodes=N_nodes, steps_per_node=steps_per_node)
    #print(X)
    #print(N)
    X[:], N, time, header, X_states_train[:] = train_stage(oscillator=oscillator, oscillator_type=oscillator_type, time=time, header=header, limit_header=limit_header, 
                                              h=h, h_half=h_half, X=X, N=N, X_states_train=X_states_train, K_X_1=K_X_1,K_X_2=K_X_2,K_X_3=K_X_3,K_X_4=K_X_4, input_train=input_train, 
                                              mask_array=mask_array, N_nodes=N_nodes, steps_per_node=steps_per_node)
    #print(X)
    #print(N)
    #real_part = X_states_train.real
    #imag_part = X_states_train.imag
    #print(X_states_train)
    X_states_train = abs(X_states_train)**2
    #print(X_states_train)

    #X_states_train_real_imag = np.concatenate((real_part,imag_part),axis = 0)
    #target_train = np.concatenate((target_train,target_train))

    model = Ridge(alpha=regularization)
    model.fit(X_states_train, target_train)

    W = model.coef_
    b = model.intercept_

    X_states_test[:] = test_stage(oscillator=oscillator, oscillator_type=oscillator_type, time=time, header=header, limit_header=limit_header, 
                                      h=h, h_half=h_half, X=X, N=N, X_states_test=X_states_test, K_X_1=K_X_1,K_X_2=K_X_2,K_X_3=K_X_3,K_X_4=K_X_4,input_test=input_test, 
                                      mask_array=mask_array, N_nodes=N_nodes, steps_per_node=steps_per_node)
    #real_part = X_states_test.real
    #imag_part = X_states_test.imag
    X_states_test[:] = abs(X_states_test)**2
    #X_states_test_real_imag = np.concatenate((real_part,imag_part),axis = 0)
    #Idk if this is the correct dimensionality.
    Y = X_states_test @ W.T + b
    Y = np.asarray(Y).ravel()
    #target_test = np.asarray(target_test).ravel()
    
    #target_test = np.concatenate((target_test,target_test))
    var_y = np.var(target_test)

    #print(oscillator.eta)
    #print(oscillator.tau)
    #print(oscillator.gamma)
    #print(W.shape)
    #print(X_states_test.shape)
    #print(b.shape)
    #print("---")
    #print(var_y)
    #print(1/len(target_test))
#
#
    #print(Y.shape)
    #print(sum(Y-target_test))
    #print(target_test.shape)
    MSE = np.mean((np.abs(Y - target_test)**2))
    NMSE =  MSE / var_y
    NRMSE = np.sqrt(NMSE)


    print(f'MSE = {MSE}')
    print(f'NMSE = {NMSE}')
    print(f'NRMSE = {NRMSE}')

    
    return X_states_test, Y, NMSE
    







    
    

