Project that I am currently doing in my free time where I study different models of neurons in the context of dynamical systems and their performance in order to do computation. THe main idea is to look at the dynamics, implement the models in the context of delay reservoir computing, conventional reservoir computing and then looking at different architectures. Only time knows what will come after that :)

### Current state

In this repository there is a jupyter notebook where one can investigate the dynamics of the following four neuron models:
- LIF (Leaky Integrate-and-Fire) 1D + auxiliary function
- FHN (FitzHugh-Nagumo) 2D
- Izhikevich 2D + auxiliary function
- HH (Hodgkin-Huxley)

The parameters of the different models are taken from [Analysis of biologically plausible neuron models for regression with spiking neural networks](https://arxiv.org/abs/2401.00369), being in the excitable regime (subthreshold). As a side note, could be interesting to look at the different dynamical regimes and types of excitability and see which configurations works best for computation, for different architectures and tasks.

Moreover, I'm working on a code that merges Reservoir Computing (RC) and Spiking Neuronal Networks (SNNs). Some work has been done already in Time-Delay Spiking Reservoir Computing (TDSRC), although there are different approaches to work on this problem. The most similar one to the work that I am doing, at least as far as I know, is [Analog hardware implementation of spike-based delayed feedback reservoir computing system](https://ieeexplore.ieee.org/abstract/document/7966288). The current implementation of the code can work well in the context of RC (reaching NMSE=0.0610 with just 50 nodes for the Izhikevich model, although I've seen cases with better accuracy), but fails when implementing the spiking mechanism (I've managed to obtain as low as NMSE=0.7). I propose different directions:
- The most obvious one, changing parameters of the neuron and the mask, especially the injection and feedback strengths.
- Changing the delay from $k \cdot V(t-\tau)$ to $k \cdot [V(t-\tau) - V(t)]$, although I'm not so sure about this option.
- Encode the input as spikes. The problem is that the current task is the Santa Fe time step prediction, so how do you encode a time series with spikes?
- In the paper that I've cited before, the authors implement multiple LIFs in the input layer to encode the information and also they substitute the mask for a temporal encoder. Maybe the correct approach is doing something similar to that.
