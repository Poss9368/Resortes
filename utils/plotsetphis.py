import math
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from distribution_super_exp import SamplerSuperExp
from distribution_power_law import SamplerPowerLaw

def set_phis_difusivo(N: int, phi: float, seed: int):
    # generador de números aleatorios con distribución definida
    sampler = SamplerPowerLaw(alpha=2.5, tol=1e-9, seed=seed)
    if N%2 != 0: 
        N = N+1
        
    #Mitad de N como entero 
    half_N: float = int(N/2)
    total_segment_length = 0
    
    # Inicializar phis y l0
    phis: np.array = np.zeros(N)
    l0: np.array = np.zeros(N)
    
    # Inicializacion de orientación 
    orientation = 1

    i = 0
    while total_segment_length < half_N:
        if orientation == 1:
            orientation = -1
        else:
            orientation = 1
        
        rand = sampler.sample()
        i_temp = 0
        while i_temp < rand and total_segment_length < half_N:
            phis[i] = orientation
            l0[i] = 1.0
            total_segment_length += 1
            i += 1
            i_temp += 1
    
    ## reflejar phis y l0
    for i in range(half_N):
        phis[N-i-1] = -phis[i]
        l0[N-i-1] = l0[i]

    phis = phis*phi 
    return phis, l0


if __name__ == '__main__':
    N = 256
    phi =  np.pi/4
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    result = []
    for i in range(0, 20):
        seed = 123 + i
        phis, l0 = set_phis_difusivo(N, phi, seed)
        
        y = [0]
        x = [0]
        for i in range(0, N):
            y.append(y[i] + np.sin(phis[i]) * l0[i])
            x.append(x[i] + np.cos(phis[i]) * l0[i])
        
        ax1.plot(x, y)
        result.append(y[1:int(N/2)])
        

    plt.show()
    
        
        
        

        
    
    
    