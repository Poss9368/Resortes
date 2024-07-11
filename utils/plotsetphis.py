import math
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from utils import set_phis_difusivo 

if __name__ == '__main__':
    N = 200
    phi =  np.pi/4
    
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    
    result = []
    for i in range(0, 10000):
        seed = 123 + i
        phis, l0 = set_phis_difusivo(N, phi, seed, exponent=0.25)
        l0 = l0/np.sin(phi)
        
        y = [0]
        for i in range(0, N):
            y.append(y[i] + np.sin(phis[i]) * l0[i])
        
        ax1.plot(y)
        result.append(y[0:int(N/2)])
        
        
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    
    # desviacion estandar de todos los elementos de la primera fila
    ax2.plot(np.std(result, axis=0))
    ax2.grid(True)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()
    
        
        
        

        
    
    
    