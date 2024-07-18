import math
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt

def sing(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def set_phis_difusivo_2(N: int, phi: float, seed: int, exponent: float = 0):
    
    # seed para reproducibilidad
    random.seed(seed)
    
    if N%2 != 0: 
        N = N+1
        
    #Mitad de N como entero 
    half_N: float = int(N/2)
    
    phis: np.array = np.zeros(N)
    l0: np.array = np.zeros(N)
    
    suma: float = 0
    
    for i in range(0, half_N):
    
        rnd = random.uniform(-1.0, 1.0)
        
        c = sing(suma) * abs(suma/(i+1))**exponent 
        
        if rnd > c:
            numero_aleatorio = +1
            suma += numero_aleatorio 
        else:
            numero_aleatorio = -1
            suma += numero_aleatorio
  
        
        phis[i] = numero_aleatorio
        phis[N-i-1] = -numero_aleatorio
        
        l0[i] = 1.0
        l0[N-i-1] = l0[i]
    
    
    #l0 = l0 * (N / np.sum(l0))
    phis = phis*phi 
    
    return phis, l0


if __name__ == '__main__':
    N = 200
    phi =  np.pi/4
    
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    result = []
    
    
    for i in range(0, 100):
        seed = 123 + i
        phis, l0 = set_phis_difusivo_2(N, phi, seed, exponent=0.5)
        l0 = l0/np.sin(phi)
        
        y = [0]
        for i in range(0, N):
            y.append(y[i] + np.sin(phis[i]) * l0[i])
        
        ax1.plot(y)
        result.append(y[1:int(N/2)])
        
        
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    
    # desviacion estandar de todos los elementos de la primera fila
    ax2.plot(np.std(result, axis=0), label='std', color='blue' , marker='o')
    ax2.grid(True)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()
    
        
        
        

        
    
    
    