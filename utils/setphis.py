import math
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt


def sing(x: float):
    if x >= 0:
        return 1
    else:
        return -1

# Funciones que determina los N phis iniciales desde una variable aleatoria
def set_phis(N, phi, seed: int): 
    # seed para reproducibilidad
    random.seed(seed)
    
    if N%2 != 0: 
        N = N+1
        
    #Mitad de N como entero 
    half_N: float = int(N/2)
    
    phis: np.array = np.zeros(N)
    
    for i in range(0, half_N):
        numero_aleatorio = random.choice([-1, 1])
        phis[i] = phi*numero_aleatorio
        phis[N-i-1] = -phi*numero_aleatorio
        
    return phis

def set_phis_difusivo(N, phi, seed: int, exponent: float = 0.5):
    # seed para reproducibilidad
    random.seed(seed)
    
    if N%2 != 0: 
        N = N+1
        
    #Mitad de N como entero 
    half_N: float = int(N/2)
    
    phis: np.array = np.zeros(N)
    l0: np.array = np.zeros(N)
    
    for i in range(0, half_N):
        numero_aleatorio = random.choice([-1, 1])
        phis[i] = phi*numero_aleatorio
        phis[N-i-1] = -phi*numero_aleatorio
        
        l0[i] = (i+1)**(exponent-0.5)
        l0[N-i-1] = l0[i]
        
    return phis, l0


if __name__ == '__main__':
    N = 200
    phi =  np.pi/4
    
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    
    result = []
    for i in range(0, 10000):
        seed = 123 + i
        phis, l0 = set_phis_difusivo(N, phi, seed, exponent=0.75)
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
    
        
        
        

        
    
    
    