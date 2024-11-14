import math
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt

def generar_aleatorios_potencia(exponent, cantidad = 1):
    # Generar números aleatorios siguiendo una distribución de potencia
    r = np.random.uniform(0, 0.99, cantidad)
    numeros = pow(1-r, 1/(1-exponent))
    return numeros

def set_phis_difusivo(N: int, phi: float, seed: int, exponent: float = 2):
    # seed para reproducibilidad
    random.seed(seed)
    
    if N%2 != 0: 
        N = N+1
        
    #Mitad de N como entero 
    half_N: float = int(N/2)
    
    phis: np.array = np.zeros(N)
    l0: np.array = np.zeros(N)
    
    suma: float = 0
    numero_aleatorio = 1
    for i in range(0, half_N):
        #rnd = random.uniform(-1.0, 1.0)
        #
        #if rnd > 0:
        #    numero_aleatorio = 1.0
        #else:
        #    numero_aleatorio = -1.0
  
        if numero_aleatorio == 1:
            numero_aleatorio = -1
        else:
            numero_aleatorio = 1
        
        phis[i] = numero_aleatorio
        phis[N-i-1] = -numero_aleatorio
        
        l0[i] =generar_aleatorios_potencia(exponent)
        l0[N-i-1] = l0[i]
    
    phis = phis*phi 
    l0 = l0/np.sum(l0)*N
    
    return phis, l0


if __name__ == '__main__':
    N = 2000
    phi =  np.pi/4
    
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    result = []
    for i in range(0, 100):
        seed = 123 + i
        phis, l0 = set_phis_difusivo(N, phi, seed, exponent=1.1)
        
        y = [0]
        x = [0]
        for i in range(0, N):
            y.append(y[i] + np.sin(phis[i]) * l0[i])
            x.append(x[i] + np.cos(phis[i]) * l0[i])
        
        ax1.plot(x, y)
        result.append(y[1:int(N/2)])
        
        
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    
    # desviacion estandar de todos los elementos de la primera fila
    ax2.plot(np.std(result, axis=0), label='std', color='blue' , marker='o')
    ax2.grid(True)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()
    
        
        
        

        
    
    
    