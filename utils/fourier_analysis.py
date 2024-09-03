import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt

from utils import set_positions_from_phis, set_phis


def analysis_fourier(N, phi, seed):
    phis, l0 = set_phis(N, phi, seed)
    x, y = set_positions_from_phis(phis, l0)
    #agregar un valor un 0 al final de y 
    #y = np.append(y, 0)
    print(y)
    transformada = np.fft.fft(y)
    real = transformada.real 
    imaginaria = transformada.imag
    #print(imaginaria)
    magnitud = np.abs(transformada)
    
    #real = real[1:half_N]
    #magnitud = magnitud[1:half_N]
    
    return real

N = 4
phi =  np.pi/4

results = []
for seed in range(10):
    result = analysis_fourier(N, phi, seed)
    results.append(result)

 
#promedio
results = np.array(results)
promedio = np.mean(results, axis=0)
print(promedio)

plt.plot(promedio)
plt.show()

