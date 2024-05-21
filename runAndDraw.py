import numpy as np
import random 
import matplotlib.pyplot as plt
from utils import *


if __name__ == "__main__":
    iteraciones = 60   # numero de iteraciones por simulación
    lambda_ML_vector = np.logspace(-3,1 , iteraciones)  # Vector de lambdas a probar, en escala logarítmica
    presicion = 1e-11 # Presición para la minimización
    
    N  = 100 # Número de eslabones y partículas
    l0 = 1 # Longitud de los eslabones
    k  = 1  # Constante del resorte
    phi  = np.pi/4 # angulo inicial
    
    seed = 1  # Semilla para reproducibilidad
    x0, y0, phis0, thetas0, L0, L_max= make_spring(N, l0, phi, seed) # Crear resorte
    phis = phis0.copy() # Copiar phis para guardar el estado inicial
    step_size = 0.1
    lambda_ML = 0
    
    fig1, ax1 = plt.subplots(figsize=(15, 2))
    fig2, ax2 = plt.subplots(figsize=(5 , 4))
    plot_spring(x0, y0, L0, ax1)
    plt.pause(0.05)
    input("Press Enter to continue...")
    
    lambda_ML_vector_cache = np.zeros(iteraciones)
    L_vector_cache = np.zeros(iteraciones)
    
    for itr in range(iteraciones):
        lambda_ML = lambda_ML_vector[itr]
        alpha_CG = 0 
        phis_punto_punto_CG = np.zeros(N)
        
        phis_punto_punto = -modified_hamiltonian_gradient(phis, phis0, lambda_ML, k, l0) 
        f_aux_1 = mean_force(phis_punto_punto)
        
        while f_aux_1 > presicion:
            phis_punto_punto_CG = phis_punto_punto + alpha_CG * phis_punto_punto_CG
            phis += step_size * phis_punto_punto_CG
            
            phis_punto_punto = -modified_hamiltonian_gradient(phis, phis0, lambda_ML, k, l0)
            f_aux_2 = mean_force(phis_punto_punto)
            
            if f_aux_2 < f_aux_1:
                alpha_CG = f_aux_2/f_aux_1
            else:
                alpha_CG = f_aux_1/f_aux_2
            
            f_aux_1 = f_aux_2    
            
        L = calculate_total_extention(phis, l0)
        x , y = set_positions_from_phis(phis, l0)
        lambda_ML_vector_cache[itr] = lambda_ML
        L_vector_cache[itr] = (L- L0)/L0
        
        print('Step:', itr+1, 'Largo actual:', L, 'Largo máximo:', L_max, 'lambda:', lambda_ML)
       
        plot_spring(x, y, L, ax1)
        plot_lambda_vs_L(L_vector_cache[:itr+1], lambda_ML_vector_cache[:itr+1], ax2)
        plt.pause(0.05)
        

    
        
        
        
    
            
            
   
    
       
           
       
       
        
        
       
    
    
    
    
