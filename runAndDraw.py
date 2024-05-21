import numpy as np
import random 
import matplotlib.pyplot as plt
from utils import *


if __name__ == "__main__":
    N  = 100 # Número de eslabones y partículas
    l0 = 1 # Longitud de los eslabones
    k  = 1  # Constante del resorte
    
    phi  = np.pi/4 # Ángulo inicial
    seed = 1  # Semilla para reproducibilidad

    x0, y0, phis0, thetas0, L0, L_max= make_spring(N, l0, phi, seed) # Crear resorte
    
    #plot_resorte(x0, y0, L0)
    
    phis = phis0.copy() # Copiar phis para guardar el estado inicial
    step_size = 0.05  
    lambda_ML = 0
    
    fig, ax = plt.subplots(figsize=(15, 2))

    plot_spring(x0, y0, L0, ax)
    plt.pause(0.05)
    
    input("Press Enter to continue...")
    
    iteraciones = 60
    lambda_ML_vector = np.logspace(-3,1 , iteraciones)    
    for i in range(iteraciones):
        lambda_ML = lambda_ML_vector[i]
        alpha_CG = 0 
        phis_punto_punto_CG = np.zeros(N)
        
        phis_punto_punto = -modified_hamiltonian_gradient(phis, phis0, lambda_ML, k, l0) 
        f_aux_1 = mean_force(phis_punto_punto)
        
        while f_aux_1 > 1e-11:
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
        print("Largo actual: {} - Largo máximo: {} - lambda: {}".format(L, L_max, lambda_ML))
        x , y = set_positions_from_phis(phis, l0)
        plot_spring(x, y, L, ax)
        plt.pause(0.001)
    
        
        
        
    
            
            
   
    
       
           
       
       
        
        
       
    
    
    
    
