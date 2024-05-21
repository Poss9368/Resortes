import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from utils import *

if __name__ == "__main__":
    PATH: str = 'results/' # Carpeta donde se guardarán los resultados
    
    number_of_simulations = 2 #Número de simulaciones a realizar
    iteraciones = 60   # numero de iteraciones por simulación
    lambda_ML_vector = np.logspace(-3,1 , iteraciones)  # Vector de lambdas a probar, en escala logarítmica
    presicion = 1e-11 # Presición para la minimización
    
    N  = 100 # Número de eslabones y partículas
    l0 = 1 # Longitud de los eslabones
    k  = 1  # Constante del resorte
    phi  = np.pi/4 # angulo inicial

    for simulation in range(number_of_simulations):
        seed = simulation + 123
        x0, y0, phis0, thetas0, L0, L_max= make_spring(N, l0, phi, seed) # Crear resorte
        save_spring(x0, y0, phis0, thetas0, N, simulation, 0, PATH)
        
        phis = phis0.copy() # Copiar phis para guardar el estado inicial
        step_size = 0.1  # Tamaño del paso de integración para minimización
        lambda_ML = 0    # lambda del multiplicador de Lagrange, inicializado en 0
        
        data = []      # Lista para guardar la evolución de la simulación y guardarla en un archivo
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
            data.append({'L': L, 'L_max': L_max, 'lambda': lambda_ML})
            print('Step:', itr+1, 'Largo actual:', L, 'Largo máximo:', L_max, 'lambda:', lambda_ML)
        save_evolution(data, N, simulation, PATH)
        


        
        
        
        
        
    
            
            
   
    
       
           
       
       
        
        
       
    
    
    
    
