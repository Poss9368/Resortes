import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
import multiprocessing
from utils import *

PATH_RESULTS = 'results/'
N = 64*64 # Número de eslabones y partículas

def run_simulation(simulation):
        lambda_ML_min = -6 # Mínimo valor de lambda en --ESCALA LOGARÍTMICA--
        lambda_ML_max = 1  # Máximo valor de lambda en --ESCALA LOGARÍTMICA--
        iteraciones = (lambda_ML_max - lambda_ML_min)*5 + 1 # Número de iteraciones
        lambda_ML_vector = np.logspace(lambda_ML_min, lambda_ML_max, iteraciones) # Vector de lambdas
        
        presicion = 1e-10 # Presición para la minimización

        l0 = 1 # Longitud de los eslabones
        k  = 1  # Constante del resorte
        phi  = np.pi/4 # angulo inicial
        
        seed = simulation + 123
        x0, y0, phis0, thetas0, L0, L_max= make_spring(N, l0, phi, seed) # Crear resorte
        save_spring(x0, y0, phis0, thetas0, N, simulation, 0, PATH_RESULTS) # Guardar estado inicial
        
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
                    #alpha_CG = f_aux_1/f_aux_2
                    alpha_CG = 0.0
                    
                f_aux_1 = f_aux_2    
                
            L = calculate_total_extention(phis, l0) # Calcular la extensión total del resorte en el paso actual
            
            # Guardar información de la simulación en la lista para guardarla en un archivo
            data.append({'L': L, 
                         'L_max': L_max, 
                         'L_0': L0, 
                         'lambda': lambda_ML}) 
            
            # Imprimir en consola el paso actual --NO USAR EN PARALELO-- 
            #print('Step:', itr+1, 'Largo actual:', L, 'Largo máximo:', L_max, 'lambda:', lambda_ML) 
        
        save_evolution(data, N, simulation, PATH_RESULTS) # Guardar evolución de la simulación



RUN_IN_PARALLEL = True # Correr simulaciones en paralelo
NUM_CORES = 16 # Número de núcleos a utilizar

if __name__ == "__main__":        
    number_of_simulations = 16 #Número de simulaciones a realizar   
    
    if RUN_IN_PARALLEL:
        pool = multiprocessing.Pool(processes=NUM_CORES)
        pool.map(run_simulation, range(number_of_simulations))
        pool.close()
        pool.join()
    else:
        for simulation in range(number_of_simulations):
            run_simulation(simulation)

        
        
    
        
        
        


        
        
        
        
        
    
            
            
   
    
       
           
       
       
        
        
       
    
    
    
    
