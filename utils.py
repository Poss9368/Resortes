import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt

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

# calcula thetas desde phis
def set_thetas_from_phis(phis: np.array):
    N = len(phis)
    
    thetas = np.zeros(N)
    
    thetas[0] = phis[0]- phis[N-1] + np.pi
    
    for i in range(1, N):
        thetas[i] = phis[i] - phis[i-1] + np.pi
        
    return thetas
      
# Calcular posiciones desde phis y l0
def set_positions_from_phis(phis: np.array, l0: float):
    N = len(phis)
    
    x = np.zeros(N)
    y = np.zeros(N)
    
    x[0] = 0
    y[0] = 0
    
    for i in range(1, N):
        x[i] = x[i-1] + l0*np.cos(phis[i-1])
        y[i] = y[i-1] + l0*np.sin(phis[i-1])
        
    return x, y

# Crear resorte desde N, l0, phi y seed
def make_spring(N, l0, phi, seed: int ):
    if N%2 != 0: 
        N = N+1
        
    
    phis = set_phis(N, phi, seed)
    x, y = set_positions_from_phis(phis, l0)
    thetas = set_thetas_from_phis(phis)
    L_caja = N*l0*np.cos(phi)
    L_max  = N*l0
    
    return x, y, phis, thetas, L_caja, L_max
    
# Plotear resorte
def plot_spring(x, y, L_caja, ax):
    ax.clear()
    x = np.append(x, x[0] + L_caja)
    y = np.append(y, y[0])
    ax.plot(x, y, marker='o')
    ax.set_xlim(-0.005, L_caja + 0.005)
    ax.set_ylim(-3, 3)
    ax.grid(True)
    ax.set_aspect('equal', 'box') 
    ax.set_title('Spring Evolution')
    
def plot_lambda_vs_L(L_vector, lambda_ML_vector, ax):
    ax.clear()
    ax.plot(L_vector, lambda_ML_vector, marker='o')
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Lambda')
    ax.set_xlabel('(L-L0)/L0')
    ax.set_title('Lambda vs (L-L0)/L0')
    

# Guardar resorte
def save_spring(x: np.array, 
                 y: np.array, 
                 phis: np.array,
                 thetas: np.array, 
                 N: int,
                 serial: int,
                 step: int,
                 path: str):
                 
    df = pd.DataFrame({'x': x, 'y': y , 'phi': phis, 'theta': thetas})
    name = path +'/spring_position_' + str(N) + '_' + str(serial).zfill(5) + '_step_' + str(step).zfill(6) + '.csv'
    df.to_csv(name, index=False)
    
# Guardar evolución del resorte
def save_evolution(data: list,
                   N: int,
                   simulation_number: int,
                   path: str):
    
    df = pd.DataFrame(data)
    file_name =  path + '/spring_evolution_' + str(N) + '_' + str(simulation_number).zfill(5) + '.csv'
    df.to_csv(file_name, index=False)

# Función de potencial
def potential(phis: np.array, phis_0: np.array, k: float): 
    sum = 0
    N = len(phis) 
    
    sum  += 0.5 * k * ((phis[0]- phis_0[0]) - (phis[N-1]- phis_0[N-1]) )**2
    
    for i in range(1, N): 
        sum += 0.5 * k * ((phis[i]- phis_0[i]) - (phis[i-1]- phis_0[i-1]) )**2
        
    return sum
 
# Gradiente de la función de potencial    
def potential_gradient(phis: np.array, phis_0: np.array, k: float): 
    N = len(phis)
    phi_punto = np.zeros(N) 
    
    phi_punto[0] = k * ( 2 * phis[0] - phis[1] - phis[N-1] - (2 * phis_0[0] - phis_0[1] - phis_0[N-1]) )
    for i in range(1, N-1): 
        phi_punto[i] = k * ( 2 * phis[i] - phis[i+1] - phis[i-1] - (2 * phis_0[i] - phis_0[i+1] - phis_0[i-1]) )
        
    phi_punto[N-1] = k * ( 2 * phis[N-1] - phis[0] - phis[N-2] - (2 * phis_0[N-1] - phis_0[0] - phis_0[N-2]) )
    
    return phi_punto
    
# Función de restricción
def constraint_term(phis, L, l0):
    return L - np.sum(l0*np.cos(phis)) 

# Gradiente de la función de restricción
def constraint_term_gradient(phis,l0):
    return l0*np.sin(phis)

# Función del Hamiltoniano modificado
def modified_hamiltonian(phis, phis_0, L, l0, lambda_ML, k):
    return potential(phis, phis_0, k) + lambda_ML * constraint_term(phis, L, l0)

# Gradiente del Hamiltoniano modificado
def modified_hamiltonian_gradient(phis, phis_0, lambda_ML, k, l0):
    return potential_gradient(phis, phis_0, k) + lambda_ML * constraint_term_gradient(phis,l0)
    
# Función para calcular la fuerza media
def mean_force(phis_punto_punto):
    N = len(phis_punto_punto)
    return np.sqrt(np.mean(phis_punto_punto**2))
        
# Función para calcular la extensión total
def calculate_total_extention(phis, l0):
    return np.sum(l0*np.cos(phis))