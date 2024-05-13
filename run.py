import pandas as pd 
import numpy as np
import random 
import matplotlib.pyplot as plt


def set_phis(N, phi, seed: int): 
    # seed para reproducibilidad
    random.seed(seed)
    
    if N%2 != 0: 
        N = N+1
        
    # mitad de N como entero 
    half_N = int(N/2)
    
    phis: np.array = np.zeros(N)
    
    for i in range(0, half_N):
        numero_aleatorio = random.choice([-1, 1])
        phis[i] = phi*numero_aleatorio
        phis[N-i-1] = -phi*numero_aleatorio
        
    return phis

def set_thetas_from_phis(phis: np.array):
    N = len(phis)
    
    thetas = np.zeros(N)
    
    thetas[0] = phis[0]- phis[N-1] + np.pi
    
    for i in range(1, N):
        thetas[i] = phis[i] - phis[i-1] + np.pi
        
    return thetas
      
def make_resorte(N, l, phi, seed: int ):
    if N%2 != 0: 
        N = N+1
        
    # mitad de N como entero 
    half_N = int(N/2)
    
    x = np.zeros(N)
    y = np.zeros(N)

    x[0] = 0
    y[0] = 0
    
    phis = set_phis(N, phi, seed)
    
    for i in range(1, N):
        x[i] = x[i-1] + l*np.cos(phis[i-1])
        y[i] = y[i-1] + l*np.sin(phis[i-1])
        
    L_caja = N*l*np.cos(phi)
    
    thetas = set_thetas_from_phis(phis)
    
    
    return x, y, phis, thetas, L_caja

def plot_resorte(x: np.array, y: np.array, L_caja: float):
    
    x = np.append(x, x[0]+L_caja)
    y = np.append(y, y[0])
    
    plt.plot(x, y, marker='o')
    plt.xlim(0-0.01, L_caja+0.01)
    plt.grid(True)
    plt.show()
    
def save_resorte(x: np.array, 
                 y: np.array, 
                 phis: np.array,
                 thetas: np.array, 
                 L_caja: float,
                 N: int,
                 serial: int):
                 
    
    df = pd.DataFrame({'x': x, 'y': y , 'phi': phis, 'theta': thetas})
    name = 'resorte_' + str(N) + '_' + str(serial).zfill(5) + '.csv'
    df.to_csv(name, index=False)

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
def constraint_term(phis, L):
    return np.sum(np.cos(phis)) - L

# Gradiente de la función de restricción
def constraint_term_gradient(phis):
    return -np.sin(phis)

# Función del Hamiltoniano modificado
def modified_hamiltonian(phis, phis_0, L, lamda, k):
    return potential(phis, phis_0, k) + lamda * constraint_term(phis, L)

# Gradiente del Hamiltoniano modificado
def modified_hamiltonian_gradient(phis, phis_0, L, lamda, k):
    return potential_gradient(phis, phis_0, k) + lamda * constraint_term_gradient(phis)



if __name__ == "__main__":
    N = 30
    l = 1
    k = 1 
    
    phi = np.pi/4
    seed = 1 
    
    x, y, phis, thetas, L = make_resorte(N, l, phi, seed)
    phis_0 = phis.copy()
    
    step_size = 0.1
    lamda = +0.0001
    deformation = 0.1
    
    iteraciones = 1
    
    for i in range(iteraciones):
        L = L*(1+deformation)
        p = np.zeros(N)
        p_lambda = 0
        
        for i in range(100000):
            p += -0.5*step_size*modified_hamiltonian_gradient(phis, phis_0, L, lamda, k)
            phis += step_size*p
            p += -0.5*step_size*modified_hamiltonian_gradient(phis, phis_0, L, lamda, k)
            
            p_lambda += -0.5*step_size*constraint_term_gradient(phis)
            lamda += step_size*p_lambda
            p_lambda += -0.5*step_size*constraint_term_gradient(phis)
            
            
            
            print(constraint_term(phis, L))
        
            
        
        
        
        
       
    
    
    
    
    
    
#def hmc_with_constraint(num_samples, num_steps, step_size, L, lamda):
#    samples = []
#    phi = np.random.randn(num_steps)  # Inicializar las coordenadas aleatoriamente
#    
#    for _ in range(num_samples):
#        p = np.random.randn(num_steps)  # Inicializar los momentos aleatoriamente
#        
#        current_position = phi.copy()
#        current_momentum = p.copy()
#        
#        for _ in range(num_steps):
#            current_momentum -= 0.5 * step_size * modified_hamiltonian_gradient(current_position, c, L, lamda)
#            current_position += step_size * current_momentum
#            current_momentum -= 0.5 * step_size * modified_hamiltonian_gradient(current_position, c, L, lamda)
#        
#        current_momentum *= -1
#        
#        # Aceptar o rechazar la propuesta
#        proposed_hamiltonian = modified_hamiltonian(current_position, c, L, lamda)
#        current_hamiltonian = modified_hamiltonian(phi, c, L, lamda)
#        
#        if np.log(np.random.rand()) < current_hamiltonian - proposed_hamiltonian:
#            phi = current_position
#        samples.append(phi)
#    
#    return np.array(samples)
#
## Parámetros
#num_samples = 1000
#num_steps = 50
#step_size = 0.1
#L = 10
#lamda = 1
#c = np.random.randn(num_steps)  # Parámetros de los resortes
#
## Ejecutar HMC con restricción
#samples = hmc_with_constraint(num_samples, num_steps, step_size, L, lamda)