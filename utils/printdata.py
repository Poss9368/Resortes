import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH_RESULTS = 'results/'

## read data 
def read_data(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)

def read_average_data(N: int, exponente: float) -> pd.DataFrame:
    return pd.read_csv(PATH_RESULTS + 'spring_evolution_' + str(N)+ '_' + "E" + "{:.3f}".format(exponente)  + '_average.csv')

def calculate_and_save_average(N: int, exponente: float, simulation_to_read: int) -> pd.DataFrame:
    sum_df = None
    for step in range(simulation_to_read):
        file_name = PATH_RESULTS + 'spring_evolution_' + str(N) + '_' + "E" + "{:.3f}".format(exponente) + '_' + str(step).zfill(5) + '.csv'
        data: pd.DataFrame = read_data(file_name)
        if sum_df is None:
            sum_df = data.copy()  # Inicializa sum_df con el primer DataFrame leído
        else:
            sum_df = sum_df.add(data)  # Suma los valores al DataFrame acumulativo
    
    # Calcula el promedio dividiendo por el número total de simulaciones
    avg_df = sum_df / simulation_to_read
    
    # Guarda el DataFrame promedio en un archivo CSV
    avg_df.to_csv(PATH_RESULTS + 'spring_evolution_' + str(N) + '_' + "E" + "{:.3f}".format(exponente) + '_average.csv', index=False)
    
    # Retorna el DataFrame promedio
    return avg_df

if __name__ == "__main__":
    
    fig1, ax1 = plt.subplots(figsize=(12 , 8))
    ax1.clear()

    exponentes = [2] 
    simulation_to_read = 16
    for i in range(1):
        N = 1024*(2**i)
        for exp in exponentes:    
            avg_df = calculate_and_save_average(N, exp,  simulation_to_read) # Uncomment this line to calculate the average data and save it to a file
            #avg_df = read_average_data(N, exponente ) # Uncomment this line to read the average data from a file
            
            delta_gamma = (avg_df['L'].values - avg_df['L_0'].values[0])/avg_df['L_0'].values[0]
            #delta_gamma = delta_gamma/delta_gamma[5]
            lambda_ML_vector = avg_df['lambda'].values
            ax1.plot(delta_gamma, lambda_ML_vector, marker='o', label='N = ' + str(N))
    

    fit_exp = 1
    ax1.plot(delta_gamma,0.000075*delta_gamma**fit_exp, '--')

    ax1.grid(True)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Lambda')
    ax1.set_xlabel('Delta Gamma')
    ax1.set_title('Lambda vs Delta Gamma') 
    plt.show()
    
        