#Find the file with minimum MSE

import os

def find_min_mse(directory:str):
    min_mse = float('inf')
    min_mse_file = ''

    for file_name in os.listdir(directory):
        if 'MSE_' not in file_name:
            continue
        start_index = file_name.index('MSE_') + len('MSE_')
        end_index = file_name.index('MAE_')
        mse = float(file_name[start_index:end_index])
        if mse < min_mse:
            min_mse = mse
            min_mse_file = file_name
    return min_mse_file


mse_dir = "Revised_NYISO_RTDA_load_24"
smallest_mse_file = find_min_mse(mse_dir)
print(f'The file with the smallest NMAE is: {smallest_mse_file}')