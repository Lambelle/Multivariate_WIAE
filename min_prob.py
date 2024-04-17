#Find the file with minimum MSE

import os

def find_min_mse(directory:str):
    min_mse = float('inf')
    min_mse_file = ''

    for file_name in os.listdir(directory):
        if 'UC90_' not in file_name:
            continue
        start_index = file_name.index('UC90_') + len('UC90_')
        end_index = file_name.index('UC50_')
        mse = float(file_name[start_index:end_index])
        if abs(mse-0.9) < min_mse:
            min_mse = abs(mse-0.9)
            min_mse_file = file_name
    return min_mse_file


mse_dir = "Revised_CTS_2D_5"
smallest_mse_file = find_min_mse(mse_dir)
print(f'The file with the smallest NMAE is: {smallest_mse_file}')