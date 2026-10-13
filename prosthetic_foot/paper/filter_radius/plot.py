import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 1. Define the folder and file pattern
results_folder = 'results'
r2_5_pattern = 'P5_2*.csv'
r5_0_pattern = 'P5_5*.csv'
r7_5_pattern = 'P5_7*.csv'
r10_pattern = 'P5_10*.csv'
r12_5_pattern = 'P5_12*.csv'

def read_data(results_folder, file_pattern):
    file_pattern = os.path.join(results_folder, file_pattern)

    files = glob.glob(file_pattern)

    all_data = []

    # 2. Iterate through files and extract the final state
    for file in files:
        try:
            # We now read WITH headers as per your new structure
            df = pd.read_csv(file)
            
            if not df.empty:
                # Get the last row (converged results)
                last_row = df.iloc[-1]
                
                all_data.append({
                    'V_value': last_row['V_value'],
                    'C_value': last_row['C_value'],
                    'E': last_row['E'],
                    'filter_radius':last_row['filter_radius']

                })
        except Exception as e:
            print(f"Could not read {file}: {e}")

    sorted_data = pd.DataFrame(all_data).sort_values(by='V_value')

    return sorted_data

r2_5_data = read_data(results_folder, r2_5_pattern)
r5_0_data = read_data(results_folder, r5_0_pattern)
r7_5_data = read_data(results_folder, r7_5_pattern)
r10_data = read_data(results_folder, r10_pattern)
r12_5_data = read_data(results_folder, r12_5_pattern)

thickness = 0.080 #m
print(r2_5_data['filter_radius'], r2_5_data['C_value']*thickness)
print(r5_0_data['filter_radius'], r5_0_data['C_value']*thickness)
print(r7_5_data['filter_radius'], r7_5_data['C_value']*thickness)
print(r10_data['filter_radius'], r10_data['C_value']*thickness)
print(r12_5_data['filter_radius'], r12_5_data['C_value']*thickness)