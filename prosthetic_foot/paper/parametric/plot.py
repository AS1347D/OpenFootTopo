import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 1. Define the folder and file pattern
results_folder = 'results'
p3_pattern = 'P3_*.csv'
p4_pattern = 'P4_*.csv'
p5_pattern = 'P5_*.csv'
p6_pattern = 'P6_*.csv'
p7_pattern = 'P7_*.csv'
p8_pattern = 'P8_*.csv'

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

                })
        except Exception as e:
            print(f"Could not read {file}: {e}")

    sorted_data = pd.DataFrame(all_data).sort_values(by='V_value')

    return sorted_data

p3_data = read_data(results_folder, p3_pattern)
p4_data = read_data(results_folder, p4_pattern)
p5_data = read_data(results_folder, p5_pattern)
p6_data = read_data(results_folder, p6_pattern)
p7_data = read_data(results_folder, p7_pattern)
p8_data = read_data(results_folder, p8_pattern)

F_p3 = 1610
F_p5 = 2240
F_p7 = 2900

thickness = 0.080 #m

# factor_p3 = p3_data['E']*thickness/F_p3**2
# factor_p5 = p5_data['E']*thickness/F_p5**2
# factor_p7 = p7_data['E']*thickness/F_p7**2

# factor_p3 = 1/F_p3**2
# factor_p5 = 1/F_p5**2
# factor_p7 = 1/F_p7**2

# compliance threshold

F_array = np.linspace(1610, 2900, 100)/thickness
sigma_eff = 0.001 # mm
C_threshold = sigma_eff*F_array
print(C_threshold) 
print(np.log10(C_threshold)) 

plt.figure(figsize=(6, 4))
plt.plot(p3_data['V_value'], np.log10(p3_data['C_value']*thickness), marker='o', linestyle='-', linewidth=2, label='P3')
plt.plot(p4_data['V_value'], np.log10(p4_data['C_value']*thickness), marker='o', linestyle='-', linewidth=2, label='P4')
plt.plot(p5_data['V_value'], np.log10(p5_data['C_value']*thickness), marker='o', linestyle='-', linewidth=2, label='P5')
plt.plot(p6_data['V_value'], np.log10(p6_data['C_value']*thickness), marker='o', linestyle='-', linewidth=2, label='P6')
plt.plot(p7_data['V_value'], np.log10(p7_data['C_value']*thickness), marker='o', linestyle='-', linewidth=2, label='P7')
plt.plot(p8_data['V_value'], np.log10(p8_data['C_value']*thickness), marker='o', linestyle='-', linewidth=2, label='P8')

# Formatting
plt.title('Volume fraction vs. Compliance', fontsize=14)
plt.xlabel(r'Volume fraction - $V_f$', fontsize=12)
# plt.xlabel(r'$\text{log}(V_f)$', fontsize=12)
# plt.ylabel(r'Compliance ($C$)', fontsize=12)
plt.ylabel(r'$\text{log}(C)$', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()

# Save the plot
plt.savefig('V_vs_C_plot.png')
plt.show()
print("Plot successfully saved as V_vs_C_plot.png")
