import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 1. Define the folder and file pattern
results_folder = 'results'
p3_pattern = 'P3_*.csv'
p5_pattern = 'P5_*.csv'
p7_pattern = 'P7_*.csv'

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
p5_data = read_data(results_folder, p5_pattern)
p7_data = read_data(results_folder, p7_pattern)

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

F_array = np.linspace(1610, 2900, 100)#/thickness
sigma_eff = 0.005 # mm
C_threshold = sigma_eff*F_array
print(C_threshold) 
print(np.log10(C_threshold)) 

def get_thresh(force_val):
    return np.log10(sigma_eff * (force_val))

plt.figure(figsize=(9, 6))

# 1. Plot the Optimization Curves
plt.plot(p3_data['V_value'], np.log10(p3_data['C_value']*thickness), 'o-', color='blue', label='P3 Results')
plt.plot(p5_data['V_value'], np.log10(p5_data['C_value']*thickness), 'o-', color='red', label='P5 Results')
plt.plot(p7_data['V_value'], np.log10(p7_data['C_value']*thickness), 'o-', color='black', label='P7 Results')

# 2. Plot Horizontal Limit Lines (The Thresholds)
plt.axhline(get_thresh(1610), color='blue', linestyle='--', alpha=0.5, label='P3 Threshold')
plt.axhline(get_thresh(2240), color='red', linestyle='--', alpha=0.5, label='P5 Threshold')
plt.axhline(get_thresh(2900), color='black', linestyle='--', alpha=0.5, label='P7 Threshold')

# 3. Label the "Safe" zone
plt.fill_between([0.1, 0.8], np.array(np.log10(p3_data['C_value']*thickness))[-1], get_thresh(1610), color='green', alpha=0.05, label='Feasible Zone')

plt.title("Design Feasibility: Optimization vs. Stiffness Thresholds")
plt.xlabel("Volume Fraction ($V_f$)")
plt.ylabel(r"$\log_{10}(\text{Compliance Metric})$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('V_vs_C_plot3.png')
plt.show()
print("Plot successfully saved as V_vs_C_plot.png")
