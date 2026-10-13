import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# plt.rcParams['font.size'] = 20
plt.rcParams.update({
    "text.usetex": False,            # Set to True only if you have a full LaTeX install
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],    # This is the "Times" equivalent bundled with Matplotlib
    "mathtext.fontset": "stix",       # Ensures $V_f$ and $\log$ look like Times
    "font.size": 20
})

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
F_p4 = 2065
F_p5 = 2240
F_p6 = 2490
F_p7 = 2900
F_p8 = 3200

thickness = 80 # mm
foot_length = 260  

# factor_p3 = p3_data['E']*thickness**2/F_p3**2
# factor_p5 = p5_data['E']*thickness**2/F_p5**2
# factor_p7 = p7_data['E']*thickness**2/F_p7**2

factor_p3 = p3_data['E']/F_p3**2*thickness*foot_length
factor_p4 = p4_data['E']/F_p4**2*thickness*foot_length
factor_p5 = p5_data['E']/F_p5**2*thickness*foot_length
factor_p6 = p6_data['E']/F_p6**2*thickness*foot_length
factor_p7 = p7_data['E']/F_p7**2*thickness*foot_length
factor_p8 = p8_data['E']/F_p8**2*thickness*foot_length


fig, ax1 = plt.subplots(figsize=(8, 6))

# Plotting your P-data on the primary axis (ax1)
ax1.plot(p3_data['V_value'], np.log10(p3_data['C_value']*factor_p3), linestyle='-', linewidth=3, label='P3')
ax1.plot(p4_data['V_value'], np.log10(p4_data['C_value']*factor_p4), linestyle='-', linewidth=3, label='P4')
ax1.plot(p5_data['V_value'], np.log10(p5_data['C_value']*factor_p5), linestyle='-', linewidth=3, label='P5')
ax1.plot(p6_data['V_value'], np.log10(p6_data['C_value']*factor_p6), linestyle='-', linewidth=3, label='P6')
ax1.plot(p7_data['V_value'], np.log10(p7_data['C_value']*factor_p7), linestyle='-', linewidth=3, label='P7')
ax1.plot(p8_data['V_value'], np.log10(p8_data['C_value']*factor_p8), color='black', linestyle='-', linewidth=3, label='P8')

ax1.set_xlabel('Volume fraction - $V_f$')
ax1.set_ylabel(r'$\log_{10}(\gamma)$')
ax1.legend(loc='upper right')

# # --- Create the Second X-Axis for Force (ax2) ---
# ax2 = ax1.twiny() 
# # We plot the threshold on this second axis
# ax2.plot(F_array, np.log10(C_threshold), '--', color='green', label='Threshold')
# ax2.set_xlabel('Force ($F/t$)', color='green', fontsize=12)
# ax2.tick_params(axis='x', labelcolor='green')

# plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('nondim.png', dpi=450, bbox_inches='tight', pad_inches=0)
plt.show()
print("Plot successfully saved as nondim.png")