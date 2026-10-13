import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

plt.rcParams['font.size'] = 20

# 1. Collect all files from the results folder
files = glob.glob("results/P[3-8]_*.csv")

data_list = []

for file in files:
    try:
        # Extract the Penalty value (P) from the filename (e.g., P3 from P3_15.csv)
        match = re.search(r'P(\d+)', file)
        p_val = int(match.group(1)) if match else None
        
        df = pd.read_csv(file)
        if not df.empty:
            df.columns = df.columns.str.strip()  # Clean whitespace from headers
            
            # Use the last converged iteration
            last_row = df.iloc[-1]
            
            data_list.append({
                'P': p_val,
                'vol_frac': last_row['target_vol_frac'],
                'compliance': last_row['C_value'],
                'u_max': last_row['u_max']
            })
    except Exception as e:
        print(f"Error processing {file}: {e}")

# 2. Convert to DataFrame and sort for clean line plotting
master_df = pd.DataFrame(data_list)
master_df['log_C'] = np.log10(master_df['compliance'])
master_df = master_df.sort_values(by=['P', 'vol_frac'])

# Get unique penalty values for the legend
p_values = sorted(master_df['P'].unique())

# --- Plot 1: Volume Fraction vs. Log(C) ---
plt.figure(figsize=(8, 6))
for p in p_values:
    subset = master_df[master_df['P'] == p]
    plt.plot(subset['vol_frac'], subset['log_C'], marker='o', label=f'P={p}')

plt.xlabel('Volume fraction - $V_{f}$')
plt.ylabel('Log compliance - $\log_{10}C$')
plt.legend()
plt.savefig('both1.png', dpi=450, bbox_inches='tight', pad_inches=0)
plt.show()

# --- Plot 2: Volume Fraction vs. u_max ---
plt.figure(figsize=(8, 6))
for p in p_values:
    subset = master_df[master_df['P'] == p]
    plt.plot(subset['vol_frac'], np.log10(subset['u_max']), marker='s', label=f'P={p}')

plt.xlabel('Volume fraction - $V_{f}$')
plt.ylabel('Log max displacement - $\log{u_{max}}$')
plt.legend()
plt.savefig('both2.png', dpi=450, bbox_inches='tight', pad_inches=0)

plt.show()