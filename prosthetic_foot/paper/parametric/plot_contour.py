import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf
import glob
import re

plt.rcParams['font.size'] = 20

# 1. Collect all files matching the pattern
files = glob.glob("results/P[3-8]_*.csv")

data_list = []

for file in files:
    try:
        df = pd.read_csv(file)
        if not df.empty:
            df.columns = df.columns.str.strip()
            
            if 'u_max' in df.columns:
                last_row = df.iloc[-1]
                data_list.append({
                    'vol_frac': last_row['target_vol_frac'],
                    'compliance': last_row['C_value'],
                    'u_max': last_row['u_max']
                })
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

# 2. Convert to DataFrame and calculate Log Compliance
master_df = pd.DataFrame(data_list)
master_df['log_compliance'] = np.log10(master_df['compliance'])

# 3. Create a grid for interpolation based on Log Compliance
xi = np.linspace(master_df['vol_frac'].min(), master_df['vol_frac'].max(), 100)
yi = np.linspace(master_df['log_compliance'].min(), master_df['log_compliance'].max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 4. Define the Boundaries (Envelopes)
# Group by vol_frac to find the min and max log_compliance at each x-point
envelope = master_df.groupby('vol_frac')['log_compliance'].agg(['min', 'max']).sort_index()

# Interpolate these envelopes onto our xi grid so we can check every grid point
upper_bound = np.interp(xi[0, :], envelope.index, envelope['max'])
lower_bound = np.interp(xi[0, :], envelope.index, envelope['min'])

# 5. Interpolate u_max (using linear or cubic)
zi = griddata(
    (master_df['vol_frac'], master_df['log_compliance']), 
    master_df['u_max'], 
    (xi, yi), 
    method='linear' 
)

# 6. MASKING: Set zi to NaN where yi is outside the envelope
for i in range(len(xi[0, :])):
    # For each column in the grid, find points outside [lower_bound, upper_bound]
    mask = (yi[:, i] > upper_bound[i]) | (yi[:, i] < lower_bound[i])
    zi[mask, i] = np.nan

# 5. Plotting (matplotlib naturally ignores NaNs in contourf)
# plt.figure(figsize=(18, 6))
plt.figure(figsize=(10, 6)) #single cluster
# plt.figure(figsize=(6, 6))

# Create the filled contour (only inside the envelope)
contour = plt.contourf(xi, yi, np.log10(zi), levels=100, cmap='jet')
cbar = plt.colorbar(contour)
cbar.set_label('Log max displacement - $\log_{10}(u_{max})$')

# lines = plt.contour(xi, yi, zi, levels=10, colors='white', alpha=0.0)
# plt.clabel(lines, inline=True)

# plt.scatter(master_df['vol_frac'], master_df['log_compliance'], 
#             color='red', marker='x', alpha=0.6, label='Simulation Data')


plt.xlabel(r'Volume fraction - $V_{f}$')
plt.ylabel(r'Log compliance - $\log_{10}(C)$')

from matplotlib.ticker import AutoMinorLocator
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

plt.tick_params(axis='both', which='major', direction='in', length=8)
plt.tick_params(axis='x', which='minor', direction='in', length=4)
plt.tick_params(axis='y', which='minor', direction='in', length=4)

plt.ylim([1,3.9])
plt.tight_layout()
plt.savefig('contour_masked_horizantal.png', dpi=450,  pad_inches=0)
# plt.savefig('contour_masked_vertical.png', dpi=450, bbox_inches='tight', pad_inches=0)
plt.show()