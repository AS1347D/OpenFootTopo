import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import numpy as np
import glob
import os

# --- Global Plot Settings ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 18
})

# --- Configuration ---
cases = {
    'E = 3GPa': 'parametric/results',
    'E = 30GPa': 'parametric2/results',
    'E = 300GPa': 'parametric3/results'
}

penalties = [3, 4, 5, 6, 7, 8]
forces = {3: 1610, 4: 2065, 5: 2240, 6: 2490, 7: 2900, 8: 3200}
linestyles = {'E = 3GPa': '-', 'E = 30GPa': '--', 'E = 300GPa': ':'}
thickness = 80 
foot_length = 260  

def read_data_by_p(folder, p_value):
    file_pattern = os.path.join(folder, f'P{p_value}_*.csv')
    files = glob.glob(file_pattern)
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                last_row = df.iloc[-1]
                all_data.append({
                    'V_value': last_row['V_value'],
                    'C_value': last_row['C_value'],
                    'E': last_row['E']
                })
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return pd.DataFrame(all_data).sort_values(by='V_value')



# --- Data Processing & Plotting ---
fig, ax1 = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(penalties)))

all_y_values = []
common_v = None 

# Plot individual simulation traces
for case_name, folder in cases.items():
    for idx, p in enumerate(penalties):
        data = read_data_by_p(folder, p)
        if not data.empty:
            # Calculate non-dimensional log-compliance
            factor = data['E'] / (forces[p]**2) * thickness * foot_length
            y_val = np.log10(data['C_value'] * factor) 
            
            all_y_values.append(y_val)
            if common_v is None:
                common_v = data['V_value'].values
            
            # ax1.plot(data['V_value'], y_val, 
            #          linestyle=linestyles[case_name], 
            #          color=colors[idx] if p != 8 else 'black',
            #          linewidth=1.2, alpha=0.3)


# --- Bi-Exponential Fitting ---

def bi_exponential(Vf, a, b, c, d, intercept):
    """Two-stage decay: (a,b) for steep start, (c,d) for shallow tail."""
    return a * np.exp(b * Vf) + c * np.exp(d * Vf) + intercept

if all_y_values:
    mean_y = np.mean(all_y_values, axis=0)
    try:
        # p0: [a, b, c, d, intercept]
        p0 = [10, -20, 2, -2, 2.4] 
        popt, _ = curve_fit(bi_exponential, common_v, mean_y, p0=p0)
        a, b, c, d, intercept = popt
        
        # Performance Metric
        y_pred = bi_exponential(common_v, *popt)
        r2 = 1 - (np.sum((mean_y - y_pred)**2) / np.sum((mean_y - np.mean(mean_y))**2))
        
        # Plotting the Fit
        v_smooth = np.linspace(common_v.min(), 1.0, 300)
        ax1.plot(v_smooth, bi_exponential(v_smooth, *popt), 
                 color='red', linestyle='--', linewidth=2, label='Bi-Exponential Fit', zorder=20)
        
        
        # Annotate Equation on Plot
        fit_info = (r"Fit: $y = ae^{bV_f} + ce^{dV_f} + y_0$" + "\n" +
                    f"$R^2 = {r2:.5f}$\n" +
                    f"$V_f \to 0: {a+c+intercept:.2f}$\n" +
                    f"$V_f \to \infty: {intercept:.4f}$")
        # ax1.text(0.05, 0.05, fit_info, transform=ax1.transAxes, fontsize=14,
        #          verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Terminal Log
        print(f"\nFit Results:\n{'-'*20}\nR^2: {r2:.5f}\nAsymptote: {intercept:.4f}\nIntercept: {a+c+intercept:.4f}")

        print("LOGARITHMIC FORM (As plotted):")
        print(f"  log10(gamma) = {a:.4f} * exp({b:.4f} * Vf) + {c:.4f} * exp({d:.4f} * Vf) + {intercept:.4f}")

    except Exception as e:
        print(f"Fit failed: {e}")

# Plot the statistical mean
# append the full domain simulation data
common_v = np.append(common_v, 1.0)
mean_y = np.append(mean_y, 2.3953)

ax1.plot(common_v, mean_y, color='blue', linewidth=4, label='Overall Mean', zorder=10)
# ax1.plot(v_smooth, bi_exponential(v_smooth, *popt), 
#                  color='blue', linewidth=4, label='Overall Mean', zorder=10)

# --- Custom Legend Construction ---
# Material entries
case_elements = [Line2D([0], [0], color='gray', lw=2, linestyle=linestyles[c], label=c) for c in cases.keys()]
# Penalty entries
penalty_elements = [Line2D([0], [0], color=colors[i] if penalties[i]!=8 else 'black', lw=2, label=f'P{penalties[i]}') 
                    for i in range(len(penalties))]
# Special lines
special_elements = [
    Line2D([0], [0], color='red', lw=4, label='Overall Mean'),
    Line2D([0], [0], color='cyan', lw=3, label='Bi-Exponential Fit')
]

# Combine all into one clean legend
# ax1.legend()

# --- Aesthetics ---
ax1.set_xlabel('Volume fraction - $V_f$')
ax1.set_ylabel(r'Log-Compliance - $\log_{10}(\gamma)$')
ax1.set_xlim([0,1.0])
ax1.set_ylim([2.1,4.8])

ax1.minorticks_on()
ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.tick_params(axis='both', which='major', direction='in', length=8)
ax1.tick_params(axis='x', which='minor', direction='in', length=4)
ax1.tick_params(axis='y', which='minor', direction='in', length=4)

plt.tight_layout()
plt.savefig('combined_expbifit.png', dpi=450)

plt.show()