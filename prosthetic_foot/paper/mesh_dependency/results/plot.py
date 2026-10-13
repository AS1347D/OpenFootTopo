import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

plt.rcParams['font.size'] = 17

# 1. Identify files named with numbers (e.g., 80.csv, 240.csv)
all_files = glob.glob("*.csv")
# Filter to keep only the files that match your numeric naming convention
target_files = [f for f in all_files if re.match(r'^\d+\.csv$', f)]

results = []

for file in target_files:
    try:
        df = pd.read_csv(file)
        if not df.empty:
            # We take the last row (final iteration results)
            last_row = df.iloc[-1]
            results.append({
                'N_cell': last_row['N_cell'],
                'C_value': last_row['C_value'],
                'u_max': last_row['u_max'],
                'sigma_max': last_row['sigma_max']
            })
    except Exception as e:
        print(f"Error reading {file}: {e}")

# 2. Convert to DataFrame and sort by number of cells (X-axis)
results_df = pd.DataFrame(results).sort_values(by='N_cell')

if results_df.empty:
    print("No data found. Ensure CSV files are in the current directory.")
else:
    # 3. Plotting with subplots
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    TOL_PCT = 1.0  # +-1% margin w.r.t. the last (finest-mesh) value

    # Plot 1: Compliance
    axes[0].plot(results_df['N_cell'], results_df['C_value'], marker='o', color='royalblue', label='Compliance')
    last_C = results_df['C_value'].iloc[-1]
    axes[0].axhspan(last_C * (1 - TOL_PCT / 100), last_C * (1 + TOL_PCT / 100), color='royalblue', alpha=0.2)
    axes[0].set_ylabel('$C$ (N.m)')
    axes[0].set_ylim([min(results_df['C_value'])-0.05,27.75])

    # Plot 2: Max Displacement
    axes[1].plot(results_df['N_cell'], results_df['u_max'], marker='s', color='crimson', label='Max Displacement')
    last_u = results_df['u_max'].iloc[-1]
    axes[1].axhspan(last_u * (1 - TOL_PCT / 100), last_u * (1 + TOL_PCT / 100), color='crimson', alpha=0.15)
    axes[1].set_ylabel('$u_{max}$ (mm)')
    axes[1].set_ylim([min(results_df['u_max'])-0.001, 1.25])

    # Plot 3: Max Stress
    axes[2].plot(results_df['N_cell'], results_df['sigma_max'], marker='^', color='forestgreen', label='Max Stress')
    last_sigma = results_df['sigma_max'].iloc[-1]
    axes[2].axhspan(last_sigma * (1 - TOL_PCT / 100), last_sigma * (1 + TOL_PCT / 100), color='forestgreen', alpha=0.15)
    axes[2].set_ylabel(r'$\sigma_{max}$ (N/mm$^2$)')
    axes[2].set_xlabel('Number of cells - ($N_{cell}$)')
    axes[2].set_ylim([min(results_df['sigma_max'])-0.2, 14.0])

    import matplotlib.ticker as ticker
    # for ax in axes:
    #     ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    #     # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


    last_val = results_df['N_cell'].iloc[-1]
    ticks = [20000, 45000, 70000, 92000]

    # 2. Apply to your subplots (axes)
    for ax in axes:
        ax.set_xticks(ticks)
        # Formats as 20,000 instead of 20000
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    
    plt.savefig('mesh_dependency_shaded.png', dpi=450)
    plt.savefig('mesh_dependency_shaded.pdf')
    plt.show()

    metrics = ['C_value', 'u_max', 'sigma_max']
    
    print(f"{'Metric':<12} | {'Min Value':<12} | {'Max Value':<12} | {'Rel. Error (%)'}")
    print("-" * 60)
    
    for m in metrics:
        v_min = results_df[m].min()
        v_max = results_df[m].max()
        
        # Calculate relative error between min and max
        # This shows the total variation across all tested meshes
        rel_error = ((v_max - v_min) / v_max) * 100
        
        print(f"{m:<12} | {v_min:<12.4e} | {v_max:<12.4e} | {rel_error:>12.2f}%")