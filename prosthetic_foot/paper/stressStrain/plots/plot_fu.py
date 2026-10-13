import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 20
# delimiter="," tells numpy to split values by the comma
data = np.loadtxt("force_displacement.txt", delimiter=",")
data_neohookean = np.loadtxt("force_displacement_neohookean.txt", delimiter=",")
data_mooneyrivlin = np.loadtxt("force_displacement_mooneyrivlin.txt", delimiter=",")


# Create the plot
plt.figure(figsize=(7, 6))
plt.plot(data[:, 0], data[:, 1], linestyle='-', color='b', linewidth=4)
plt.plot(data_neohookean[:, 0], data_neohookean[:, 1], linestyle='--', color='r', linewidth=4)
# plt.plot(data_mooneyrivlin[:, 0], data_mooneyrivlin[:, 1], linestyle='--', color='r', linewidth=4)

# Add labels and title
plt.xlabel(r"Force (N) - $F$")
plt.ylabel(r"Max displacement (mm) - $u_{max}$")

from matplotlib.ticker import AutoMinorLocator
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

plt.tick_params(axis='both', which='major', direction='in', length=8)
plt.tick_params(axis='x', which='minor', direction='in', length=4)
plt.tick_params(axis='y', which='minor', direction='in', length=4)

# 5. Show or save the plot
plt.tight_layout()
plt.savefig("force_displacement_plot.png", dpi=450)

plt.show()