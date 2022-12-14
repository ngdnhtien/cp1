import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

x = np.arange(-3.0, 3.0, 0.01)
y = np.arange(-3.0, 3.0, 0.01)
X, Y = np.meshgrid(x, y)
Z1 = np.cos(X*Y) + np.sin(X**4 + Y**4) - 1
Z2 = X**2 + Y**2 + np.sin(X*Y) - 3
v = [0, 0]

fig, ax = plt.subplots()
CS1 = ax.contour(X, Y, Z1, 0, colors='red')
CS2 = ax.contour(X, Y, Z2, 0, colors='blue')
ax.clabel(CS1, inline=True, fontsize=10)
ax.clabel(CS2, inline=True, fontsize=10)
ax.set_title(r'Visualizing solutions in $x>-3, y<3$')
fig.savefig('a.png', dpi=300)