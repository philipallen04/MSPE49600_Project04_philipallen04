from pathlib import Path

import numpy as np

chord_m = 0.3 # 300 mm
thickness_ratio = 0.12  # NACA 0012
points_per_side = 201

# Cosine spacing puts more points near the leading and trailing edges.
theta = np.linspace(0, np.pi, points_per_side)
x = (1 - np.cos(theta)) / 2  # x/c

# Half-thickness y/c. The -0.1036 coefficient closes the trailing edge.
y = 5 * thickness_ratio * (
    0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
    + 0.2843 * x**3 - 0.1036 * x**4
)
y[[0, -1]] = 0.0  # Make the endpoints coincide exactly.

path = Path("NACA0012_300_DesignModeler.txt")

with path.open("w", encoding="ascii") as file:
    file.write("# Group Point X Y Z\n")

    # Upper surface: trailing edge to leading edge.
    for point, (xi, yi) in enumerate(zip(x[::-1], y[::-1]), start=1):
        file.write(f"1 {point} {chord_m * xi:.10f} {chord_m * yi:.10f} 0\n")

    # Lower surface: leading edge to trailing edge.
    for point, (xi, yi) in enumerate(zip(x, -y), start=1):
        file.write(f"2 {point} {chord_m * xi:.10f} {chord_m * yi:.10f} 0\n")

import matplotlib.pyplot as plt

x_m = chord_m * x
y_m = chord_m * y

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(x_m, y_m, label="Upper surface")
ax.plot(x_m, -y_m, label="Lower surface")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("NACA 0012 Airfoil")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()