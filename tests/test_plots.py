from ssapy.plotUtils import *
from ssapy.simple import ssapy_orbit
from ssapy.io import listdir, sortbynum
from ssapy.utils import rotate_vector

import os
import shutil
import numpy as np
from IPython.display import clear_output

# Example usage:
v_unit = np.array([1, 0, 0])  # Replace this with your actual unit vector

figs = []

save_directory = os.path.expanduser('~/ssapy_test_plots/rotate_vector_frames/')
os.makedirs(save_directory, exist_ok=True)

i = 0
for theta in range(0, 181, 10):
    for phi in range(0, 361, 10):
        clear_output(wait=True)
        new_unit_vector = rotate_vector(v_unit, theta, phi, plot=True, save_idx=i)
        i += 1

gif_path = f"{os.path.expanduser('~/ssapy_test_plots/')}rotate_vectors_{v_unit[0]:.0f}_{v_unit[1]:.0f}_{v_unit[2]:.0f}.gif"
write_gif(gif_name=gif_path, frames=sortbynum(listdir(f'{save_directory}*')), fps=20)
shutil.rmtree(save_directory)

print(f"Rotate vector plot successfully created.")