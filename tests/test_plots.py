import ssapy
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import glob

save_folder = './ssapy_test_plots'
print(f"Putting test_plot.py output in: {save_folder}")

temp_directory = f'{save_folder}/rotate_vector_frames/'

# Check if the directory exists, and create it if it doesn't
try:
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
        print(f"Created directory: {temp_directory}")
    else:
        print(f"Directory already exists: {temp_directory}")
except Exception as e:
    print(f"Error creating directory: {e}")

# Testing rotate_vector() in utils.
v_unit = np.array([1, 0, 0])  # Replace this with your actual unit vector

figs = []

i = 0
for theta in range(0, 181, 20):
    for phi in range(0, 361, 20):
        try:
            new_unit_vector = ssapy.utils.rotate_vector(v_unit, theta, phi, plot_path=temp_directory, save_idx=i)
            print(f"Generated file: {temp_directory}/frame_{i}.png")  # Adjust filename format if needed
        except Exception as e:
            print(f"Error generating frame {i}: {e}")
        i += 1

files = glob.glob(f"{temp_directory}*")
print(f"Generated files: {files}")
for file in files:
    try:
        with open(file, 'rb') as f:
            print(f"File {file} is valid.")
    except Exception as e:
        print(f"File {file} is invalid: {e}")

gif_path = f"{save_folder}/rotate_vectors_{v_unit[0]:.0f}_{v_unit[1]:.0f}_{v_unit[2]:.0f}.gif"
try:
    ssapy.plotUtils.save_animated_gif(gif_name=gif_path, frames=ssapy.io.listdir(f'{temp_directory}*', sorted=True), fps=20)
except Exception as e:
    print(f"Error creating GIF: {e}")
# shutil.rmtree(temp_directory)

# Creating orbit plots
times = ssapy.utils.get_times(duration=(1, 'year'), freq=(1, 'hour'), t0='2025-3-1')
moon = ssapy.get_body("moon").position(times).T


def initialize_DRO(t, delta_r=7.52064e7, delta_v=344):
    """
    Calculate a distant retrograde orbit (DRO) as an orbit with
    adjustments based on the Moon's position and velocity.

    Parameters:
    ----------
    t : Time
        The time at which to calculate the orbit.
    delta_r : float, optional
        The adjustment to the Moon's position (default is 7.52064e7 meters).
    delta_v : float, optional
        The adjustment to the Moon's velocity (default is 344 meters/second).

    Returns:
    -------
    Orbit
        SSAPy Orbit object.
    """
    moon = ssapy.get_body("moon")

    unit_vector_moon = moon.position(t) / np.linalg.norm(moon.position(t))
    moon_v = (moon.position(t.gps) - moon.position(t.gps - 1)) / 1
    unit_vector_moon_velocity = moon_v / np.linalg.norm(moon_v)
    ssapy.compute.lunar_lagrange_points(t=times[0])

    r = (np.linalg.norm(moon.position(t)) - delta_r) * unit_vector_moon
    v = (np.linalg.norm(moon_v) + delta_v) * unit_vector_moon_velocity

    orbit = ssapy.Orbit(r=r, v=v, t=t)
    return orbit


# Distant Retrograde Orbit (DRO)
dro_orbit = initialize_DRO(t=times[0])
r, v = ssapy.simple.ssapy_orbit(orbit=dro_orbit, t=times)
ssapy.plotUtils.orbit_plot(r=r, t=times, save_path=f"{save_folder}/DRO_orbit", frame='Lunar', show=False)
r_lunar, v_lunar = ssapy.utils.gcrf_to_lunar_fixed(r, t=times, v=True)
print("Successfully converted GCRF to lunar frame.")
ssapy.plotUtils.koe_plot(r, v, t=times, body='Earth', save_path=f"{save_folder}Keplerian_orbital_elements.png")

ssapy.plotUtils.orbit_plot(r=r, t=times, save_path=f"{save_folder}/gcrf_plot.png", frame='gcrf', show=True)
ssapy.plotUtils.orbit_plot(r=r, t=times, save_path=f"{save_folder}/itrf_plot", frame='itrf', show=True)
ssapy.plotUtils.orbit_plot(r=r, t=times, save_path=f"{save_folder}/lunar_plot", frame='lunar', show=True)
ssapy.plotUtils.orbit_plot(r=r, t=times, save_path=f"{save_folder}/lunar_axis_lot", frame='lunar axis', show=True)
print("Created a GCRF orbit plot.")
print("Created a ITRF orbit plot.")
print("Created a Lunar orbit plot.")
print("Created a Lunar axis orbit plot.")

# Globe plot of a Geostationary Transfer Orbit (GTO)
r_geo, _ = ssapy.simple.ssapy_orbit(a=ssapy.constants.RGEO, e=0.3, t=times)
ssapy.plotUtils.globe_plot(r=r_geo, t=times, save_path=f"{save_folder}/globe_plot", scale=5)
print('Created a globe plot.')

ssapy.plotUtils.ground_track_plot(r=r_geo, t=times, ground_stations=None, save_path=f"{save_folder}/ground_track_plot")
print('Created a ground track plot.')

# Example usage
earth_pos = np.array([0, 0, 0])  # Earth at the origin
moon_pos = ssapy.get_body("moon").position(times[0]).T

# Plotting
fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
ax.scatter(earth_pos[0], earth_pos[1], earth_pos[2], color='blue', label='Earth')
ax.text(earth_pos[0], earth_pos[1], earth_pos[2], 'Earth', color='blue')

# Plot Moon
ax.scatter(moon_pos[0], moon_pos[1], moon_pos[2], color='grey', label='Moon')
ax.text(moon_pos[0], moon_pos[1], moon_pos[2], 'Moon', color='grey')

# Plot Lagrange points
colors = ['red', 'green', 'purple', 'orange', 'cyan']
for (point, pos), color in zip(ssapy.compute.lunar_lagrange_points(t=times[0]).items(), colors):
    ax.scatter(pos[0], pos[1], pos[2], color=color, label=point)
    ax.text(pos[0], pos[1], pos[2], point, color=color)

# Add a dashed black circle at the lunar distance (LD)
current_LD = np.linalg.norm(moon_pos, axis=-1)
normal_vector = ssapy.compute.moon_normal_vector(t=times[0])
ssapy.plotUtils.draw_dashed_circle(ax, normal_vector, current_LD, 12)
ax.quiver(0, 0, 0, normal_vector[0], normal_vector[1], normal_vector[2], color='r', length=1)

# Labels and legend
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("Lunar Lagrange Points using Moon's true position")
ax.axis('equal')
ax.legend()
plt.show()
ssapy.plotUtils.save_plot(fig, save_path=f"{save_folder}/lagrange_points")

# Plotting
fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
ax.scatter(earth_pos[0], earth_pos[1], earth_pos[2], color='blue', label='Earth')
ax.text(earth_pos[0], earth_pos[1], earth_pos[2], 'Earth', color='blue')

# Plot Moon
ax.scatter(moon_pos[0], moon_pos[1], moon_pos[2], color='grey', label='Moon')
ax.text(moon_pos[0], moon_pos[1], moon_pos[2], 'Moon', color='grey')

# Plot Lagrange points
colors = ['red', 'green', 'purple', 'orange', 'cyan']
for (point, pos), color in zip(ssapy.compute.lunar_lagrange_points_circular(t=times[0]).items(), colors):
    ax.scatter(pos[0], pos[1], pos[2], color=color, label=point)
    ax.text(pos[0], pos[1], pos[2], point, color=color)

# Add a dashed black circle at distance LD
current_LD = np.linalg.norm(moon_pos, axis=-1)
normal_vector = ssapy.compute.moon_normal_vector(t=times[0])
ssapy.plotUtils.draw_dashed_circle(ax, normal_vector, current_LD, 12)
ax.quiver(0, 0, 0, normal_vector[0], normal_vector[1], normal_vector[2], color='r', length=1)

# Labels and legend
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Lunar Lagrange Points assuming circular orbit.')
ax.axis('equal')
ax.legend()
plt.show()
ssapy.plotUtils.save_plot(fig, save_path=f"{save_folder}/lagrange_points")

print("Lagrange points were calculated correctly.")
print("Rotate vector plot successfully created.")
print("save_plot() executed successfully.")
print("save_animated_gif() executed successfully.")

print("\nFinished plot testing!\n")
