from ssapy import AccelKepler
from ssapy.constants import RGEO, VGEO, EARTH_RADIUS
from ssapy.compute import rv
from ssapy.orbit import Orbit
from ssapy.propagator import RK4Propagator, RK8Propagator, RK78Propagator, SciPyPropagator, KeplerianPropagator
import numpy as np
import matplotlib.pyplot as plt

# Set up the suborbital test orbit (collides with Earth)
r0 = np.array([RGEO, 0, 0])
v0 = np.array([0, VGEO / 2, 0])  # too slow to stay in orbit
t0 = 0
orbit = Orbit(r0, v0, t0)

# Time span (long enough to ensure impact)
tvals_full = np.arange(t0, 24000)

# List of propagators to test
propagators = [
    RK4Propagator(AccelKepler(), h=10),
    RK8Propagator(AccelKepler(), h=10),
    RK78Propagator(AccelKepler(), h=10),
    SciPyPropagator(AccelKepler()),
    # KeplerianPropagator()  # Keplerian won't handle impact correctly; optional to include
]

# Run test for each propagator
for prop in propagators:
    print(f"\nTesting: {prop}")
    try:
        r, v = rv(orbit, tvals_full, propagator=prop)
        altitudes = np.linalg.norm(r, axis=1) - EARTH_RADIUS
        tvals_trunc = tvals_full[:len(r)]

        print(f"Returned {len(r)} valid states before impact.")
        print(f"Final altitude: {altitudes[-1]:.2f} meters")

        # Plot Altitude vs Time
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(tvals_trunc, altitudes / 1e3, label='Altitude (km)', color='tab:blue')
        plt.axhline(0, linestyle='--', color='red', label='Earth Surface (0 km)')
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude (km)")
        plt.title(f"Altitude vs Time")
        plt.grid(True)
        plt.legend()

        # Plot Orbit in XY-plane
        plt.subplot(1, 2, 2)
        plt.plot(r[:, 0] / 1e3, r[:, 1] / 1e3, label='Trajectory (XY)', color='tab:green')
        earth = plt.Circle((0, 0), EARTH_RADIUS / 1e3, color='lightblue', alpha=0.5, label='Earth')
        plt.gca().add_patch(earth)
        plt.axis('equal')
        plt.xlabel("X Position (km)")
        plt.ylabel("Y Position (km)")
        plt.title("XY Orbit Projection")
        plt.grid(True)
        plt.legend()

        plt.suptitle(f"Propagator: {prop}")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error with {prop}: {e}")
