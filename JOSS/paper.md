title: 'SSAPy - Space Situational Awareness for Python'
tags:
  - Python
  - space domain awareness
  - orbits
  - cislunar space
authors:
  - name: SSAPy team
    affiliation: 1

affiliations:
 - name: Lawrence Livermore National Laboratory, 7000 East Ave., Livermore, CA 94550, USA
   index: 1


date: 13 August 2017
bibliography: paper.bib

aas-doi:
aas-journal:
---

# Summary

SSAPy is a fast, flexible, high-fidelity orbital modeling and analysis tool for orbits spanning from 
low-Earth orbits (<2000 Km alitiude) orbits in the cislunar regime (between the earth and moon). 
Orbits can be flexiabley specified from common input format such as Keplerian elements or two-line 
element data files. Addionally, SSAPy allows users to model satellite and specify parameters such 
as satilite area, mass, and drag coefficents. SSAPy includes a customizable force propagations
with a range of Earth, Lunar, Radiation, Atmospheric, and Maneuvering models. SSAPy makes 
use of various integration methods (e.g., Runge-Kutta, Keplerian, and Taylor series)


Various community used integrators: SGP4, Runge-Kutta (4, 8, and 7/8), SciPy, Keplerian, Taylor Series

User definable timesteps with the ability to return various parameters for any orbit and at any desired timestep (e.g., magnitude, state vector, TLE, Keplerian elements, periapsis, apoapsis, specific angular momentum, and many more.)

Ground and space-based observer models

Location and time of various lighting conditions of interest

Multiple-hypothesis tracking (MHT) UCT linker

Vectorized computations (use of array broadcasting for fast computation, easily parallelizable and deployable on HPC machines)

Short arc probabilistic orbit determination methods

Conjunction probability estimation

Built-in uncertainty quantification

Support for Monte Carlo runs and data fusion

Support for multiple coordinate frames and coordinate frame conversions (GCRF, IERS, GCRS Cartesian, TEME Cartesian, ra/dec, NTW, zenith/azimuth, apparent positions, orthoginal tangent plane, and many more.)

Various plotting capabilities (ground tracks, 3D orbit plotting, cislunar trajectory visualization, etc.)

User definable timesteps and orbit information retrieval times, in which the user can query parameters of interest for that orbit and time.


# Statement of need


# feature summary

# Acknowledgements

`SSAPy` depends on numpy [@Harris2020], scipy [@Virtanen2020], matplotlib [@Hunter2007], emcee [@ForemanMackey2013], 
astropy [@astropy:2013,@astropy2018,@astropy2022], pyerfa [@], lmfit [@], and sqp4 [@].
This work was performed under the auspices of the U.S.
Department of Energy by Lawrence Livermore National
Laboratory (LLNL) under Contract DE-AC52-07NA27344.
The document number is LLNL-JRNL-XXXX and the code number 
is LLNL-CODE-862420. SSAPy was developed with support 
from Lawrence Livermore National Laboratory’s (LLNL) Laboratory
Directed Research and Development (LDRD) Program under 
projects 19-SI-004 and 22-ERD-054.

# References