---
title: 'SSAPy - Space Situational Awareness for Python'
tags:
  - Python
  - space domain awareness
  - orbits
  - cislunar space
authors:
  - name: Joshua E. Meyers
    affiliation: "1, 2"
    orcid: 0000-0002-2308-4230
  - name: Michael D. Schneider
    affiliation: 3
    orcid: 0000-0002-8505-7094
  - name: Julia T. Ebert
    affiliation: 4
    orcid: 0000-0002-1975-772X
  - name: Edward F. Schlafly
    affiliation: 5
    orcid: 0000-0002-3569-7421
  - name: Travis Yeager
    affiliation: 3
    orcid: 0000-0002-2582-0190
  - name: Alexx Perloff
    affiliation: 3
    orcid: 0000-0001-5230-0396
  - name: Daniel Merl
    affiliation: 3
    orcid: 0000-0003-4196-5354
  - name: Noah Lifset
    affiliation: 6
    orcid: 0000-0003-3397-7021
  - name: Jason Bernstein
    affiliation: 3
    orcid: 0000-0002-3391-5931
  - name: William A. Dawson
    affiliation: 3
    orcid: 0000-0003-0248-6123
  - name: Nathan Golovich
    affiliation: 3
    orcid: 0000-0003-2632-572X
  - name: Denvir Higgins
    affiliation: 3
    orcid: 0000-0002-7579-1092
  - name: Peter McGill
    affiliation: 3
    orcid: 0000-0002-1052-6749
    corresponding: true
  - name: Caleb Miller
    affiliation: 3
    orcid: 0000-0001-6249-0031
  - name: Kerianne Pruett
    affiliation: 3
    orcid: 0000-0002-2911-8657
affiliations:
 - name: SLAC National Accelerator Laboratory, 2575 Sand Hill Road, Menlo Park, CA 94025, USA
   index: 1
 - name: Kavli Institute for Particle Astrophysics and Cosmology, Stanford University, 452 Lomita Mall, Stanford, CA 94035, USA
   index: 2
 - name: Lawrence Livermore National Laboratory, 7000 East Ave., Livermore, CA 94550, USA
   index: 3
 - name: Fleet Robotics, 21 Properzi Way, Somerville, MA 02143, USA
   index: 4
 - name: Space Telescope Science Institute, 3700 San Martin Drive, Baltimore, MD 21218, USA
   index: 5
 - name: University of Texas at Austin, 2515 Speedway, Austin, TX 78712, USA
   index: 6



date: 13 January 2025
bibliography: paper.bib

aas-doi:
aas-journal:
---

# Summary

SSAPy is a fast and flexible orbit modeling and analysis tool for orbits spanning from
low-Earth into the cislunar regime. Orbits can be flexibly specified from common
input formats such as Keplerian elements or two-line
element data files. SSAPy allows users to model satellites and specify parameters such
as satellite area, mass, and drag coefficients. SSAPy includes a customizable force propagation
with a range of Earth, Lunar, radiation, atmospheric, and maneuvering models. SSAPy makes
use of various community integration methods and can calculate
time-evolved orbital quantities, including satellite magnitudes and state vectors.
Users can specify various space- and ground-based observation models with support for
multiple coordinate and reference frames. SSAPy also supports orbit analysis and
propagation methods such as multiple hypothesis tracking and has built-in uncertainty quantification.
The majority of SSAPy's methods are vectorized and parallelizable, allowing effective use of
high performance computer (HPC) systems. Finally, SSAPy has plotting functionality, allowing users to
visualize orbits and trajectories, an example of which is shown in Figures 1 and 2.

SSAPy has been used for the
classification of cislunar [@Higgins2024], and closely-spaced [@Pruett2024], orbits as
well as for studying the long-term stability of orbits in cislunar space [@Yeager2023]. SSAPy
has also been used to build a case study for rare events analysis in the context of satellites
passing close to each other in space [@Miller2022;@Bernstein2021].

# Statement of need

Cislunar space is a region between earth out to beyond the Moon's orbit that includes the
Lagrange points. This region of space is of growing importance to scientific and other space exploration endeavors [e.g., @Duggan2019].
Understanding, mapping, and modeling orbits through cislunar space is
critical to all of these endeavors. The challenge for cislunar orbits is that n-body dynamics (e.g., gravitational forces
from the Sun, Earth, Moon and other planets) are significant, leading to unpredictable and chaotic orbital motion.
In this chaotic regime, orbits cannot be reduced to simple parametric descriptions making scalable orbit
simulation and modeling a critical analysis tool [@Yeager2023]. Current orbit modeling software tools
are predominantly used via graphical user interfaces (e.g., The General Mission Analysis Tool; @Hughes2014 or the Systems Tool Kit)
and are not optimized for large scale simulation on HPC systems. Orbital modeling codes that
can be run on HPC systems (e.g., REBOUND; @Rein2012) lack full observable generation and modeling capabilities
with uncertainty quantification. Existing N-body propagators (e.g., Orekit; @OREKIT_2024) (e.g., Tudat; @TUDAT) rely on spherical harmonics or model the Moon as a point-mass, whereas SSAPy incorporates more comprehensive physical modeling relevant for cislunar dynamics. SSAPy, with its full-featured modeling framework and scalable, parallelizable
functionality, fills the gap in the orbital software landscape.


![Example SSAPy visualization plot of an orbit ground track over the surface of the Earth. The 48 hour orbit has a semi-major axis of 0.75 GEO (27,000km), an eccentricity 0.2 and an inclination of 45 degrees.](ground_track.png)

![Example SSAPy visualization plot. Simulated trajectory of the orbit shown in Figure 1. The color on this plot represents time.](orbit_plot.png){ width=50% }

# Acknowledgements

`SSAPy` depends on NumPy [@Harris2020], SciPy [@Virtanen2020], matplotlib [@Hunter2007], emcee [@ForemanMackey2013],
astropy [@astropy2022], pyerfa [@Kerkwijk2023], lmfit [@newville2024], and sqp4 [@Vallado2006].
We would like to thank Robert Armstrong and Iméne Goumiri for valuable contributions to this project.
 This work was performed under the auspices of the U.S.
Department of Energy by Lawrence Livermore National
Laboratory (LLNL) under Contract DE-AC52-07NA27344.
The document number is LLNL-JRNL-871602-DRAFT and the code number is LLNL-CODE-862420. SSAPy was developed with support
from LLNL's Laboratory Directed Research and Development Program under projects 19-SI-004 and 22-ERD-054.

# References
