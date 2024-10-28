---
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

SSAPy is a fast and flexible orbital modeling and analysis tool for orbits spanning from 
low-Earth into the cislunar regime. Orbits can be flexibly specified from common 
input format such as Keplerian elements or two-line 
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
visualize orbits and trajectories, an example of which is shown in Figure 1. 

SSAPy has been used for the 
classification of cislunar [@Higgins2024], and closely-spaced [@Pruett2024], orbits as 
well as for studying the long-term stability of orbits in cislunar space [@Yeager2023]. SSAPy
has also be used to build a case study for rare events analysis in the context of satellites
passing close to each other in space [@Miller2022]. 

![Example SSAPy visualization plots. The left plot shows an example relative angle between an orbit and the Earth's 
surface and is used for determining reentry points. The middle plot is the corresponding orbit in the 
International Terrestrial Reference Frame (ITRF), a frame fixed to the Earths surface. The third frame is the orbit in 
Geocentric Celestial Reference Frame (GCRF), a frame fixed to the stars. Color on the right two plots represents time.](image.png)

# Statement of need

Cislunar space is a region between earth out to beyond the Moon's orbit that includes the
Lagrange points. This region of space is of growing importance to space exploration endeavors, 
scientific [e.g., @Duggan2019] or otherwise. Understanding, mapping, and modeling orbits through cislunar space is 
critical to all of these endeavors. The challenge for cislunar orbits is that n-body dynamics (e.g., gravitational forces 
from the Sun, Earth, Moon and other planets) are significant, leading to unpredictable and chaotic orbital motion. 
In this chaotic regime, orbits cannot be reduced to simple parametric descriptions making scalable orbit 
simulation and modeling a critical analysis tool [@Yeager2023]. Current orbit modeling software tools
are predominantly used via graphical user interfaces (e.g., The General Mission Analysis Tool; @Hughes2014 or the Systems Tool Kit)
and are not optimized for large scale simulation on HPC systems. Orbital modeling codes that
can be run on HPC systems (e.g., REBOUND; @Rein2012) lack full observable generation and modeling capabilities
with uncertainty quantification. SAPPy, with its full-featured modeling framework and scalable, parallelizable
functionality, fills the gap in the orbital software landscape. 

# Acknowledgements

`SSAPy` depends on numpy [@Harris2020], scipy [@Virtanen2020], matplotlib [@Hunter2007], emcee [@ForemanMackey2013], 
astropy [@astropy2022], pyerfa [@Kerkwijk2023], lmfit [@newville2024], and sqp4 [@Vallado2006].
This work was performed under the auspices of the U.S.
Department of Energy by Lawrence Livermore National
Laboratory (LLNL) under Contract DE-AC52-07NA27344.
The document number is LLNL-JRNL-XXXX and the code number 
is LLNL-CODE-862420. SSAPy was developed with support 
from LLNL's Laboratory Directed Research and Development Program under 
projects 19-SI-004 and 22-ERD-054.

# References
