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
 - name: Space Science Institute, Lawrence Livermore National Laboratory, 7000 East Ave., Livermore, CA 94550, USA
   index: 1


date: 13 August 2017
bibliography: paper.bib

aas-doi:
aas-journal:
---

# Summary

SSAPy is a fast, flexible, high-fidelity orbital modeling and analysis tool for orbits spanning from 
low-Earth orbits into the cislunar regime. Orbits can be flexiabley specified from common 
input format such as Keplerian elements or two-line 
element data files. Addionally, SSAPy allows users to model satellite and specify parameters such 
as satilite area, mass, and drag coefficents. SSAPy includes a customizable force propagations
with a range of Earth, Lunar, Radiation, Atmospheric, and Maneuvering models. SSAPy makes 
use of various integration methods (e.g., Runge-Kutta, Keplerian, and Taylor series) and can calculate
time-evolved orbital quantities including, satellite magnites, state vectors and specfic 
angular momentum. Users can specify various space- and ground- based observation models with support for
mutlitle coordinates and references frames such as the Geocentric Celestial Reference Frame and
the International Earth Rotation and Reference Systems Service. SSAPy also supports orbit analysis and
propigation methods such as Multiple-hypothesis tracking and has buit-in uncertainty qunatification
via short orbit arc probabilisitic orbit determinarion methods and Monte Carlo simulation runs. 
The majority of SSAPy's methods are vectorized and parallaelizable allowing effective use of 
high performance computer systems. Finally, SSAPy has plotting functionality allowing users to 
visualize orbits, trajectories, and their ground tracks. SSAPy has been used for the 
classification of cislunar [@Higgins2024] and closely spaced orbits [@Pruett2024] as 
well as studying the long-term stability of orbits in cislunar space [@Yeager2023]. SSAPy
has also be used to build a case study for rare events analysis in the context satellites
passing close to oneanother in space @Miller2022 

# Statement of need

There are many available software tools that are used for orbital analysis. The more prevalanetly
used tools are the General Mission Analysis Tool (GMAT) [@Hughes2014], REBOUND [@Rein2012], 
Systems Tool Kit (STK).

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