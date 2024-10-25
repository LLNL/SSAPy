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
low-Earth into the cislunar regime. Orbits can be flexiabley specified from common 
input format such as Keplerian elements or two-line 
element data files. Addionally, SSAPy allows users to model satellite and specify parameters such 
as satilite area, mass, and drag coefficents. SSAPy includes a customizable force propagations
with a range of Earth, Lunar, radiation, atmospheric, and maneuvering models. SSAPy makes 
use of various community integration methods and can calculate
time-evolved orbital quantities including, satellite magnites and state vectors.
Users can specify various space- and ground- based observation models with support for
mutliple coordinate and references frames. SSAPy also supports orbit analysis and
propigation methods such as multiple-hypothesis tracking and has buit-in uncertainty qunatification
via short orbit arc probabilisitic orbit determinarion methods and Monte Carlo simulation runs. 
The majority of SSAPy's methods are vectorized and parallaelizable allowing effective use of 
high performance computer systems. Finally, SSAPy has plotting functionality allowing users to 
visualize orbits and trajectories which is shown in Figure 1. 

SSAPy has been used for the 
classification of cislunar [@Higgins2024] and closely spaced orbits [@Pruett2024] as 
well as studying the long-term stability of orbits in cislunar space [@Yeager2023]. SSAPy
has also be used to build a case study for rare events analysis in the context of satellites
passing close to each other in space [@Miller2022]. 

![Example SSAPy visualization plots.](image.png)

# Statement of need

Cislunar space is a region between earth out to beyond the the Moon's orbit that uncludes the
Lagranges points. This region of space is of growing importance to to space exploration edevours, 
scienfitic [e.g., @Duggan2019] or otherwise. Understanding, mapping, and modeling orbits through Cislunar space is 
critical to all of these endevours. The challenge for cislunar orbits is that n-body dynamics (e.g., graviational forces 
from the Sun, Earth, Moon and other planets) are significant, leading to unpredicatable and chaotic orbital motion. 
In this chaotic regime, orbits cannot be redcued to simple parameteric descriptions making scalable orbit 
simulation and modeling a citicial analysis tool [@Yeager2023]. Current obrit modeling software tools
are predominatly Graphical User Interfaced based [e.g., The General Mission Analysis Tool @Hughes2014 or the Systems Tool Kit]
and are not optimized for large scale simulation on high performance computer systems. 

There are many available software tools that are used for orbital analysis. The more prevalanetly
used tools are the General Mission Analysis Tool (GMAT) [@Hughes2014], REBOUND [@Rein2012], 
Systems Tool Kit (STK).

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