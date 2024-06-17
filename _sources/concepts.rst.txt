SSAPy Concepts
==============

THIS PAGE IS UNDER DEVELOPMENT

This page explains concepts and terminology used within the SSAPy package.
It is intended to provide a high-level overview, while details can be found in other sections of the documentation.

.. _def-coordinate-systems:

Coordinate Systems
------------------
A `coordinate system <https://en.wikipedia.org/wiki/Coordinate_system>`_ is a set of one or more values which define a unique position in space. The SSAPy package is capable of using a variety of coordinate systems:

..
    - Cartesian coordinates
    - ra/dec of stars from catalog positions (J2000/ICRS)
    - apparent positions
    - NTW
        - T gives the projection of (rcoord - r) along V  (tangent to track)
        - W gives the projection of (rcoord - r) along (V cross r) (normal to plane)
        - N gives the projection of (rcoord - r) along (V cross (V cross r))
            (in plane, perpendicular to T)
    - theta-phi-like coordinates: the system where theta is the angle between zenith and the point in question, and phi is the corresponding azimuthal angle. (radians)
    - lb-like coordinates & proper motions
        - if converting from theta-phi, sets b = pi - theta and renames phi -> l.  Everything is in radians.
    - orthographic tangent plane (radiants)
        - if converting from lb-like coordinates, the tangent plane is always chosen so that +Y is towards b = 90, and +X is towards +l.
    - GCRF
    - IERS
    - GCRS Cartesian Coordinate
    - TEME cartesian coordinate

    Not every function will be able to make use of every coordinate system, so please be sure to read the documentation associated with a given operation.

.. _def-time-standards:

Time Standards
--------------
..
    - UTC vs others

.. _def-orbits:

Types of Standard Orbits
------------------------
While SSAPy is not technically limited to modeling a specific orbit, there are certain types of orbits which are more closely related to SSAPy's capabilities. Those orbit types are listed below:

- Low Earth Orbit (LEO):
- Mid-earth orbit (MEO)
- Geosynchronous Earth Orbit (GEO):
- Geostationary Earth Orbit:
- Highly Elliptical Orbit (HEO)
- Cislunar Orbits:

Of course, many other types of near Earth orbits are possible (i.e. lunar rectilinear halo orbits). The sky, or the universe as it may be, is the limit!

.. _def-models:

Types of Models
---------------
..
    Some text about why we model these things

Gravitational models
^^^^^^^^^^^^^^^^^^^^
..
    - What means to have harmonics turned on
    - What it means to be a point source
    - Solar/Planetary point source modeling
    - Earth/Lunar modeling

Radiation pressure
^^^^^^^^^^^^^^^^^^
..
    text

Atmospheric Modeling
^^^^^^^^^^^^^^^^^^^^
..
    text

.. _def-numerical-integrators:

Numerical Integrators
---------------------
..
    - Why different
    - What step sizes mean

.. _def-computing-considerations:

Computing Considerations
------------------------
..
    - Many orbits
    - vectorization
    - orbit sampling and propagation for error estimation (rvsampling)
        - Linking observations together

.. _def-other-codes:

Other Codes
-----------
Below is a list of other orbit propagation codes, both commercial and free. While these other pieces of software may have some features in common with SSAPy, we believe SSAPy brings a more complete list of capabilities within one package.

- `General Mission Analysis Tool (GMAT) <https://software.nasa.gov/software/GSC-17177-1>`_
- `Ansys Systems Tool Kit (STK) <https://www.ansys.com/products/missions/ansys-stk>`_
- `a.i. solutions FreeFlyer Astrodynamics Software <https://ai-solutions.com/freeflyer-astrodynamic-software/>`_
- `MathWorks MATLAB <https://www.mathworks.com/products/matlab.html?s_tid=hp_products_matlab>`_
- `AstroPy <https://docs.astropy.org/en/stable/index.html>`_
- `REBOUND <https://rebound.readthedocs.io/en/latest/>`_
    - `REBOUNDx <https://reboundx.readthedocs.io/en/latest/index.html>`_
