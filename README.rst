SSAPy - Space Situational Awareness for Python
==============================================

|ci_badge|  |docs_badge| |codecov_badge| |joss_badge| |pypi-badge|

.. |ci_badge| image:: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml

.. |docs_badge| image:: https://github.com/LLNL/SSAPy/actions/workflows/pages/pages-build-deployment/badge.svg
    :target: https://LLNL.github.io/SSAPy

.. |codecov_badge| image:: https://codecov.io/gh/LLNL/SSAPy/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/LLNL/SSAPy

.. |joss_badge| image:: https://joss.theoj.org/papers/a629353cbdd8d64a861bb807e12c5d06/status.svg
   :target: https://joss.theoj.org/papers/a629353cbdd8d64a861bb807e12c5d06

.. |pypi-badge| image:: https://badge.fury.io/py/llnl-ssapy.svg
    :target: https://badge.fury.io/py/llnl-ssapy

`SSAPy <https://github.com/LLNL/SSAPy>`_ is a fast, flexible, high-fidelity orbital modeling and analysis tool for orbits spanning from low-Earth orbit into the cislunar regime, and includes the following:

- Ability to define satellite parameters (area, mass, radiation and drag coefficients, etc.)
- Support for multiple data types (e.g., read in orbit from TLE file, define a set of Keplerian, Equinoctial, or Kozai Mean Keplerian elements, etc.)
- Define a fully customizable analytic force propagation model including the following:
    - Earth gravity models (WGS84, EGM84, EGM96, EGM2008)
    - Lunar gravity model (point source & harmonic)
    - Radiation pressure (Earth & solar)
    - Forces for all planets out to Neptune
    - Atmospheric drag models
    - Maneuvering (takes a user defined burn profile)
- Various community used integrators: SGP4, Runge-Kutta (4, 8, and 7/8), SciPy, Keplerian, Taylor Series
- User definable timesteps with the ability to return various parameters for any orbit and at any desired timestep (e.g., magnitude, state vector, TLE, Keplerian elements, periapsis, apoapsis, specific angular momentum, and many more.)
- Ground and space-based observer models
- Location and time of various lighting conditions of interest
- Multiple-hypothesis tracking (MHT) UCT linker
- Vectorized computations (use of array broadcasting for fast computation, easily parallelizable and deployable on HPC machines)
- Short arc probabilistic orbit determination methods
- Conjunction probability estimation
- Built-in uncertainty quantification
- Support for Monte Carlo runs and data fusion
- Support for multiple coordinate frames and coordinate frame conversions (GCRF, IERS, GCRS Cartesian, TEME Cartesian, ra/dec, NTW, zenith/azimuth, apparent positions, orthoginal tangent plane, and many more.)
- Various plotting capabilities (ground tracks, 3D orbit plotting, cislunar trajectory visualization, etc.)
- User definable timesteps and orbit information retrieval times, in which the user can query parameters of interest for that orbit and time.

Installation
------------

For installation details, see the `Installing SSAPy <https://LLNL.github.io/SSAPy/installation.html>`_ section of the documentation.

Strict dependencies
-------------------

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`_ (3.8+)

The following are installed automatically when you install SSAPy:

- `numpy <https://scipy.org/install.html>`_;
- `scipy <https://scipy.org/scipylib/index.html>`_ for many statistical functions;
- `astropy <https://www.astropy.org/>`_ for astronomy related functions;
- `pyerfa <https://pypi.org/project/pyerfa/>`_ a Python wrapper for the ERFA library;
- `emcee <https://pypi.org/project/emcee/>`_ an affine-invariant ensemble sampler for Markov chain Monte Carlo;
- `lmfit <https://pypi.org/project/lmfit/>`_ a package for non-linear least-squares minimization and curve fitting;
- `sgp4 <https://pypi.org/project/sgp4/>`_ contains functions to compute the positions of satellites in Earth orbit;
- `matplotlib <https://matplotlib.org/>`_ as a plotting backend;
- and other utility packages, as enumerated in `setup.py`.

Documentation
-------------

All documentation is hosted at `https://LLNL.github.io/SSAPy/ <https://LLNL.github.io/SSAPy/>`_.

The API documentation may also be seen by doing:

.. code-block:: bash

    python3
    >>> import ssapy
    >>> help(ssapy)

Contributing
------------

Contributing to SSAPy is relatively easy. Just send us a `pull request <https://help.github.com/articles/using-pull-requests/>`_. When you send your request, make `main` the destination branch on the `SSAPy repository <https://github.com/LLNL/SSAPy>`_.

Your PR must pass SSAPy's unit tests and documentation tests, and must be `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ compliant. We enforce these guidelines with our CI process. To run these tests locally, and for helpful tips on git, see our `Contribution Guide <https://ssapy.reathedocs.io/en/latest/contribution_guide.html>`_.

SSAPy's `main` branch has the latest contributions. Pull requests should target `main`, and users who want the latest package versions, features, etc. can use `main`.

Releases
--------

For multi-user site deployments or other use cases that need very stable software installations, we recommend using SSAPy's `stable releases <https://github.com/LLNL/SSAPy/releases>`_.

Each SSAPy release series also has a corresponding branch, e.g. `releases/v0.14` has `0.14.x` versions of SSAPy, and `releases/v0.13` has `0.13.x` versions. We backport important bug fixes to these branches but we do not advance the package versions or make other changes that would change the way SSAPy concretizes dependencies within a release branch. So, you can base your SSAPy deployment on a release branch and `git pull` to get fixes, without the package churn that comes with `main`.

The latest release is always available with the `releases/latest` tag.

See the `docs on releases <https://ssapy.reathedocs.io/en/latest/contribution_guide.html#releases>`_ for more details.

Code of Conduct
---------------

Please note that SSAPy has a `Code of Conduct <https://github.com/LLNL/SSAPy/blob/main/CODE_OF_CONDUCT.md>`_. By participating in the SSAPy community, you agree to abide by its rules.

Authors
-------

SSAPy was developed with support from Lawrence Livermore National Laboratory's (LLNL) Laboratory Directed Research and Development (LDRD) Program under projects
`19-SI-004 <https://ldrd-annual.llnl.gov/archives/ldrd-annual-2021/project-highlights/high-performance-computing-simulation-and-data-science/madstare-modeling-and-analysis-data-starved-or-ambiguous-environments>`_ and
`22-ERD-054 <https://ldrd-annual.llnl.gov/ldrd-annual-2023/project-highlights/space-security/data-demand-capable-space-domain-awareness-architecture>`_, by the following individuals (in alphabetical order):

- `Robert Armstrong <https://orcid.org/0000-0002-6911-1038>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Nathan Golovich <https://orcid.org/0000-0003-2632-572X>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Julia Ebert <https://orcid.org/0000-0002-1975-772X>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now at Fleet Robotics)
- `Noah Lifset <https://orcid.org/0000-0003-3397-7021>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now PhD student at `UT Austin <https://www.utexas.edu>`_)
- `Dan Merl <https://orcid.org/0000-0003-4196-5354>`_ (`LLNL <https://www.llnl.gov/>`_) - Developer
- `Joshua Meyers <https://orcid.org/0000-0002-2308-4230>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now at `KIPAC <https://kipac.stanford.edu/>`_) - Former Lead Developer
- `Caleb Miller <https://orcid.org/0000-0001-6249-0031>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Alexx Perloff <https://orcid.org/0000-0001-5230-0396>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Kerianne Pruett <https://orcid.org/0000-0002-2911-8657>`_ (formerly `LLNL <https://www.llnl.gov/>`_)
- `Edward Schlafly <https://orcid.org/0000-0002-3569-7421>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now `STScI <https://www.stsci.edu/>`_) - Former Lead Developer
- `Michael Schneider <https://orcid.org/0000-0002-8505-7094>`_ (`LLNL <https://www.llnl.gov/>`_) - Creator, Former Lead Developer
- `Travis Yeager <https://orcid.org/0000-0002-2582-0190>`_ (`LLNL <https://www.llnl.gov/>`_) - Current Lead Developer

Many thanks go to SSAPy's other `contributors <https://github.com/llnl/ssapy/graphs/contributors>`_.


Citing SSAPy
^^^^^^^^^^^^

On GitHub, you can copy this citation in APA or BibTeX format via the "Cite this repository" button.
If you prefer MLA or Chicago style citations, see the comments in `CITATION.cff <https://github.com/LLNL/SSAPy/blob/main/CITATION.cff>`_.

You may also cite the following publications (click `here <https://github.com/LLNL/SSAPy/blob/main/docs/source/citations.bib>`_ for list of BibTeX citations):

 - Yeager, T., Pruett, K., & Schneider, M. (2022). *Unaided Dynamical Orbit Stability in the Cislunar Regime.* [Poster presentation]. Cislunar Security Conference, USA.
 - Yeager, T., Pruett, K., & Schneider, M. (2023). *Long-term N-body Stability in Cislunar Space.* [Poster presentation]. Advanced Maui Optical and Space Surveillance (AMOS) Technologies Conference, USA.
 - Yeager, T., Pruett, K., & Schneider, M. (2023, September). Long-term N-body Stability in Cislunar Space. In S. Ryan (Ed.), *Proceedings of the Advanced Maui Optical and Space Surveillance (AMOS) Technologies Conference* (p. 208). Retrieved from `https://amostech.com/TechnicalPapers/2023/Poster/Yeager.pdf <https://amostech.com/TechnicalPapers/2023/Poster/Yeager.pdf>`_

License
-------

SSAPy is distributed under the terms of the MIT license. All new contributions must be made under the MIT license.

See `Link to license <https://github.com/LLNL/SSAPy/blob/main/LICENSE>`_ and `NOTICE <https://github.com/LLNL/SSAPy/blob/main/NOTICE>`_ for details.

SPDX-License-Identifier: MIT

LLNL-CODE-862420

Documentation Inspiration
-------------------------
The structure and organization of this repository's documentation were inspired by the excellent design and layout of the `Coffea <https://coffea-hep.readthedocs.io/en/latest/index.html>`_ project. 
