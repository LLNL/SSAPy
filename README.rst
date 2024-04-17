SSAPy - Space Situational Awareness for Python
==============================================

.. image:: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml

.. image:: https://github.com/LLNL/SSAPy/actions/workflows/build-containers.yml/badge.svg
    :target: https://github.com/LLNL/SSAPy/actions/workflows/build-containers.yml

.. image:: https://github.com/LLNL/SSAPy/actions/workflows/pages/pages-build-deployment/badge.svg
    :target: https://llnl.github.io/SSAPy

.. image:: https://codecov.io/gh/LLNL/SSAPy/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/LLNL/SSAPy

SSAPy is a python package allowing for fast and precise orbital modeling.

SSAPy is much faster than other orbit modeling tools and offers:

- A variety of integrators, including Runge-Kutta, SciPy, SGP4, etc.
- Customizable force propagation models, including a variety of Earth gravity models, lunar gravity, radiation pressure, etc.
- Multiple-hypothesis tracking (MHT) UCT linker
- Vectorized computations
- Short arc probabilistic orbit determination
- Conjunction probability estimation
- Uncertainty quantification
- Monte Carlo data fusion
- Support for multiple coordinate frames (with coordinate frame conversions)

Installation
------------

For installation details, see the `Installing SSAPy <https://ssapy.github.io/ssapy/installation.html>`_ section of the documentation.

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

All documentation is hosted at `https://ssapy.github.io/ssapy/ <https://ssapy.github.io/ssapy/>`_.

The API documentation may also be seen by doing:

.. code-block:: bash

    python3
    >>> import ssapy
    >>> help(ssapy)

Contributing
------------

Contributing to SSAPy is relatively easy. Just send us a `pull request <https://help.github.com/articles/using-pull-requests/>`_. When you send your request, make `develop` the destination branch on the `SSAPy repository <https://gitub.com/LLNL/SSAPy>`_.

Your PR must pass SSAPy's unit tests and documentation tests, and must be `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ compliant. We enforce these guidelines with our CI process. To run these tests locally, and for helpful tips on git, see our `Contribution Guide <https://ssapy.reathedocs.io/en/latest/contribution_guide.html>`_.

SSAPy's `develop` branch has the latest contributions. Pull requests should target `develop`, and users who want the latest package versions, features, etc. can use `develop`.

Releases
--------

For multi-user site deployments or other use cases that need very stable software installations, we recommend using SSAPy's `stable releases <https://github.com/LLNL/SSAPy/releases>`_.

Each SSAPy release series also has a corresponding branch, e.g. `releases/v0.14` has `0.14.x` versions of SSAPy, and `releases/v0.13` has `0.13.x` versions. We backport important bug fixes to these branches but we do not advance the package versions or make other changes that would change the way SSAPy concretizes dependencies within a release branch. So, you can base your SSAPy deployment on a release branch and `git pull` to get fixes, without the package churn that comes with develop.

The latest release is always available with the `releases/latest` tag.

See the `docs on releases <https://ssapy.reathedocs.io/en/latest/contribution_guide.html#releases>`_ for more details.

Code of Conduct
---------------

Please note that SSAPy has a `Code of Conduct <https://github.com/LLNL/SSAPy/blob/main/CODE_OF_CONDUCT.md>`_. By participating in the SSAPy community, you agree to abide by its rules.

Authors
-------

Many thanks go to SSAPy's `contributors <https://github.com/llnl/ssapy/graphs/contributors>`_.

SSAPy was created by:

- `Michael Schneider <https://people.llnl.gov/schneider42>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Josh Meyers <https://kipac.stanford.edu/people/josh-meyers>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now at `KIPAC <https://kipac.stanford.edu/>`_)
- `Edward Schlafly <https://www.stsci.edu/stsci-research/research-directory/edward-schlafly>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now `STScI <https://www.stsci.edu/>`_)
- Julia Ebert (formerly `LLNL <https://www.llnl.gov/>`_, now at Outrider).

..
    Citing SSAPy
    ^^^^^^^^^^^^

    If you are referencing SSAPy in a publication, please cite the following paper:

        - citation

    On GitHub, you can copy this citation in APA or BibTeX format via the "Cite this repository" button. Or, see the comments in CITATION.cff for the raw BibTex.

License
-------

SSAPy is distributed under the terms of the MIT license. All new contributions must be made under the MIT license.

See `Link to license <https://github.com/LLNL/SSAPy/blob/main/LICENSE>`_ and `NOTICE <https://github.com/LLNL/SSAPy/blob/main/NOTICE>`_ for details.

SPDX-License-Identifier: MIT

LLNL-CODE-862420
