Getting Started
===============

System Prerequisites
--------------------

``SSAPy`` has the following minimum system requirements, which are assumed to be present on the machine where ``SSAPy`` is run:

.. csv-table:: System prerequisites for SSAPy
   :file: tables/system_prerequisites.csv
   :header-rows: 1

These requirements can be easily installed on most modern macOS and Linux systems.

.. tabs::

    .. tab:: Debian/Ubuntu

       .. code-block:: console

          apt update
          apt install build-essential git git-lfs python3 python3-distutils python3-venv graphviz

    .. tab:: RHEL

       .. code-block:: console

          dnf install epel-release
          dnf group install "Development Tools"
          dnf install git git-lfs gcc-gfortran python3 python3-pip python3-setuptools graphviz

    .. tab:: macOS Brew

       .. code-block:: console

          brew update
          brew install gcc git git-lfs python3 graphviz

Installation
------------

As the package has not yet been published on `PyPI <https://pypi.org/>`_, it CANNOT be installed using pip. Instead, use the following directions to install the package.

After cloning the main git repository, you must run the following command to clone the submodules:

.. code-block:: console

    git submodule update --init --recursive

Then checkout the large files

.. code-block:: console

   git lfs install
   git lfs pull

Then run the usual setup commands:

.. code-block:: console

    python3 setup.py build
    python3 setup.py install

Orekit dependency
^^^^^^^^^^^^^^^^^

`Orekit <https://www.orekit.org/>`_ is an optional dependency, including the ``Orekit`` Python wrapper that is hard to find. Clone the python wrappper from here:

    `https://gitlab.orekit.org/orekit-labs/python-wrapper <https://gitlab.orekit.org/orekit-labs/python-wrapper>`_

Alternatively, the ``Orekit`` python wrapper can be installed from `Anaconda <https://www.anaconda.com/>`_.
