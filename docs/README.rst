SSAPy Documentation
===================

Building the docs
-----------------

First, build and _install_ `ssapy` following the `Installing SSAPy <https://LLNL.github.io/SSAPy/installation.html>`_ section of the documentation.

Then, to build the HTML documentation locally, from the `docs` directory, run

.. code-block:: bash

    make html

(run it twice to generate all files.)

Then open `_build/html/index.html` to browse the docs locally.

Note that if you updated docstrings, you'll need to re-build and re-install ssapy before re-building the docs.
