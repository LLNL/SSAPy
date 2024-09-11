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

Alternatively, you can run `sphinx-autobuild source _build/html` to start a server that watches for changes in `/docs`
and regenerates the HTML docs automatically while serving them at http://127.0.0.1:8000/.

Note that if you updated docstrings, you'll need to re-build and re-install ssapy before re-building the docs.
