#!/bin/bash
git submodule update --init --recursive
git lfs install
git lfs pull

python3 setup.py build
python3 setup.py install