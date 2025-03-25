#!/bin/bash
apt update
apt install build-essential git git-lfs python3 python3-distutils python3-venv graphviz
git submodule update --init --recursive
git lfs install
git lfs pull
