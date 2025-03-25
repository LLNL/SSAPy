#!/bin/bash
pip install git-lfs
git submodule update --init --recursive
git lfs install
git lfs pull
