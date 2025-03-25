#!/bin/bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs -y
git submodule update --init --recursive
git lfs install
git lfs pull
