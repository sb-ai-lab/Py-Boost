#!/bin/bash

rm -rf py_boost_venv
python -m venv py_boost_venv
source ./py_boost_venv/bin/activate

pip install -U pip
pip install -U poetry
pip install -U cupy-cuda110

poetry lock
poetry install
poetry build
