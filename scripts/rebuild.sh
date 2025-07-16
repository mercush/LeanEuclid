#!/bin/bash
git pull
python3 -m venv venv
source venv/bin/activate
pip install smt-portfolio tqdm genlm-control lean_interact 
export PYTHONPATH=/LeanEuclid