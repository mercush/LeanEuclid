#!/bin/bash

START_INDEX=${1:-1}

if [ "$START_INDEX" -eq 1 ]; then
    read -p "Starting from index 1. This will delete the 'result/proof' directory. Are you sure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting 'result/proof' and continuing."
        rm -rf result/proof
    else
        echo "Operation cancelled by user."
        exit 1
    fi
else
    echo "Starting from index $START_INDEX."
fi

source ~/lean-experiments/mau/src/LeanEuclid/venv/bin/activate
python3 -m AutoFormalization.proof.autoformalize \
rm -rf result
rm -rf tmp
source ~/lean-experiments/mau/src/LeanEuclid/venv/bin/activate
python3 AutoFormalization/statement/autoformalize.py \
	--dataset Book \
	--category "" \
	--reasoning text-only \
	--num_query 1 \
	--num_examples 5 \
	--project_dir "." \
	--tensor_parallel_size 1 \
        --start_index "$START_INDEX"
	--model_type base \
	--model_name "AI-MO/Kimina-Prover-72B" \
	--preamble "import SystemE" \
	--project_dir "." \
	--tensor_parallel_size 8
deactivate
