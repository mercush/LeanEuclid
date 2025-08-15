#!/bin/bash

START_INDEX=${1:-1}

if [ "$START_INDEX" -eq 1 ]; then
    read -p "Starting from index 1. This will delete the 'result/' directory. Are you sure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting 'result/' and continuing."
	rm -rf result
	rm -rf tmp
    else
        echo "Operation cancelled by user."
        exit 1
    fi
else
    echo "Starting from index $START_INDEX."
fi

source ~/lean-experiments/mau/src/LeanEuclid/venv/bin/activate
python3 AutoFormalization/statement/autoformalize.py \
	--dataset Book \
	--category "" \
	--reasoning text-only \
	--num_query 1 \
	--num_examples 5 \
	--project_dir "." \
	--tensor_parallel_size 1 \
        --start_index "$START_INDEX" \
	--model_type featherless \
	--model_name "AI-MO/Kimina-Prover-72B" \
	--preamble "import SystemE" \
	--with_cot True \
	--max_tokens 4000

python3 AutoFormalization/statement/evaluate.py \
	--dataset Book \
	--category "" \
	--reasoning text-only \
	--num_examples 5
deactivate
