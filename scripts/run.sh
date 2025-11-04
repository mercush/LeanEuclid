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

source ~/lean-experiments/.venv/bin/activate

echo "Starting autoformalization..."
python3 ~/lean-experiments/src/LeanEuclid/AutoFormalization/statement/autoformalize.py \
	--dataset Book \
	--category "" \
	--num_query 1 \
	--num_examples 5 \
	--project_dir "." \
	--tensor_parallel_size 4 \
	--start_index "$START_INDEX" \
	--model_type chat \
	--model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
	--preamble "import SystemE" \
	--typecheck "all" \
	--max_tokens 250 \
    --n_particles 5 \
    --lhts
    --lhts_power 4.0

echo "Starting evaluation..."
python3 AutoFormalization/statement/evaluate.py \
	--dataset Book \
	--category "" \
	--reasoning text-only \
	--num_examples 5

deactivate
