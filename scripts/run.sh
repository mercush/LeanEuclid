#!bash
source ~/lean-experiments/mau/src/LeanEuclid/venv/bin/activate
python3 -m AutoFormalization.statement.autoformalize \
	--dataset Book \
	--category "" \
	--reasoning text-only \
	--num_query 1 \
	--num_examples 5 \
	--model_type gemini
	--model_name "gemini-2.0-flash" \
	--preamble "import SystemE" \
	--lean_version "v4.8.0-rc2" \
	--project_dir "." \
	--tensor_parallel_size 1
