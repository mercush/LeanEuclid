#!/bin/bash
#SBATCH --job-name=leaneuclid_lhts
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=barba@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:2
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-1

# Load required modules
module load apptainer

# Create logs directory if it doesn't exist
mkdir -p logs

# Set configuration based on array task ID
case $SLURM_ARRAY_TASK_ID in
    0)
        TYPECHECK_FLAG="--typecheck none"
        PREFIX="lhts_only"
        ;;
    1)
        TYPECHECK_FLAG="--typecheck all"
        PREFIX="lhts_typecheck"
        ;;
esac

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================"
echo "Running LeanEuclid with LHTS"
echo "Model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
echo "GPUs: 2x h200"
echo "Typecheck flag: $TYPECHECK_FLAG"
echo "Prefix: $PREFIX"
echo "================================"

# Run the benchmark inside Singularity container with GPU support
singularity exec --nv lean_env.sif bash -c "source ~/.elan/env && time uv run python -u src/LeanEuclid/AutoFormalization/statement/autoformalize.py \
    --dataset Book \
    --category '' \
    --model_type chat \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --tensor_parallel_size 2 \
    --prefix $PREFIX \
    $TYPECHECK_FLAG \
    --lhts \
    --lhts_power 4.0 \
    --n_particles 5 \
    --max_tokens 250"

echo "================================"
echo "End Time: $(date)"
echo "Job completed"
