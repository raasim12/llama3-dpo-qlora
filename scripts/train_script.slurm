#!/bin/bash
#SBATCH --job-name=llama_dpo_qlora_train     # Job name
#SBATCH --output=../logs/%x_%j.out        # STDOUT file (%x=job name, %j=job ID)
#SBATCH --error=../logs/%x_%j.err         # STDERR file
#SBATCH --qos=medium                   # Quality of Service
#SBATCH --partition=beacon              # Partition name
#SBATCH --ntasks=2                       # Number of tasks
#SBATCH --gres=gpu:2                   # Number of GPUs
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --mem=64G                      # Memory
#SBATCH --time=24:00:00                # Wall time (24 hours)
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-user=raasim@umd.edu  # Email for notifications

# ============================================================
# 1. Navigate to project directory
# ============================================================
cd /beacon-scratch/raasim/projects/llama3-dpo-qlora
mkdir -p ./logs

# ============================================================
# 2. Load modules and activate conda environment
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv # Activate your conda environment



# ============================================================
# 3. Set cache directories to scratch
# ============================================================
export HF_HOME=/beacon-scratch/raasim/huggingface
export HUGGINGFACE_HUB_CACHE=/beacon-scratch/raasim/huggingface
export TRANSFORMERS_CACHE=/beacon-scratch/raasim/huggingface
mkdir -p $HF_HOME

# ============================================================
# 4. Setup WandB environment variables (safe for multi-GPU)
# ============================================================
export WANDB_PROJECT=llama_dpo_qlora
export WANDB_WATCH=all
export WANDB_NAME=llama_dpo_qlora_run_${SLURM_JOB_ID}

# ============================================================
# 5. Other environment variables
# ============================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================
# 6. Print debug info
# ============================================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Python: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Cache directory: $HF_HOME"
echo "=========================================="

# ============================================================
# 7. Run the training script
# ============================================================
torchrun --nproc_per_node=2 train_dpo_qlora.py

echo "Training completed at $(date)"