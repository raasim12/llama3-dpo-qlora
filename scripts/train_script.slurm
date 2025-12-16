#!/bin/bash
#SBATCH --job-name=vit_embed_train     # Job name
#SBATCH --output=../logs/%x_%j.out        # STDOUT file (%x=job name, %j=job ID)
#SBATCH --error=../logs/%x_%j.err         # STDERR file
#SBATCH --qos=medium                   # Quality of Service
#SBATCH --partition=beacon              # Partition name
#SBATCH --ntasks=2                       # Number of tasks
#SBATCH --gres=gpu:2                   # Number of GPUs
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --mem=64G                      # Memory
#SBATCH --time=5:00:00                # Wall time (3 hours)
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-user=raasim@umd.edu  # Email for notifications

# ============================================================
# 1. Navigate to project directory
# ============================================================
cd /beacon-scratch/raasim/projects/vit_EMBED
mkdir -p ./logs

# ============================================================
# 2. Load modules and activate conda environment
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv # Activate your conda environment

# ============================================================
# 3. Setup WandB environment variables (safe for multi-GPU)
# ============================================================
export WANDB_PROJECT=llama_dpo_qlora
export WANDB_WATCH=all
export WANDB_NAME=llama_dpo_qlora_run_${SLURM_JOB_ID}

# ============================================================
# 4. Print some info and run the training
# ============================================================
echo "Using Python from: $(which python)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"

#CPU threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the training script
torchrun --nproc_per_node=2 train_dpo_qlora.py