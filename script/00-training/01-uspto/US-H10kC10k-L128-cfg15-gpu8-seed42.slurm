#!/bin/bash
#SBATCH --job-name=US-H10kC10k-L128-cfg15-gpu8-seed42
#SBATCH --output=./script/00-training/01-uspto/%x_%j.out
#SBATCH --partition=cryoem      # set partition
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=4               # total number of tasks across all nodes
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=48G
export CHEFNMR_DIR='/scratch/gpfs/ZHONGE/zx8205/dev/chefnmr'

module load anaconda3/2024.6 # Change to your Anaconda module or delete if not needed
conda activate nmr3d
cd "$CHEFNMR_DIR"

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_IB_DISABLE=0

srun python -m src.main \
    +condition=h1c13nmr-10k-10k \
    +data=uspto \
    +device=gpu8 \
    +embedder=hybrid-baseline \
    +lddt=threshold5124 \
    +model=dit-l \
    +training_transform=center_rot_trans \
    +diffusion=edm-train_af3-sample_edm_sde \
    dataset_args.batch_size=128 \
    general.experiment_name="$SLURM_JOB_NAME" \
    general.seed=42 \
    +aug_conf=conf3 \
    +guidance=cfg15