#!/bin/bash
#SBATCH --job-name=SB-H10kC80-S128-cfg2-gpu4-seed42
#SBATCH --output=./script/00-training/00-spectrabase/%x_%j.out
#SBATCH --partition=cryoem      # set partition
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4               # total number of tasks across all nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=12G
export CHEFNMR_DIR='/scratch/gpfs/ZHONGE/zx8205/dev/chefnmr'

module load anaconda3/2024.6 # Change to your Anaconda module or delete if not needed
conda activate nmr3d
cd "$CHEFNMR_DIR"
srun python -m src.main \
    +condition=h1c13nmr-10k-80 \
    +data=spectrabase \
    +device=gpu4 \
    +embedder=hybrid-baseline \
    +lddt=threshold5124 \
    +model=dit-s \
    +training_transform=center_rot_trans \
    +diffusion=edm-train_af3-sample_edm_sde \
    dataset_args.batch_size=128 \
    general.experiment_name="$SLURM_JOB_NAME" \
    +aug_conf=conf3 \
    +guidance=cfg2 \
    general.seed=42