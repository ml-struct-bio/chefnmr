#!/bin/bash
#SBATCH --job-name=SB-H10k-L128-cfg2-seed42
#SBATCH --output=./script/01-sampling/00-spectrabase/%x_%j.out
#SBATCH --partition=cryoem      # set partition
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4               # total number of tasks across all nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=12G

export CHEFNMR_DIR='/scratch/gpfs/ZHONGE/zx8205/dev/chefnmr'
export CKPT_PATH="$CHEFNMR_DIR/checkpoints/SB-H10kC80-L128-epoch5249.ckpt

module load anaconda3/2024.6 # Change to your Anaconda module or delete if not needed
conda activate nmr3d
cd "$CHEFNMR_DIR"

python -m src.main \
    +condition=h1c13nmr-10k-80 \
    +data=spectrabase \
    +embedder=hybrid-baseline \
    +lddt=threshold5124 \
    +model=dit-l  \
    +training_transform=center_rot_trans \
    +diffusion=edm-train_af3-sample_edm_sde \
    dataset_args.batch_size=128 \
    general.experiment_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    +aug_conf=conf3 \
    +guidance=cfg2 \
    general.ckpt_abs_path="$CKPT_PATH" \
    +exp=eval_ckpt \
    test_args.diffusion_samples=10 \
    dataset_args.test_args.test_samples=100000 \
    dataset_args.test_args.test_batch_size=496 \
    test_args.num_sampling_steps=50 \
    general.seed=42 \
    dataset_args.test_args.test_p_drop_c13nmr=1.0