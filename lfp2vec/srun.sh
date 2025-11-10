#!/bin/bash
#SBATCH --account=pr_136_tandon_advanced
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=2              
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=lfp2vec
#SBATCH --output=output/nn_pickle.out

module purge

singularity exec \
    --nv --overlay /scratch/yfw215/lfp2vec/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python run_train.py"

# srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 --time=2:00:00 --mem=128GB --account=pr_136_tandon_advanced /bin/bash45uj