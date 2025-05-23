#!/bin/bash
#SBATCH --account=pr_126_tandon_priority
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=2              
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=312GB
#SBATCH --job-name=wav2vec
#SBATCH --output=output/nn_pickle.out

module purge

singularity exec \
    --nv --overlay /scratch/th3129/region_decoding/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate blind_localization; cd script; python wav2vec_ibl.py --data=Neuronexus"