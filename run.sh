#!/bin/bash
#SBATCH --partition=main #long
#SBATCH --output=./run_scripts/dt%j.txt
#SBATCH --error=./run_scripts/dt%jerror.txt 
#SBATCH --cpus-per-task=2                    # Ask for 4 CPUs
#SBATCH --gres=gpu:0                         # Ask for 1 titan xp
#SBATCH --mem=20G                             # Ask for 32 GB of RAM
#SBATCH --time=5:20:00                       # The job will run for 1 day

export HOME="/home/mila/r/razieh.shirzadkhani/EpidemyForecast"
module load miniconda/3
source $CONDA_ACTIVATE
conda activate /home/mila/r/razieh.shirzadkhani/.conda/envs/dt


pwd
python /home/mila/r/razieh.shirzadkhani/EpidemyForecast/contact_network_exp.py