#!/bin/bash
#Set batch job requirements
#SBATCH -t 25:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=coronanet

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./NLI-experiments

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon

### run 2500
## base-standard
python analysis-transf-hyperparams.py --learning_rate 5e-7 9e-4 --epochs 20 70 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "coronanet" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
python analysis-transf-run.py --dataset "coronanet" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220501 --carbon_tracking

