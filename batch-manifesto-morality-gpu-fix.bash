#!/bin/bash
#Set batch job requirements
#SBATCH -t 9:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=manifesto-morality

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./NLI-experiments

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon

## base-nli
python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 1 15 --batch_size 4 8 16 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-morality" --sample_interval 100 500 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
python analysis-transf-run.py --dataset "manifesto-morality" --sample_interval 100 500 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220430 --carbon_tracking



