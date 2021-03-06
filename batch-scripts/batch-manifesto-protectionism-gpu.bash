#!/bin/bash
#Set batch job requirements
#SBATCH -t 35:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=protectionism

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./NLI-experiments

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon


### run 5000
## dataset size: 2116

### run 2500
## base-nli
python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 3 21 --batch_size 8 16 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 2500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
python analysis-transf-run.py --dataset "manifesto-protectionism" --sample_interval 2500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220430 --carbon_tracking
## base-standard
python analysis-transf-hyperparams.py --learning_rate 5e-7 9e-4 --epochs 30 100 --batch_size 8 16 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
python analysis-transf-run.py --dataset "manifesto-protectionism" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220430 --carbon_tracking

### run 1000
## base-nli
python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 1 15 --batch_size 4 8 16 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
python analysis-transf-run.py --dataset "manifesto-protectionism" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking
## base-standard
python analysis-transf-hyperparams.py --learning_rate 1e-6 9e-4 --epochs 30 100 --batch_size 4 8 16 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 1000 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
python analysis-transf-run.py --dataset "manifesto-protectionism" --sample_interval 1000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

### run 100 & 500
## base-nli
python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 1 11 --batch_size 4 8 16 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
python analysis-transf-run.py --dataset "manifesto-protectionism" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220428 --carbon_tracking
## base-standard
python analysis-transf-hyperparams.py --learning_rate 1e-6 9e-4 --epochs 40 100 --batch_size 4 8 16 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 100 500 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
python analysis-transf-run.py --dataset "manifesto-protectionism" --sample_interval 100 500 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220428 --carbon_tracking

