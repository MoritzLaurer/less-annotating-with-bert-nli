#!/bin/bash
#Set batch job requirements
#SBATCH -t 10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=manifesto-8

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./NLI-experiments

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon


### run 10000
## base-nli
python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 10000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220428 --carbon_tracking
## base-standard
python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 10000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220428 --carbon_tracking

### run 5000
## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 3 36 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220502 --carbon_tracking
## base-standard
#python analysis-transf-hyperparams.py --learning_rate 5e-7 9e-4 --epochs 30 100 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220503 --carbon_tracking

### run 2500
## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 3 21 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 2500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 2500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220430 --carbon_tracking
## base-standard
#python analysis-transf-hyperparams.py --learning_rate 5e-7 9e-4 --epochs 30 100 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220430 --carbon_tracking

### run 1000
## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 1 15 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking
## base-standard
#python analysis-transf-hyperparams.py --learning_rate 1e-6 9e-4 --epochs 30 100 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 1000 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 1000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

### run 100 & 500
## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 1 11 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220428 --carbon_tracking
## base-standard
#python analysis-transf-hyperparams.py --learning_rate 1e-6 9e-4 --epochs 40 100 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 100 500 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 100 500 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220428 --carbon_tracking


## mini-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-6 9e-4 --epochs 1 16 --batch_size 8 16 --n_trials 10 --n_trials_sampling 6 --n_trials_pruning 4 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 100 500 1000 --method "nli" --model "MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 100 500 1000 --method "nli" --model "MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary" --n_cross_val_final 3 --hyperparam_study_date 20220422 --carbon_tracking
## mini-standard
#python analysis-transf-hyperparams.py --learning_rate 1e-6 9e-4 --epochs 40 80 --batch_size 8 16 --n_trials 10 --n_trials_sampling 6 --n_trials_pruning 4 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 100 500 1000 --method "standard_dl" --model "microsoft/xtremedistil-l6-h256-uncased" --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 100 500 1000 --method "standard_dl" --model "microsoft/xtremedistil-l6-h256-uncased" --n_cross_val_final 3 --hyperparam_study_date 20220422 --carbon_tracking

