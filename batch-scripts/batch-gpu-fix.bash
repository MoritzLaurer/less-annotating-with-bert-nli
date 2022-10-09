#!/bin/bash
#Set batch job requirements
#SBATCH -t 30:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=gpu-fix

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./NLI-experiments

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon


### 5000 & 10000 run (5426 max in this dataset)
## base-nli
#python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 10000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking

### run 5000 (3188)
## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 3 39 --batch_size 8 16 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-morality" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --hyperparam_study_date 20221006 --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-morality" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking
## base-standard
#python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 30 100 --batch_size 8 16 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-morality" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --hyperparam_study_date 20221006 --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-morality" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking

### run 5000 (3970)
## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 3 39 --batch_size 8 16 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-military" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --hyperparam_study_date 20221006 --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-military" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking
## base-standard
#python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 30 100 --batch_size 8 16 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "manifesto-military" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --hyperparam_study_date 20221006 --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-military" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking
