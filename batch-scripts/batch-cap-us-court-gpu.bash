#!/bin/bash
#Set batch job requirements
#SBATCH -t 65:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=cap-us-court

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./NLI-experiments

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon


### 100 & 500 run
## base-nli
python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 3 39 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --hyperparam_study_date 20221006 --carbon_tracking
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking
## base-standard
python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 40 100 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 100 500 --method "standard_dl" --model "microsoft/deberta-v3-base" --hyperparam_study_date 20221006 --carbon_tracking
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 100 500 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking

### 1000 run
## base-nli
python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 3 39 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --hyperparam_study_date 20221006 --carbon_tracking
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking
## base-standard
python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 20 80 --batch_size 8 16 32 --n_trials 15 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 1000 --method "standard_dl" --model "microsoft/deberta-v3-base" --hyperparam_study_date 20221006 --carbon_tracking
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 1000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking

### 2500 run
## base-nli
python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 3 39 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 2500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --hyperparam_study_date 20221006 --carbon_tracking
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 2500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking
## base-standard
python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 20 70 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --hyperparam_study_date 20221006 --carbon_tracking
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 2500 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking

### 5000 & 10000 run (5426 max in this dataset)
## base-nli
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 5000, 10000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --hyperparam_study_date 20221006 --carbon_tracking
## base-standard
python analysis-transf-run.py --dataset "cap-us-court" --sample_interval 5000, 10000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20221006 --hyperparam_study_date 20221006 --carbon_tracking

