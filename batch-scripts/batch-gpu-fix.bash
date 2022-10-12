#!/bin/bash
#Set batch job requirements
#SBATCH -t 30:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=g-fix

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./NLI-experiments

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon




# earlier 15runs NLI:  prior run: 3 39 epochs, 15 trials; (manifesto-8 3 42, 16 for 100 - 1000)

## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 3 42 --batch_size 16 32 --n_trials 16 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --hyperparam_study_date 20221006 --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 100 500 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking

## nli manifesto-8 sample 1000 issue (not 500)
#python analysis-transf-hyperparams.py --learning_rate 1e-6 5e-4 --epochs 3 42 --batch_size 16 32 --n_trials 16 --n_trials_sampling 9 --n_trials_pruning 6 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --hyperparam_study_date 20221006 --carbon_tracking
#python analysis-transf-run.py --dataset "manifesto-8" --sample_interval 1000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20221006 --carbon_tracking



