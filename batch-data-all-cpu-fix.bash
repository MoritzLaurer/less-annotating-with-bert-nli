#!/bin/bash
# Set batch job requirements
#SBATCH -t 10:00:00
#SBATCH --partition=thin
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=cpu

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
cd ./NLI-experiments

# install packages
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip uninstall -y codecarbon


## sentiment-news-econ
#python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 50 --n_cross_val_hyperparam 2 --context --dataset "sentiment-news-econ" --sample_interval 10000 --method "classical_ml" --model "SVM" --carbon_tracking
#python analysis-classical-run.py --dataset "sentiment-news-econ" --sample_interval 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

## coronanet
#python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 50 --n_cross_val_hyperparam 2 --context --dataset "coronanet" --sample_interval 10000 --method "classical_ml" --model "SVM" --carbon_tracking
#python analysis-classical-run.py --dataset "coronanet" --sample_interval 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

## cap-sotu
#python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 50 --n_cross_val_hyperparam 2 --context --dataset "cap-sotu" --sample_interval 10000 --method "classical_ml" --model "SVM" --carbon_tracking
#python analysis-classical-run.py --dataset "cap-sotu" --sample_interval 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

## cap-us-court
python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 50 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 10000 --method "classical_ml" --model "SVM" --carbon_tracking
python analysis-classical-run.py --dataset "cap-us-court" --sample_interval 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

## manifesto-8
python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 50 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --carbon_tracking
python analysis-classical-run.py --dataset "manifesto-8" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

## manifesto-military
#python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 50 --n_cross_val_hyperparam 2 --context --dataset "manifesto-military" --sample_interval 10000 --method "classical_ml" --model "SVM" --carbon_tracking
#python analysis-classical-run.py --dataset "manifesto-military" --sample_interval 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

## manifesto-morality
#python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 60 --n_cross_val_hyperparam 2 --context --dataset "manifesto-morality" --sample_interval 10000 --method "classical_ml" --model "SVM" --carbon_tracking
#python analysis-classical-run.py --dataset "manifesto-morality" --sample_interval 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

## manifesto-protectionism
python analysis-classical-hyperparams.py --n_trials 70 --n_trials_sampling 40 --n_trials_pruning 50 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 2500 5000 10000 --method "classical_ml" --model "SVM" --carbon_tracking
python analysis-classical-run.py --dataset "manifesto-protectionism" --sample_interval 2500 5000 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220429 --carbon_tracking

