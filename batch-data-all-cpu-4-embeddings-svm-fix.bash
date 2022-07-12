#!/bin/bash
# Set batch job requirements
#SBATCH -t 10:00:00
#SBATCH --partition=thin
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=cpu-embed-svm

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
cd ./NLI-experiments

# install packages
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip uninstall -y codecarbon
python -m spacy download en_core_web_lg


python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 10000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "cap-us-court" --sample_interval 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"



python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 10000 --method "classical_ml" --model "logistic" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "cap-us-court" --sample_interval 10000 --method "classical_ml" --model "logistic" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"




### SVM with embeddings
## sentiment-news-econ
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "sentiment-news-econ" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
#python analysis-classical-run.py --dataset "sentiment-news-econ" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

## coronanet
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "coronanet" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
#python analysis-classical-run.py --dataset "coronanet" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

## cap-sotu
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "cap-sotu" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "cap-sotu" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

## cap-us-court
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "cap-us-court" --sample_interval 100 500 1000 2500 5000 5426 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "cap-us-court" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

## manifesto-8
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "manifesto-8" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "manifesto-8" --sample_interval 100 500 1000 2500 5000 10000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

## manifesto-military
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "manifesto-military" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "manifesto-military" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

## manifesto-morality
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "manifesto-morality" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "manifesto-morality" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

## manifesto-protectionism
#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "manifesto-protectionism" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --hyperparam_study_date 20220709 --vectorizer "embeddings"
python analysis-classical-run.py --dataset "manifesto-protectionism" --sample_interval 100 500 1000 2500 5000 --method "classical_ml" --model "SVM" --n_cross_val_final 3 --hyperparam_study_date 20220709 --zeroshot --vectorizer "embeddings"

