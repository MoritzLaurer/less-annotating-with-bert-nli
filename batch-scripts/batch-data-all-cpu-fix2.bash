#!/bin/bash
# Set batch job requirements
#SBATCH -t 2:00:00
#SBATCH --partition=thin
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=cpu-fix2
#SBATCH --ntasks=32

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


#### script run to fix some issues during previous runs - script should be deletable

# had typo in previous run where run was 3000 instead of 5000 for logistic on cap-sotu
python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "cap-sotu" --sample_interval 5000 --method "classical_ml" --model "logistic" --hyperparam_study_date 20220713 --vectorizer "tfidf"
python analysis-classical-run.py --dataset "cap-sotu" --sample_interval 5000 --method "classical_ml" --model "logistic" --n_cross_val_final 3 --hyperparam_study_date 20220713 --vectorizer "tfidf"


