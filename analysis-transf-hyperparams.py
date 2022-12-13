
#### This script conducts the hyperparameter search for transformer algorithms
## the script is executed via the batch files in the batch-scripts folder

EXECUTION_TERMINAL = True

# ## Load packages
import transformers
import datasets
import torch
import optuna

import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
from datetime import date
import random
import os
import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import time
import joblib

## set global seed for reproducibility and against seed hacking
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

# setting working directory for local runs
print(os.getcwd())
if (EXECUTION_TERMINAL==False) and ("NLI-experiments" not in os.getcwd()):
    os.chdir("./NLI-experiments")
print(os.getcwd())



# ## Main parameters

### argparse for command line execution
import argparse
# https://realpython.com/command-line-interfaces-python-argparse/
# https://docs.python.org/3/library/argparse.html

# Create the parser
parser = argparse.ArgumentParser(description='Run hyperparameter tuning with different algorithms on different datasets')

## Add the arguments
# arguments for hyperparameter search
parser.add_argument('-lr', '--learning_rate', type=float, nargs='+',
                    help='List of two floats: first lower then upper bound of learning rate hyperparameter search')
parser.add_argument('-epochs', '--epochs', type=int, nargs='+',
                    help='List of two floats: first lower then upper bound of epoch hyperparameter search')
parser.add_argument('-b', '--batch_size', type=int, nargs='+',  #default=[8],
                    help='Choose a batch size. Can be single number, or multiple.')
parser.add_argument('-t', '--n_trials', type=int,
                    help='How many optuna trials should be run?')
parser.add_argument('-ts', '--n_trials_sampling', type=int,
                    help='After how many trials should optuna start sampling?')
parser.add_argument('-tp', '--n_trials_pruning', type=int,
                    help='After how many trials should optuna start pruning?')
parser.add_argument('-cvh', '--n_cross_val_hyperparam', type=int, default=2,
                    help='How many times should optuna cross validate in a single trial?')
parser.add_argument('-context', '--context', action='store_true',
                    help='Take surrounding context sentences into account. Only use flag if context available.')

# arguments for both hyperparam and test script
parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality"')
parser.add_argument('-samp', '--sample_interval', type=int, nargs='+',
                    help='Interval of sample sizes to test.')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "nli", "standard_dl"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-tqdm', '--disable_tqdm', action='store_true',
                    help='Adding the flag enables tqdm for progress tracking')
parser.add_argument('-carbon', '--carbon_tracking', action='store_true',
                    help='Adding the flag enables carbon tracking via CodeCarbon')  # not used, as CodeCarbon caused bugs https://github.com/mlco2/codecarbon/issues/305

# arguments only for test script
parser.add_argument('-cvf', '--n_cross_val_final', type=int, default=3,
                    help='For how many different random samples should the algorithm be tested at a given sample size?')
parser.add_argument('-hp_date', '--hyperparam_study_date', type=str,
                    help='Date string to specifiy which hyperparameter run should be selected. e.g. "20220304"')



### choose arguments depending on execution in terminal or in script for testing
if EXECUTION_TERMINAL == True:
  print("Arguments passed via the terminal:")
  # Execute the parse_args() method
  args = parser.parse_args()
  # To show the results of the given option to screen.
  print("")
  for key, value in parser.parse_args()._get_kwargs():
      #if value is not None:
          print(value, "  ", key)

elif EXECUTION_TERMINAL == False:
  # parse args if not in terminal, but in script. adapt manually
  args = parser.parse_args(["--learning_rate", "1.2", "1.5", "--epochs", "3", "16", "--batch_size", "8", "16", "--n_trials", "12", "--n_trials_sampling", "5", "--n_trials_pruning", "4", "--n_cross_val_hyperparam", "3", 
                            "--context", "--dataset", "manifesto-military", "--sample_interval", "1000", "2500", #"100", "500", "1000", "2500", "5000", "10000", 
                            "--method", "nli", "--model", "MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary", #"microsoft/xtremedistil-l6-h256-uncased", #"MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary", 
                            "--n_cross_val_final", "3", "--hyperparam_study_date", "20220418", "--carbon_tracking"])


### args only for hyperparameter tuning
LR_LOW, LR_HIGH = args.learning_rate  # 5e-6, 5e-4
EPOCHS_LOW, EPOCHS_HIGH = args.epochs  # 3, 16

TRAIN_BATCH_SIZE = args.batch_size

N_TRIALS = args.n_trials
N_STARTUP_TRIALS_SAMPLING = args.n_trials_sampling
N_STARTUP_TRIALS_PRUNING = args.n_trials_pruning
CROSS_VALIDATION_REPETITIONS_HYPERPARAM = args.n_cross_val_hyperparam
CONTEXT = args.context   # not in use. would need to adapt code below. Currently always includes context & not-context in hyperparameter search. seems like best option

### args for both hyperparameter tuning and test runs
# choose dataset
DATASET_NAME = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "nsp"
MODEL_NAME = args.model

DISABLE_TQDM = args.disable_tqdm
CARBON_TRACKING = args.carbon_tracking

### args only for test runs
HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final



# ## Load data
if DATASET_NAME == "cap-us-court":
  df_cl = pd.read_csv("./data_clean/df_cap_us_court_all.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_cap_us_court_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_cap_us_court_test.csv", index_col="idx")
elif DATASET_NAME == "sentiment-news-econ":
  df_cl = pd.read_csv("./data_clean/df_sentiment_news_econ_all.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_sentiment_news_econ_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_sentiment_news_econ_test.csv", index_col="idx")
elif DATASET_NAME == "cap-sotu":
  df_cl = pd.read_csv("./data_clean/df_cap_sotu_all.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_cap_sotu_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_cap_sotu_test.csv", index_col="idx")
elif "manifesto-8" in DATASET_NAME:
  df_cl = pd.read_csv("./data_clean/df_manifesto_all.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_manifesto_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_manifesto_test.csv", index_col="idx")
#elif DATASET_NAME == "manifesto-complex":
#  df_cl = pd.read_csv("./data_clean/df_manifesto_complex_all.csv", index_col="idx")
#  df_train = pd.read_csv("./data_clean/df_manifesto_complex_train.csv", index_col="idx")
#  df_test = pd.read_csv("./data_clean/df_manifesto_complex_test.csv", index_col="idx")
elif DATASET_NAME == "coronanet":
  df_cl = pd.read_csv("./data_clean/df_coronanet_20220124_all.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_coronanet_20220124_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_coronanet_20220124_test.csv", index_col="idx")
elif DATASET_NAME == "manifesto-military":
  df_cl = pd.read_csv("./data_clean/df_manifesto_military_cl.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_manifesto_military_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_manifesto_military_test.csv", index_col="idx")
elif DATASET_NAME == "manifesto-protectionism":
  df_cl = pd.read_csv("./data_clean/df_manifesto_protectionism_cl.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_manifesto_protectionism_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_manifesto_protectionism_test.csv", index_col="idx")
elif DATASET_NAME == "manifesto-morality":
  df_cl = pd.read_csv("./data_clean/df_manifesto_morality_cl.csv", index_col="idx")
  df_train = pd.read_csv("./data_clean/df_manifesto_morality_train.csv", index_col="idx")
  df_test = pd.read_csv("./data_clean/df_manifesto_morality_test.csv", index_col="idx")
else:
  raise Exception(f"Dataset name not found: {DATASET_NAME}")

## special preparation of manifesto simple dataset - chose 8 or 44 labels
if DATASET_NAME == "manifesto-8":
  df_cl["label_text"] = df_cl["label_domain_text"]
  df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]
  df_train["label_text"] = df_train["label_domain_text"]
  df_train["label"] = pd.factorize(df_train["label_text"], sort=True)[0]
  df_test["label_text"] = df_test["label_domain_text"]
  df_test["label"] = pd.factorize(df_test["label_text"], sort=True)[0]


## reduce max sample size interval list to fit to max df_train length
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample < len(df_train)]
if len(df_train) < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len(df_train)]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)


LABEL_TEXT_ALPHABETICAL = np.sort(df_cl.label_text.unique())
TRAINING_DIRECTORY = f"results-raw/{DATASET_NAME}"

## data checks
print(DATASET_NAME, "\n")
# verify that numeric label is in alphabetical order of label_text
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").label.tolist()  # label num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text 
assert all(labels_num_via_numeric == labels_num_via_text)

# test if columns from full-train-test datasets correspond
assert df_cl.columns.tolist() == df_train.columns.tolist() == df_test.columns.tolist()



# ## Load helper functions

import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)

from helpers import format_nli_testset, format_nli_trainset, data_preparation  # custom_train_test_split, custom_train_test_split_sent_overlapp
from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer
from helpers import compute_metrics_standard, compute_metrics_nli_binary, compute_metrics_classical_ml, clean_memory

### load suitable hypotheses_hyperparameters and text formatting function
from hypothesis_hyperparams import hypothesis_hyperparams


### load the hypothesis hyperparameters for the respective dataset
hypothesis_hyperparams_dic, format_text = hypothesis_hyperparams(dataset_name=DATASET_NAME, df_cl=df_cl)

# check which template fits to standard_dl or NLI
print("")
print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template])
print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" not in template])

# in case -context flag is passed, but dataset actually does not have context
nli_templates = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" not in template]
if "context" not in nli_templates:
  CONTEXT = False





# ## Hyperparameter tuning

# carbon tracker  https://github.com/mlco2/codecarbon/tree/master
#if CARBON_TRACKING:
#  from codecarbon import OfflineEmissionsTracker
#  tracker = OfflineEmissionsTracker(country_iso_code="NLD", log_level='warning', measure_power_secs=300,  #output_dir=TRAINING_DIRECTORY,
#                                    project_name=f"{DATASET_NAME}-{MODEL_NAME.split('/')[-1]}")
#  tracker.start()


# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa" in MODEL_NAME: fp16_bool = False  # mDeBERTa does not support FP16 yet


def inference_run_transformer(df_train=None, df_dev=None, random_seed=None, hyperparams_dic=None, n_sample=None):
  # can add random_seed here, but not really necessary, because HF Transformers Trainer also uses 42 as default seed

  clean_memory()
  model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
  encoded_dataset = tokenize_datasets(df_train_samp=df_train, df_test=df_dev, tokenizer=tokenizer, method=METHOD, max_length=None)

  train_args = set_train_args(hyperparams_dic=hyperparams_dic, training_directory=TRAINING_DIRECTORY, disable_tqdm=DISABLE_TQDM, evaluation_strategy="no", fp16=fp16_bool) 
  trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=encoded_dataset, train_args=train_args, 
                           method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
  clean_memory()

  if n_sample != 0:
    trainer.train()
  results = trainer.evaluate()  # eval_dataset=encoded_dataset_test

  clean_memory()

  run_info_dic = {"method": METHOD, "n_sample": n_sample, "model": MODEL_NAME, "results": results, "hyper_params": hyperparams_dic}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic

  return run_info_dic


def optuna_objective(trial, hypothesis_hyperparams_dic=None, n_sample=None, df_train=None, df=None):  #df_train=None,
  np.random.seed(SEED_GLOBAL)  # don't understand why this needs to be run here at each iteration. it should stay constant once set globally?! Explanation could be this: https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f

  if METHOD == "nli":
    hyperparam_epochs = {"num_train_epochs": trial.suggest_int("num_train_epochs", EPOCHS_LOW, EPOCHS_HIGH, log=False, step=5)}
    hyperparam_lr_scheduler = {"lr_scheduler_type": "linear"}
    hyperparam_warmup = {"warmup_ratio":  trial.suggest_categorical("warmup_ratio", [0.06, 0.2, 0.4, 0.6])}   # only tested this for hyperparam search on 2500 samp
    #hyperparam_warmup = {"warmup_ratio":  0.06}
  elif METHOD == "standard_dl":
    hyperparam_epochs = {"num_train_epochs": trial.suggest_int("num_train_epochs", EPOCHS_LOW, EPOCHS_HIGH, log=False, step=10)}
    hyperparam_lr_scheduler = {"lr_scheduler_type": "constant"}
    hyperparam_warmup = {"warmup_ratio":  0.06}
  hyperparams = {
    #"lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["constant", "linear"]),
    #"lr_scheduler_type": "linear",  # hf default: linear, not constant.  FB paper uses constant
    **hyperparam_lr_scheduler,
    **hyperparam_warmup,
    "learning_rate": trial.suggest_float("learning_rate", LR_LOW, LR_HIGH, log=True),  # deberta-base-nli liked lr at 1e-6 or lower
    #"learning_rate": trial.suggest_categorical("learning_rate", [5e-6, 2e-5, 8e-5, 5e-4]),  # deberta-base-nli liked lr at 1e-6 or lower
    #"num_train_epochs": trial.suggest_categorical("num_train_epochs", [EPOCHS_LOW, EPOCHS_HIGH]),
    **hyperparam_epochs,
    #"num_train_epochs": trial.suggest_int("num_train_epochs", EPOCHS_LOW, EPOCHS_HIGH, log=False, step=3),  # step=10
    #"num_train_epochs": 100,
    #"seed": trial.suggest_categorical("seed_hf_trainer", np.random.choice(range(43), size=5).tolist() ),  # for training order of examples in hf trainer. For standard_dl, this also influences head
    "seed": SEED_GLOBAL,
    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", TRAIN_BATCH_SIZE),  # mDeBERTa cannot handle 32+ batch  (even DeBERTA-base crashes with 16 batch and normal ram)
    #"per_device_train_batch_size": 8,
    #"warmup_ratio": 0.06,  # hf default: 0  # FB paper uses 0.0
    "weight_decay": 0.05,
    "per_device_eval_batch_size": 160,  # increase eval speed
    #"gradient_accumulation_steps": 2,
  }

  if METHOD == "nli":
    # CONTEXT decides if hypothesis templates with or without context are chosen in format_text function
    # disactivated here. Best to always test context in hyperparam search
    #if CONTEXT == True:
    #  hypothesis_template_nli = [template for template in list(hypothesis_hyperparams_dic.keys()) if ("not_nli" not in template) and ("context" in template)]
    #elif CONTEXT == False:
    hypothesis_template_nli = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" not in template]
    #else:
    #  raise Exception(f"CONTEXT variable is {CONTEXT}. Can only be True/False")
    hypothesis_template = trial.suggest_categorical("hypothesis_template", hypothesis_template_nli)
    hyperparams_optuna = dict(**hyperparams, **{"hypothesis_template": hypothesis_template})
  elif METHOD == "standard_dl":
    # not choosing a hypothesis template here, but the way of formatting the input text (e.g. with preceding sentence or not). need to keep same object names though
    #if CONTEXT == True:
    #  text_template_standard_dl = [template for template in list(hypothesis_hyperparams_dic.keys()) if ("not_nli" in template) and ("context" in template)]
    #elif CONTEXT == False:
    text_template_standard_dl = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template]
    #else:
    #  raise Exception(f"CONTEXT variable is {CONTEXT}. Can only be True/False")
    if len(text_template_standard_dl) >= 2:  # if there is only one reasonable text format for standard_dl
      hypothesis_template = trial.suggest_categorical("hypothesis_template", text_template_standard_dl)
    else:
      hypothesis_template = text_template_standard_dl[0]
    hyperparams_optuna = dict(**hyperparams, **{"hypothesis_template": hypothesis_template})
  else:
    raise Exception(f"The specified METHOD '{METHOD}' is not covered by the code.")
  
  trial.set_user_attr("hyperparameters_all", hyperparams_optuna)
  print("Hyperparameters for this run: ", hyperparams_optuna)


  # cross-validation loop. Objective: determine F1 for specific sample for specific hyperparams, without a test set
  run_info_dic_lst = []
  for step_i, random_seed_cross_val in enumerate(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_HYPERPARAM)):
    np.random.seed(SEED_GLOBAL)
    df_train_samp, df_dev_samp = data_preparation(random_seed=random_seed_cross_val, df_train=df_train, df=df,
                                                  hypothesis_template=hypothesis_template, 
                                                  hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template], 
                                                  n_sample=n_sample, format_text_func=format_text, method=METHOD)
    #import pdb; pdb.set_trace()
    run_info_dic = inference_run_transformer(df_train=df_train_samp, df_dev=df_dev_samp, hyperparams_dic=hyperparams, n_sample=n_sample)
    run_info_dic_lst.append(run_info_dic)
    
    # Report intermediate objective value.
    intermediate_value = (run_info_dic["results"]["eval_f1_macro"] + run_info_dic["results"]["eval_f1_micro"]) / 2
    trial.report(intermediate_value, step_i)
    # Handle pruning based on the intermediate value.
    if trial.should_prune() and (CROSS_VALIDATION_REPETITIONS_HYPERPARAM > 1):
      raise optuna.TrialPruned()
    if n_sample == 999_999:  # no cross-validation necessary for full dataset
      break

  f1_macro_crossval_lst = [dic["results"]["eval_f1_macro"] for dic in run_info_dic_lst]
  f1_micro_crossval_lst = [dic["results"]["eval_f1_micro"] for dic in run_info_dic_lst]
  metric_details = {
      "F1_macro_mean": np.mean(f1_macro_crossval_lst), "F1_micro_mean": np.mean(f1_micro_crossval_lst),
      "F1_macro_std": np.std(f1_macro_crossval_lst), "F1_micro_std": np.std(f1_micro_crossval_lst)
  }
  trial.set_user_attr("metric_details", metric_details)

  results_lst = [dic["results"] for dic in run_info_dic_lst]
  trial.set_user_attr("results_trainer", results_lst)

  # objective: maximise mean of f1-macro & f1-micro. HP should be good for imbalanced data, but also important/big classes
  metric = (np.mean(f1_macro_crossval_lst) + np.mean(f1_micro_crossval_lst)) / 2
  std = (np.std(f1_macro_crossval_lst) + np.std(f1_micro_crossval_lst)) / 2

  print(f"\nFinal metrics for run: {metric_details}. With hyperparameters: {hyperparams}\n")

  return metric





#warnings.filterwarnings(action='ignore')
#from requests import HTTPError  # for catching HTTPError, if model download does not work for one trial for some reason
# catch catch following error. unclear if good to catch this. [W 2022-01-12 14:18:30,377] Trial 9 failed because of the following error: HTTPError('504 Server Error: Gateway Time-out for url: https://huggingface.co/api/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')

def run_study(n_sample=None):
  np.random.seed(SEED_GLOBAL)

  optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS_PRUNING, n_warmup_steps=0, interval_steps=1, n_min_trials=1)  # https://optuna.readthedocs.io/en/stable/reference/pruners.html
  optuna_sampler = optuna.samplers.TPESampler(seed=SEED_GLOBAL, consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False, 
                                              n_startup_trials=N_STARTUP_TRIALS_SAMPLING, n_ei_candidates=24, multivariate=False, group=False, warn_independent_sampling=True, constant_liar=False)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
  study = optuna.create_study(direction="maximize", study_name=None, pruner=optuna_pruner, sampler=optuna_sampler)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html

  study.optimize(lambda trial: optuna_objective(trial, hypothesis_hyperparams_dic=hypothesis_hyperparams_dic, n_sample=n_sample, df_train=df_train, df=df_cl),   #df_train=df_train,
                n_trials=N_TRIALS, show_progress_bar=True, #catch=(HTTPError,) 
                )  # Objective function with additional arguments https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
  return study


hp_study_dic = {}
for n_sample in tqdm.tqdm(N_SAMPLE_DEV):
  study = run_study(n_sample=n_sample)
  hp_study_dic_step = {f"hp_search_{METHOD}_{n_sample}": {"n_max_sample_class": n_sample, "method": METHOD, "dataset": DATASET_NAME, "algorithm": MODEL_NAME, "optuna_study": study} } 
  hp_study_dic.update(hp_study_dic_step)
  # save study_dic after each new study added
  while len(str(n_sample)) <= 4:
    n_sample = "0" + str(n_sample)
  joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")

  ## copy some hps for higher sample sizes to save compute
  if n_sample == "05000":
    joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_10000samp_{HYPERPARAM_STUDY_DATE}.pkl")
  if (n_sample == "02500") and (DATASET_NAME == "cap-us-court"):  # dataset is particularly slow and benefit of additional hp-search is unclear
    joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_05000samp_{HYPERPARAM_STUDY_DATE}.pkl")
    joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_05426samp_{HYPERPARAM_STUDY_DATE}.pkl")
  if (n_sample == "02500") and (DATASET_NAME == "sentiment-news-econ"):  # slow dataset, small data size difference between intervals
    joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_03000samp_{HYPERPARAM_STUDY_DATE}.pkl")


## stop carbon tracker
#if CARBON_TRACKING:
#  tracker.stop()  # writes csv file to directory specified during initialisation. Does not overwrite csv, but append new runs

print("Script done.")


