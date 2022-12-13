
#### This script executes the training and testing for transformer algorithms
## the script is executed via the batch files in the batch-scripts folder

EXECUTION_TERMINAL = True

# ## Load packages
import transformers
import datasets
import torch

import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
import random
import os
import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from datetime import date
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
parser.add_argument('-zeroshot', '--zeroshot', action='store_true',
                    help='Start training run with a zero-shot run')



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
                            "--context", "--dataset", "cap-sotu", "--sample_interval", "1000", "2500", #"100", "500", "1000", "2500", "5000", "10000", 
                            "--method", "nli", "--model", "MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary", #"microsoft/xtremedistil-l6-h256-uncased", #"MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary", 
                            "--n_cross_val_final", "2", "--hyperparam_study_date", "20220418", "--carbon_tracking"
                            ])


### args for both hyperparameter tuning and test runs
# choose dataset
DATASET_NAME = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality",
N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "nsp"
MODEL_NAME = args.model

DISABLE_TQDM = args.disable_tqdm
CARBON_TRACKING = args.carbon_tracking

### args only for test runs
HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final
ZEROSHOT = args.zeroshot



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

print(DATASET_NAME)


## reduce max sample size interval list to fit to max df_train length
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len(df_train)]
if len(df_train) < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len(df_train)]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)

"""
# tests for code above
N_SAMPLE_DEV = [5000, 10000]
len_df_train = 5555
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len_df_train]
if len_df_train < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len_df_train]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)"""


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




# ## Final test with best hyperparameters
np.random.seed(SEED_GLOBAL)

# carbon tracker  https://github.com/mlco2/codecarbon/tree/master
#if CARBON_TRACKING:
#  from codecarbon import OfflineEmissionsTracker
#  tracker = OfflineEmissionsTracker(country_iso_code="NLD", log_level='warning', measure_power_secs=300,  # output_dir=TRAINING_DIRECTORY,
#                                    project_name=f"{DATASET_NAME}-{MODEL_NAME.split('/')[-1]}")
#  tracker.start()

### parameters for final tests with best hyperparams
## load existing studies with hyperparams
hp_study_dic = {}
for n_sample in N_SAMPLE_DEV:
  if N_SAMPLE_DEV != [0]:
    while len(str(n_sample)) <= 4:
      n_sample = "0" + str(n_sample)
    hp_study_dic_step = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")
    #hp_study_dic_step = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'.split('/')[-1]}_{n_sample}samp_20220220.pkl")
    hp_study_dic.update(hp_study_dic_step)
  else:
    break


# add hyperparameters (particularly hypo list for NLI) for 0-shot run, because 0-shot has no hyperparameters otherwise
if (ZEROSHOT == True) and (N_SAMPLE_DEV != [0]):
  N_SAMPLE_TEST = [0] + N_SAMPLE_DEV  # [0] for zero-shot
  print(N_SAMPLE_TEST)

  HYPER_PARAMS_LST = [study_value['optuna_study'].best_trial.user_attrs["hyperparameters_all"] for study_key, study_value in hp_study_dic.items()]
  HYPOTHESIS_TEMPLATE_LST = [hyperparams_dic["hypothesis_template"] for hyperparams_dic in HYPER_PARAMS_LST]
  HYPOTHESIS_TEMPLATE_LST = [HYPOTHESIS_TEMPLATE_LST[0]] + HYPOTHESIS_TEMPLATE_LST  # zero-shot gets same hypo template as first study run (study with smaller n_samples is closest to zero-shot) - could not tune on 0-shot because 0-shot assumed no dev set
  print(HYPOTHESIS_TEMPLATE_LST)

  HYPER_PARAMS_LST = [{key: dic[key] for key in dic if key!="hypothesis_template"} for dic in HYPER_PARAMS_LST]  # return dic with all elements, except hypothesis template
  HYPER_PARAMS_LST_TEST = [HYPER_PARAMS_LST[0]] + HYPER_PARAMS_LST  # add random hyperparams for 0-shot run (will not be used anyways)
  print(HYPER_PARAMS_LST_TEST)

elif (ZEROSHOT == False) and (N_SAMPLE_DEV != [0]):
  N_SAMPLE_TEST = N_SAMPLE_DEV
  print(N_SAMPLE_TEST)

  HYPER_PARAMS_LST = [study_value['optuna_study'].best_trial.user_attrs["hyperparameters_all"] for study_key, study_value in hp_study_dic.items()]
  HYPOTHESIS_TEMPLATE_LST = [hyperparams_dic["hypothesis_template"] for hyperparams_dic in HYPER_PARAMS_LST]
  print(HYPOTHESIS_TEMPLATE_LST)

  HYPER_PARAMS_LST = [{key: dic[key] for key in dic if key!="hypothesis_template"} for dic in HYPER_PARAMS_LST]  # return dic with all elements, except hypothesis template
  HYPER_PARAMS_LST_TEST = HYPER_PARAMS_LST  # add random hyperparams for 0-shot run (will not be used anyways)
  print(HYPER_PARAMS_LST_TEST)

## for only 0-shot runs
elif N_SAMPLE_DEV == [0]:
  N_SAMPLE_TEST = [0]
  zeroshot_templates = {"sentiment-news-econ": "template_complex", "coronanet": "template_quote", "cap-sotu": "template_quote_context", "cap-us-court": "template_quote", "manifesto-8": "template_quote_context",  "manifesto-military": "template_quote_2", "manifesto-morality": "template_quote_2", "manifesto-protectionism": "template_quote_2"}
  HYPOTHESIS_TEMPLATE_LST = [zeroshot_templates[DATASET_NAME]] #["template_complex"]  # sentiment-news-econ: "template_complex", coronanet: "template_quote", cap-sotu: "template_quote_context", cap-us-court: "template_quote", manifesto-8: "template_quote_context",  manifesto-military: "template_quote_2", manifesto-morality: "template_quote_2", manifesto-protectionism: "template_quote_2"
  HYPER_PARAMS_LST_TEST = [{"per_device_eval_batch_size": 160}]


## if need to manually set hyperparameters
#N_SAMPLE_TEST = [500, 2000]
#HYPOTHESIS_TEMPLATE_LST = ["template_quote_context_long_hypo", "template_quote_context_long_hypo", "template_quote_context_long_hypo"]
#HYPER_PARAMS_LST_TEST = [{'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 2, 'seed': 42, 'per_device_train_batch_size': 8, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128},
#                         {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 2, 'seed': 42, 'per_device_train_batch_size': 8, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128},
#                         {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 2, 'seed': 42, 'per_device_train_batch_size': 8, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128}]

## for specific targeted runs
#N_SAMPLE_TEST = [N_SAMPLE_TEST[1]]
#HYPOTHESIS_TEMPLATE_LST = [HYPOTHESIS_TEMPLATE_LST[1]]
#HYPER_PARAMS_LST_TEST = [HYPER_PARAMS_LST_TEST[1]]
#print(HYPER_PARAMS_LST_TEST)





# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa" in MODEL_NAME: fp16_bool = False  # mDeBERTa does not support FP16 yet

### run random cross-validation for hyperparameter search without a dev set
np.random.seed(SEED_GLOBAL)

### K example intervals loop
experiment_details_dic = {}
for n_max_sample, hyperparams, hypothesis_template in tqdm.tqdm(zip(N_SAMPLE_TEST, HYPER_PARAMS_LST_TEST, HYPOTHESIS_TEMPLATE_LST), desc="Iterations for different number of texts", leave=True):
  np.random.seed(SEED_GLOBAL)

  # log how long training of model takes
  t_start = time.time()

  k_samples_experiment_dic = {"method": METHOD, "n_max_sample": n_max_sample, "model": MODEL_NAME, "hyperparams": hyperparams}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic
  f1_macro_lst = []
  f1_micro_lst = []
  # randomness stability loop. Objective: calculate F1 across N samples to test for influence of different (random) samples
  for random_seed_sample in tqdm.tqdm(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_FINAL), desc="Iterations for std", leave=True):
    # unrealistic oracle sample. not used
    #df_train_samp = df_train.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), n_max_sample), random_state=random_seed_sample))
    # fully random sample
    #df_train_samp = random_sample_fill(df_train=df_train, n_sample_per_class=n_max_sample, seed=random_seed_sample)
    if n_max_sample == 999_999:
      df_train_samp = df_train.copy(deep=True)
    else:
      df_train_samp = df_train.sample(n=min(n_max_sample, len(df_train)), random_state=random_seed_sample).copy(deep=True)
      # old code for filling up at least 3 examples for class
      #df_train_samp = random_sample_fill(df_train=df_train, n_sample_per_class=n_max_sample, random_seed=random_seed_sample, df=df_cl)
    print("Number of training examples after sampling: ", len(df_train_samp), " . (but before cross-validation split) ")

    if (n_max_sample == 0) and (METHOD == "standard_dl"): 
      f1_macro_lst.append(0)
      f1_micro_lst.append(0)
      metric_step = {'accuracy_balanced': 0, 'accuracy_not_b': 0, 'f1_macro': 0, 'f1_micro': 0}
      k_samples_experiment_dic.update({"n_train_total": len(df_train_samp), f"metrics_seed_{random_seed_sample}": metric_step})
      break

    # chose the text format depending on hyperparams
    df_train_samp = format_text(df=df_train_samp, text_format=hypothesis_template)
    df_test_formatted = format_text(df=df_test, text_format=hypothesis_template)

    # format train and dev set for NLI etc.
    if METHOD == "nli":
      df_train_samp = format_nli_trainset(df_train=df_train_samp, hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template], random_seed=SEED_GLOBAL)
      df_test_formatted = format_nli_testset(df_test=df_test_formatted, hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template])  # hypo_label_dic_short , hypo_label_dic_long

    clean_memory()
    model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    encoded_dataset = tokenize_datasets(df_train_samp=df_train_samp, df_test=df_test_formatted, tokenizer=tokenizer, method=METHOD, max_length=None)

    train_args = set_train_args(hyperparams_dic=hyperparams, training_directory=TRAINING_DIRECTORY, disable_tqdm=DISABLE_TQDM, evaluation_strategy="no", fp16=fp16_bool)  # seed=random_seed_sample ! can the order to data loading via seed (but then different from HP search)
    trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=encoded_dataset, train_args=train_args, 
                             method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    clean_memory()

    if n_max_sample != 0:
      trainer.train()
    results = trainer.evaluate()  # eval_dataset=encoded_dataset_test

    k_samples_experiment_dic.update({"n_train_total": len(df_train_samp), f"metrics_seed_{random_seed_sample}": results})

    ## calculate aggregate metrics across random runs
    f1_macro_lst.append(results["eval_f1_macro"])
    f1_micro_lst.append(results["eval_f1_micro"])

    if (n_max_sample == 0) or (n_max_sample == 999_999):  # only one inference necessary on same test set in case of zero-shot or full dataset
      break

  # timer 
  t_end = time.time()
  t_total = round(t_end - t_start, 2)
  t_total = t_total / CROSS_VALIDATION_REPETITIONS_FINAL  # average of all random seed runs

  ## calculate aggregate metrics across random runs
  f1_macro_mean = np.mean(f1_macro_lst)
  f1_micro_mean = np.mean(f1_micro_lst)
  f1_macro_std = np.std(f1_macro_lst)
  f1_micro_std = np.std(f1_micro_lst)
  # add aggregate metrics to overall experiment dict
  metrics_mean = {"f1_macro_mean": f1_macro_mean, "f1_micro_mean": f1_micro_mean, "f1_macro_std": f1_macro_std, "f1_micro_std": f1_micro_std}
  k_samples_experiment_dic.update({"metrics_mean": metrics_mean, "dataset": DATASET_NAME, "n_classes": len(LABEL_TEXT_ALPHABETICAL), "train_eval_time_per_model": t_total})

  # harmonise n_sample file title
  while len(str(n_max_sample)) <= 4:
    n_max_sample = "0" + str(n_max_sample)

  # update of overall experiment dic
  experiment_details_dic_step = {f"experiment_sample_{n_max_sample}_{METHOD}_{MODEL_NAME}": k_samples_experiment_dic}
  experiment_details_dic.update(experiment_details_dic_step)

  ## save experiment dic after each new study
  joblib.dump(experiment_details_dic_step, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{n_max_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")


## stop carbon tracker
#if CARBON_TRACKING:
#  tracker.stop()  # writes csv file to directory specified during initialisation. Does not overwrite csv, but append new runs



# ## Print Results
for experiment_key in experiment_details_dic:
  print(f"{experiment_key}: f1_macro: {experiment_details_dic[experiment_key]['metrics_mean']['f1_macro_mean']} , f1_micro: {experiment_details_dic[experiment_key]['metrics_mean']['f1_micro_mean']}")

print("Script done.")


