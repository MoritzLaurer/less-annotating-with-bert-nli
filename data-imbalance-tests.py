#!/usr/bin/env python
# coding: utf-8

# ## Install and load packages
import pandas as pd
import numpy as np
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


# ## Data loading

#set wd
print(os.getcwd())
#os.chdir("./NLI-experiments")
print(os.getcwd())

metric = "f1_macro"

## save study  # https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-save-and-resume-studies
import joblib
from os import listdir
from os.path import isfile, join
os.getcwd()

DATASET_NAME_LST = ["sentiment-news-econ", "coronanet", "cap-sotu", "cap-us-court", "manifesto-8",
                    "manifesto-military", "manifesto-protectionism", "manifesto-morality"]
#DATASET_NAME_LST = ["sentiment-news-econ", "manifesto-morality"]

SAMPLE_SIZE = "00500"

def load_latest_experiment_dic(method_name="SVM", dataset_name=None, sample_size=None):
  # get latest experiment for each method for the respective dataset - experiments take a long time and many were conducted
  path_dataset = f"./results/{dataset_name}"
  file_names_lst = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]
  #print(dataset_name)
  #print(method_name)
  #print(file_names_lst)
  ## ! need to manually make sure that experiments for same dataset and method have same latest date
  #experiment_dates = [int(file_name.split("_")[-1].replace(".pkl", "")) for file_name in file_names_lst if method_name in file_name]
  experiment_dates = [int(file_name.split("_")[-1].replace(".pkl", "")) for file_name in file_names_lst if (method_name in file_name) and ("experiment" in file_name)]
  if len(experiment_dates) > 0:  # in case no experiment for method available yet
    latest_experiment_date = np.max(experiment_dates)
    # get only file names for latest experiment and respective method - ordered starting with smalles experiment
    file_names = np.sort([file_name for file_name in file_names_lst if all(x in file_name for x in [str(latest_experiment_date), "experiment", method_name])])
    file_names = [file_name for file_name in file_names if str(sample_size) in file_name]
    # create compile sample experiments into single dic
    experiment_dic = {}
    [experiment_dic.update(joblib.load(f"./results/{dataset_name}/{file_name}")) for file_name in file_names]
    return experiment_dic
  else: 
    return None


## load results
experiment_details_dic_all_methods_dataset = {}
for dataset_name in DATASET_NAME_LST:
  #dataset_name = "cap-us-court"

  experiment_details_dic_all_methods = {dataset_name: {}}
  for method_name in ["SVM", "xtremedistil-l6-h256-uncased", "xtremedistil-l6-h256-mnli-fever-anli-ling-binary",
                      "deberta-v3-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c"  #, "xtremedistil-l6-h256-mnli-fever-anli-ling-politicsnli"
                      ]:
    experiment_dic = load_latest_experiment_dic(method_name=method_name, dataset_name=dataset_name, sample_size=SAMPLE_SIZE)
    if experiment_dic != None:  # to catch cases where not experiment data for method available yet
      experiment_details_dic_all_methods[dataset_name].update({method_name: experiment_dic})

  experiment_details_dic_all_methods_dataset.update(experiment_details_dic_all_methods)


#####  calculate imbalance metrics
## objective
# see if standard methods overestimate majority classes and underestimate minority
# detailed disaggregation across 8 datasets does not make sense


label_share_dif_majority_dic = {"classical_ml": [], "standard_dl": [], "nli": []}
label_share_dif_minority_dic = {"classical_ml": [], "standard_dl": [], "nli": []}
for key_dataset in experiment_details_dic_all_methods_dataset:
    for key_method in experiment_details_dic_all_methods_dataset[key_dataset]:
        for key_sample, value_sample in experiment_details_dic_all_methods_dataset[key_dataset][key_method].items():
            # calculate mean over random runs
            # ...
            for key_sample_run in value_sample:
                if "metrics_seed" in key_sample_run:
                    print(key_dataset, key_method)
                    label_share_actual = pd.Series(value_sample[key_sample_run]['eval_label_gold_raw']).value_counts(normalize=True, dropna=False)
                    label_share_predicted = pd.Series(value_sample[key_sample_run]['eval_label_predicted_raw']).value_counts(normalize=True, dropna=False)
                    label_share_dif = label_share_predicted - label_share_actual  # negative number = underestimates class by that much; positive number = overestimates the class
                    label_share_dif_majority = label_share_dif.fillna(0)[0:1]
                    label_share_dif_minority = label_share_dif.fillna(0)[-1:]
                    #if any([label_share_dif_minority.isna().iloc[0], label_share_dif_majority.isna().iloc[0]]):
                    #    print(label_share_dif)
                    label_share_dif_majority_dic[value_sample["method"]].append(label_share_dif_majority.iloc[0])
                    label_share_dif_minority_dic[value_sample["method"]].append(label_share_dif_minority.iloc[0])
                    break

test = pd.DataFrame(label_share_dif_majority_dic)
print("Majority", test.mean())

test = pd.DataFrame(label_share_dif_minority_dic)
print("Minority", test.mean())




