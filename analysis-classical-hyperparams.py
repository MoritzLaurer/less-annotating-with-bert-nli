
#### This script conducts the hyperparameter search for classical algorithms
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
import random
import os
import tqdm
from collections import OrderedDict
import joblib
from datetime import date
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import svm, naive_bayes, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

import spacy


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
parser.add_argument('-t', '--n_trials', type=int,
                    help='How many optuna trials should be run?')
parser.add_argument('-ts', '--n_trials_sampling', type=int,
                    help='After how many trials should optuna start sampling?')
parser.add_argument('-tp', '--n_trials_pruning', type=int,
                    help='After how many trials should optuna start pruning?')
parser.add_argument('-cvh', '--n_cross_val_hyperparam', type=int, default=2,
                    help='How many times should optuna cross validate in a single trial?')
parser.add_argument('-context', '--context', action='store_true',
                    help='Take surrounding context sentences into account.')

# arguments for both hyperparam and test script
parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality"')
parser.add_argument('-samp', '--sample_interval', type=int, nargs='+',
                    help='Interval of sample sizes to test.')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "classical_ml"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings"')
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
  # parse args if not in terminal, but in script
  args = parser.parse_args(["--n_trials", "70", "--n_trials_sampling", "30", "--n_trials_pruning", "40", "--n_cross_val_hyperparam", "2",
                            "--context", "--dataset", "manifesto-military", "--sample_interval", "500",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "SVM", "--vectorizer", "embeddings",
                            "--n_cross_val_final", "3", "--hyperparam_study_date", "20220713"])


### args only for hyperparameter tuning
N_TRIALS = args.n_trials
N_STARTUP_TRIALS_SAMPLING = args.n_trials_sampling
N_STARTUP_TRIALS_PRUNING = args.n_trials_pruning
CROSS_VALIDATION_REPETITIONS_HYPERPARAM = args.n_cross_val_hyperparam
CONTEXT = args.context   # not in use. would need to adapt code below. Currently always includes context & not-context in hyperparameter search. seems like best option.

### args for both hyperparameter tuning and test runs
# choose dataset
DATASET_NAME = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "nsp", "classical_ml"
MODEL_NAME = args.model  # "SVM"

DISABLE_TQDM = args.disable_tqdm
CARBON_TRACKING = args.carbon_tracking

### args only for test runs
HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final

# 10k max_iter had 1-10% performance drop for high n sample (possibly overfitting to majority class)
MAX_ITER_LOW, MAX_ITER_HIGH = 1_000, 7_000  # tried 10k, but led to worse performance on larger, imbalanced dataset (?)



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

## special preparation of manifesto simple
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
import sys
sys.path.insert(0, os.getcwd())

import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)

from helpers import data_preparation, compute_metrics_classical_ml, clean_memory

### load suitable hypotheses_hyperparameters and text formatting function
from hypothesis_hyperparams import hypothesis_hyperparams


### load the hypothesis hyperparameters for the respective dataset. For classical_ml this only determines the input text format - sentence with surrounding sentences, or not
hypothesis_hyperparams_dic, format_text = hypothesis_hyperparams(dataset_name=DATASET_NAME, df_cl=df_cl)

# check which template fits to standard_dl/classical_ml or NLI
print("")
print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template])
print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" not in template])

# in case -context flag is passed, but dataset actually does not have context
classical_templates = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template]
if "context" not in classical_templates:
  CONTEXT = False




##### prepare texts for classical ML
nlp = spacy.load("en_core_web_lg")
np.random.seed(SEED_GLOBAL)

## lemmatize text
def lemmatize(text_lst, embeddings=False):
  texts_lemma = []
  texts_vector = []
  if embeddings==False:
    for doc in nlp.pipe(text_lst, n_process=4):  #  disable=["tok2vec", "ner"] "tagger", "attribute_ruler", "parser",
      doc_lemmas = " ".join([token.lemma_ for token in doc])
      texts_lemma.append(doc_lemmas)
    return texts_lemma
  # test: only include specific parts-of-speech and output word vectors
  elif embeddings==True:
    for doc in nlp.pipe(text_lst, n_process=4):  #  disable=["parser", "ner"] "tagger", "attribute_ruler", "tok2vec",
      # only use specific parts-of-speech
      doc_lemmas = " ".join([token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB", "PROPN", "ADV", "INTJ", "PRON"]])  # difficult choice what to include https://spacy.io/usage/linguistic-features#pos-tagging
      # testing word vector representations instead of TF-IDF
      doc_vector = np.mean([token.vector for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB", "PROPN", "ADV", "INTJ", "PRON"]], axis=0)
      if str(doc_vector) == "nan":  # in case none of the pos tags are in text
          doc_vector = np.mean([token.vector for token in doc], axis=0)
      texts_lemma.append(doc_lemmas)
      texts_vector.append(doc_vector)
    return texts_vector
  else: 
    raise Exception(f"pos_selection not properly specified: {embeddings}")


df_train_lemma = df_train.copy(deep=True)
df_test_lemma = df_test.copy(deep=True)


if VECTORIZER == "tfidf":
    embeddings = False
elif VECTORIZER == "embeddings":
    embeddings = True

## lemmatize depending on dataset: some have only one text column ("text"), some have three (original, preceding, following text)
if "text_original" in df_cl.columns:
  #df_cl_lemma["text"] = lemmatize(df_cl.text_original)
  df_train_lemma["text_original"] = lemmatize(df_train.text_original, embeddings=embeddings)
  df_test_lemma["text_original"] = lemmatize(df_test.text_original, embeddings=embeddings)
else:
  #df_cl_lemma["text"] = lemmatize(df_cl.text)
  df_train_lemma["text"] = lemmatize(df_train.text, embeddings=embeddings)
  df_test_lemma["text"] = lemmatize(df_test.text, embeddings=embeddings)

if "text_preceding" in df_cl.columns:
  #df_cl_lemma["text_preceding"] = lemmatize(df_cl.text_preceding.fillna(""))
  df_train_lemma["text_preceding"] = lemmatize(df_train.text_preceding.fillna(""), embeddings=embeddings)
  df_test_lemma["text_preceding"] = lemmatize(df_test.text_preceding.fillna(""), embeddings=embeddings)
  # if surrounding text was nan, insert vector of original text to avoid nan error
  if embeddings == True:
      df_train_lemma["text_preceding"] = [text_original if str(text_surrounding) == "nan" else text_surrounding for text_surrounding, text_original in
                                          zip(df_train_lemma["text_preceding"], df_train_lemma["text_original"])]
      df_test_lemma["text_preceding"] = [text_original if str(text_surrounding) == "nan" else text_surrounding for text_surrounding, text_original in
                                          zip(df_test_lemma["text_preceding"], df_test_lemma["text_original"])]
if "text_following" in df_cl.columns:
  #df_cl_lemma["text_following"] = lemmatize(df_cl.text_following.fillna(""))
  df_train_lemma["text_following"] = lemmatize(df_train.text_following.fillna(""), embeddings=embeddings)
  df_test_lemma["text_following"] = lemmatize(df_test.text_following.fillna(""), embeddings=embeddings)
  # if surrounding text was nan, insert vector of original text to avoid nan error
  if embeddings == True:
      df_train_lemma["text_following"] = [text_original if str(text_surrounding) == "nan" else text_surrounding for text_surrounding, text_original in
                                          zip(df_train_lemma["text_following"], df_train_lemma["text_original"])]
      df_test_lemma["text_following"] = [text_original if str(text_surrounding) == "nan" else text_surrounding for text_surrounding, text_original in
                                          zip(df_test_lemma["text_following"], df_test_lemma["text_original"])]



##### hyperparameter tuning

# carbon tracker  https://github.com/mlco2/codecarbon/tree/master
#if CARBON_TRACKING:
#  from codecarbon import OfflineEmissionsTracker
#  tracker = OfflineEmissionsTracker(country_iso_code="NLD",  log_level='warning', measure_power_secs=300,  #output_dir=TRAINING_DIRECTORY
#                                    project_name=f"{DATASET_NAME}-{MODEL_NAME.split('/')[-1]}")
#  tracker.start()


def optuna_objective(trial, hypothesis_hyperparams_dic=None, n_sample=None, df_train=None, df=None):
  clean_memory()
  np.random.seed(SEED_GLOBAL)  # setting seed again for safety. not sure why this needs to be run here at each iteration. it should stay constant once set globally?! explanation could be this https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
  
  if VECTORIZER == "tfidf":
      # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
      hyperparams_vectorizer = {
          'ngram_range': trial.suggest_categorical("ngram_range", [(1, 2), (1, 3)]),
          'max_df': trial.suggest_categorical("max_df", [0.9, 0.8, 0.7]),
          'min_df': trial.suggest_categorical("min_df", [0.01, 0.03, 0.06])
      }
      vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', norm="l2", use_idf=True, smooth_idf=True, analyzer="word", **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02
  if VECTORIZER == "embeddings":
      hyperparams_vectorizer = {}

  # SVM  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  if MODEL_NAME == "SVM":
      hyperparams_clf = {'kernel': trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                   'C': trial.suggest_float("C", 1, 1000, log=True),
                   "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                   "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                   "coef0": trial.suggest_float("coef0", 1, 100, log=True),  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
                   "degree": trial.suggest_int("degree", 1, 50, log=False),  # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
                   #"decision_function_shape": trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"]),  # "However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary classification."
                   #"tol": trial.suggest_categorical("tol", [1e-3, 1e-4]),
                   "random_state": SEED_GLOBAL,
                   "max_iter": trial.suggest_int("num_train_epochs", MAX_ITER_LOW, MAX_ITER_HIGH, log=False, step=1000),  #MAX_ITER,
                   }
  # Logistic Regression # ! disadvantage: several parameters only work with certain other parameters  # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn-linear-model-logisticregression
  elif MODEL_NAME == "logistic":
      hyperparams_clf = {#'penalty': trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"]),
                        'penalty': 'l2',  # works with all solvers
                        'solver': trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
                        'C': trial.suggest_float("C", 1, 1000, log=False),
                        #"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                        #"intercept_scaling": trial.suggest_float("intercept_scaling", 1, 50, log=False),
                        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                        "max_iter": trial.suggest_int("max_iter", 50, 1000, log=False),  # 100 default
                        "multi_class": "auto",  # {‘auto’, ‘ovr’, ‘multinomial’}
                        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
                        #"l1_ratio": None,
                        "n_jobs": -1,
                        "random_state": SEED_GLOBAL,
                        }
  else:
      raise Exception("Method not available: ", MODEL_NAME)
  
  # not choosing a hypothesis template here, but the way of formatting the input text (e.g. with preceding sentence or not). need to keep same object names
  # if statements determine, whether surrounding sentences are added, or not. Disactivated, because best to always try and test context
  #if CONTEXT == True:
  #  text_template_classical_ml = [template for template in list(hypothesis_hyperparams_dic.keys()) if ("not_nli" in template) and ("context" in template)]
  #elif CONTEXT == False:
  text_template_classical_ml = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template]   #and ("context" in template)
  #else:
  #  raise Exception(f"CONTEXT variable is {CONTEXT}. Can only be True/False")

  if len(text_template_classical_ml) >= 2:  # if there is only one reasonable text format for standard_dl
    hypothesis_template = trial.suggest_categorical("hypothesis_template", text_template_classical_ml)
  else:
    hypothesis_template = text_template_classical_ml[0]

  hyperparams_optuna = {**hyperparams_clf, **hyperparams_vectorizer, "hypothesis_template": hypothesis_template}
  trial.set_user_attr("hyperparameters_all", hyperparams_optuna)
  print("Hyperparameters for this run: ", hyperparams_optuna)


  # cross-validation loop. Objective: determine F1 for specific sample for specific hyperparams, without a test set
  run_info_dic_lst = []
  for step_i, random_seed_cross_val in enumerate(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_HYPERPARAM)):
    np.random.seed(SEED_GLOBAL)
    df_train_samp, df_dev_samp = data_preparation(random_seed=random_seed_cross_val, df_train=df_train, df=df,
                                                  hypothesis_template=hypothesis_template, 
                                                  hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template], 
                                                  n_sample=n_sample, format_text_func=format_text, method=METHOD, embeddings=embeddings)
    
    clean_memory()

    if VECTORIZER == "tfidf":
        # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
        vectorizer.fit(pd.concat([df_train_samp.text_prepared, df_dev_samp.text_prepared]))
        X_train = vectorizer.transform(df_train_samp.text_prepared)
        X_test = vectorizer.transform(df_dev_samp.text_prepared)
    if VECTORIZER == "embeddings":
        X_train = np.array([list(lst) for lst in df_train_samp.text_prepared])
        X_test = np.array([list(lst) for lst in df_dev_samp.text_prepared])

    y_train = df_train_samp.label
    y_test = df_dev_samp.label

    # training on train set sample
    if MODEL_NAME == "SVM":
        clf = svm.SVC(**hyperparams_clf)
    elif MODEL_NAME == "logistic":
        clf = linear_model.LogisticRegression(**hyperparams_clf)
    clf.fit(X_train, y_train)

    # prediction on test set
    label_gold = y_test
    label_pred = clf.predict(X_test)
    
    # metrics
    metric_step = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df.label_text.unique()))

    run_info_dic = {"method": METHOD, "n_sample": n_sample, "model": MODEL_NAME, "results": metric_step, "hyper_params": hyperparams_optuna}
    run_info_dic_lst.append(run_info_dic)
    
    # Report intermediate objective value.
    intermediate_value = (metric_step["eval_f1_macro"] + metric_step["eval_f1_micro"]) / 2
    trial.report(intermediate_value, step_i)
    # Handle trial pruning based on the intermediate value.
    if trial.should_prune() and (CROSS_VALIDATION_REPETITIONS_HYPERPARAM > 1):
      raise optuna.TrialPruned()
    if n_sample == 999_999:  # no cross-validation necessary for full dataset
      break


  ## aggregation over cross-val loop
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

  print(f"\nFinal metrics for run: {metric_details}. With hyperparameters: {hyperparams_optuna}\n")

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

  study.optimize(lambda trial: optuna_objective(trial, hypothesis_hyperparams_dic=hypothesis_hyperparams_dic, n_sample=n_sample, df_train=df_train_lemma, df=df_cl),
                n_trials=N_TRIALS, show_progress_bar=True)  # Objective function with additional arguments https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
  return study


hp_study_dic = {}
for n_sample in tqdm.tqdm(N_SAMPLE_DEV):
  study = run_study(n_sample=n_sample)
  hp_study_dic_step = {f"hp_search_{METHOD}_{n_sample}": {"n_max_sample_class": n_sample, "method": METHOD, "dataset": DATASET_NAME, "algorithm": MODEL_NAME, "optuna_study": study} } 
  hp_study_dic.update(hp_study_dic_step)
  # save study_dic after each new study added
  while len(str(n_sample)) <= 4:
    n_sample = "0" + str(n_sample)
  if EXECUTION_TERMINAL == True:
      if VECTORIZER == "tfidf":
        joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")
      elif VECTORIZER == "embeddings":
        joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")
  elif EXECUTION_TERMINAL == False:
      if VECTORIZER == "tfidf":
        joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}_local_test.pkl")
      elif VECTORIZER == "embeddings":
        joblib.dump(hp_study_dic_step, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}_local_test.pkl")

## stop carbon tracker
#if CARBON_TRACKING:
#  tracker.stop()  # writes csv file to directory specified during initialisation. Does not overwrite csv, but append new runs

print("Run done.")

