#!/usr/bin/env python
# coding: utf-8

# ## Load packages
import pandas as pd
import numpy as np
import os
import joblib
import optuna
## optuna visualisation  https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
from optuna.visualization import plot_optimization_history, plot_contour, plot_slice, plot_param_importances

# set path
print(os.getcwd())
if "NLI-experiments" not in os.getcwd():
    os.chdir("./NLI-experiments")
print(os.getcwd())


##### Create Hyperparameter Tables
#### Transformers
# get all optuna study file paths
path_files_lst = [os.path.join(path, name) for path, subdirs, files in os.walk("results/") for name in files if any(string not in name for string in [".DS_Store", "experiment"])]
path_files_lst = [path_files for path_files in path_files_lst if ".DS_Store" not in path_files]
path_files_lst = [path_files for path_files in path_files_lst if "experiment" not in path_files]
# exclude/only specific algo?
path_files_lst = [path_files for path_files in path_files_lst if "logistic" not in path_files]
path_files_lst = [path_files for path_files in path_files_lst if "SVM" not in path_files]

# exclude hyperparameter dictionaries which are just copies from smaller sample runs to save compute
hyperparams_copies = [# no search after 2500
                      'results/sentiment-news-econ/optuna_study_deberta-v3-base_03000samp_20220428.pkl',
                      'results/sentiment-news-econ/optuna_study_DeBERTa-v3-base-mnli-fever-docnli-ling-2c_03000samp_20220429.pkl',
                      'results/cap-us-court/optuna_study_DeBERTa-v3-base-mnli-fever-docnli-ling-2c_05000samp_20220428.pkl',
                      'results/cap-us-court/optuna_study_deberta-v3-base_05000samp_20220428.pkl',
                      'results/cap-us-court/optuna_study_DeBERTa-v3-base-mnli-fever-docnli-ling-2c_10000samp_20220428.pkl',
                      'results/cap-us-court/optuna_study_deberta-v3-base_10000samp_20220428.pkl',
                      # no search after 5000
                      'results/cap-sotu/optuna_study_deberta-v3-base_10000samp_20220428.pkl',
                      'results/manifesto-8/optuna_study_deberta-v3-base_10000samp_20220428.pkl',
                      'results/coronanet/optuna_study_deberta-v3-base_10000samp_20220428.pkl',
                      'results/cap-sotu/optuna_study_DeBERTa-v3-base-mnli-fever-docnli-ling-2c_10000samp_20220428.pkl',
                      'results/manifesto-8/optuna_study_DeBERTa-v3-base-mnli-fever-docnli-ling-2c_10000samp_20220428.pkl',
                      'results/coronanet/optuna_study_DeBERTa-v3-base-mnli-fever-docnli-ling-2c_10000samp_20220428.pkl'
                      ]
path_files_lst = [path_files for path_files in path_files_lst if path_files not in hyperparams_copies]


# add path name as key again to distinguish between datasets
hp_study_dic = {}
for path_file in path_files_lst:
  hp_study_dic.update({path_file.replace("results/", ""): joblib.load(path_file)})

#### extract relevant information from optuna study object and overall dictionary
col_dataset = []
col_algo = []
col_method = []
col_sample = []
col_lr = []
col_epochs = []
col_batch = []
col_hypo_template = []
col_lr_warmup = []
col_lr_importance = []
col_epochs_importance = []
col_batch_importance = []
col_hypo_template_importance = []
col_lr_warmup_importance = []
col_f1_macro_mean = []
col_f1_micro_mean = []
col_f1_macro_std = []
col_f1_micro_std = []
for study_key, study_value in hp_study_dic.items():
    study_value = study_value[list(study_value.keys())[0]]
    col_dataset.append(study_value["dataset"])
    col_algo.append(study_value["algorithm"])  # .split("/")[-1]
    col_method.append(study_value["method"])
    col_sample.append(study_value["n_max_sample_class"])
    col_lr.append(study_value["optuna_study"].best_params["learning_rate"])
    col_epochs.append(study_value["optuna_study"].best_params["num_train_epochs"])
    col_batch.append(study_value["optuna_study"].best_params["per_device_train_batch_size"])
    col_lr_importance.append(optuna.importance.get_param_importances(study_value['optuna_study'])["learning_rate"])
    col_epochs_importance.append(optuna.importance.get_param_importances(study_value['optuna_study'])["num_train_epochs"])
    col_batch_importance.append(optuna.importance.get_param_importances(study_value['optuna_study'])["per_device_train_batch_size"])
    col_f1_macro_mean.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_macro_mean"])
    col_f1_micro_mean.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_micro_mean"])
    col_f1_macro_std.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_macro_std"])
    col_f1_micro_std.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_micro_std"])
    if "hypothesis_template" in study_value["optuna_study"].best_params.keys():
        col_hypo_template.append(study_value["optuna_study"].best_params["hypothesis_template"])
        col_hypo_template_importance.append(optuna.importance.get_param_importances(study_value['optuna_study'])["hypothesis_template"])
    else:
        col_hypo_template.append(np.nan)
        col_hypo_template_importance.append(np.nan)
    if "warmup_ratio" in study_value["optuna_study"].best_params.keys():
        col_lr_warmup.append(study_value["optuna_study"].best_params["warmup_ratio"])
        col_lr_warmup_importance.append(optuna.importance.get_param_importances(study_value['optuna_study'])["warmup_ratio"])
    else:
        col_lr_warmup.append(np.nan)
        col_lr_warmup_importance.append(np.nan)

    #if "warmup_ratio" in optuna.importance.get_param_importances(study_value['optuna_study']).keys():
    #    col_lr_warmup_importance.append(optuna.importance.get_param_importances(study_value['optuna_study'])["warmup_ratio"])
    #else:
    #    col_lr_warmup_importance.append(np.nan)

df_hp = pd.DataFrame(data={"algorithm": col_algo, "dataset": col_dataset, "method": col_method, "sample": col_sample,
                           "learning_rate": col_lr, "epochs": col_epochs, "batch_size": col_batch, "hypothesis": col_hypo_template,
                           "lr_warmup_ratio": col_lr_warmup,
                           "learning_rate_importance": col_lr_importance, "epochs_importance": col_epochs_importance,
                           "batch_size_importance": col_batch_importance, "hypothesis_importance": col_hypo_template_importance,
                           "lr_warmup_ratio_importance": col_lr_warmup_importance,
                           "f1_macro_mean": col_f1_macro_mean, "f1_micro_mean": col_f1_micro_mean,
                           "f1_macro_std": col_f1_macro_std, "f1_micro_std": col_f1_micro_std,
                           })
df_hp = df_hp.sort_values(["algorithm", "sample"], ascending=False)
df_hp = df_hp.drop(columns=["method", "f1_macro_mean", "f1_micro_mean", "f1_macro_std", "f1_micro_std"])
df_hp = df_hp.rename(columns={"hypothesis": "hypothesis/context"})
simple_algo_names_dic = {"SVM": "SVM", "microsoft/deberta-v3-base": "BERT-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli", "logistic": "logistic regression"}
df_hp.algorithm = df_hp.algorithm.map(simple_algo_names_dic)
df_hp.learning_rate_importance = df_hp.learning_rate_importance.round(2)
df_hp.epochs_importance = df_hp.epochs_importance.round(2)
df_hp.batch_size_importance = df_hp.batch_size_importance.round(2)
df_hp.hypothesis_importance = df_hp.hypothesis_importance.round(2)
df_hp.lr_warmup_ratio_importance = df_hp.lr_warmup_ratio_importance.round(2)

print(df_hp)

## write hyperparameters to disk
os.getcwd()
#df_hp.to_excel("df_hp.xlsx")



#### SVM
# get all optuna study file paths
path_files_lst = [os.path.join(path, name) for path, subdirs, files in os.walk("results/") for name in files if any(string not in name for string in [".DS_Store", "experiment"])]
path_files_lst = [path_files for path_files in path_files_lst if ".DS_Store" not in path_files]
path_files_lst = [path_files for path_files in path_files_lst if "experiment" not in path_files]
# exclude/only specific algo?
path_files_lst = [path_files for path_files in path_files_lst if "SVM" in path_files]

# add path name as key again to distinguish between datasets
hp_study_dic = {}
for path_file in path_files_lst:
  hp_study_dic.update({path_file.replace("results/", ""): joblib.load(path_file)})

#### extract relevant information from optuna study object and overall dictionary
## Transformers
col_dataset = []
col_algo = []
col_method = []
col_sample = []
col_ngram = []
col_max_df = []
col_min_df = []
col_kernel = []
col_c = []
col_gamma = []
col_class_weight = []
col_coef0 = []
col_degree = []
col_epochs = []
col_hypo_template = []
col_f1_macro_mean = []
col_f1_micro_mean = []
col_f1_macro_std = []
col_f1_micro_std = []
for study_key, study_value in hp_study_dic.items():
    study_value = study_value[list(study_value.keys())[0]]
    col_dataset.append(study_value["dataset"])
    col_algo.append(study_value["algorithm"])  # .split("/")[-1]
    col_method.append(study_value["method"])
    col_sample.append(study_value["n_max_sample_class"])
    col_ngram.append(study_value["optuna_study"].best_params["ngram_range"])
    col_max_df.append(study_value["optuna_study"].best_params["max_df"])
    col_min_df.append(study_value["optuna_study"].best_params["min_df"])
    col_kernel.append(study_value["optuna_study"].best_params["kernel"])
    col_c.append(study_value["optuna_study"].best_params["C"])
    col_gamma.append(study_value["optuna_study"].best_params["gamma"])
    col_class_weight.append(study_value["optuna_study"].best_params["class_weight"])
    col_coef0.append(study_value["optuna_study"].best_params["coef0"])
    col_degree.append(study_value["optuna_study"].best_params["degree"])
    col_epochs.append(study_value["optuna_study"].best_params["num_train_epochs"])
    col_f1_macro_mean.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_macro_mean"])
    col_f1_micro_mean.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_micro_mean"])
    col_f1_macro_std.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_macro_std"])
    col_f1_micro_std.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_micro_std"])
    if "hypothesis_template" in study_value["optuna_study"].best_params.keys():
        col_hypo_template.append(study_value["optuna_study"].best_params["hypothesis_template"])
    else:
        col_hypo_template.append(np.nan)

df_hp_svm = pd.DataFrame(data={"algorithm": col_algo, "dataset": col_dataset, "method": col_method, "sample": col_sample,
                            "ngram": col_ngram, "max_df": col_max_df, "min_df": col_min_df, "kernel": col_kernel, "C": col_c,
                            "gamma": col_gamma, "class_weight": col_class_weight, "coef0": col_coef0, "degree": col_degree, "epochs": col_epochs,
                           "context": col_hypo_template,
                           "f1_macro_mean": col_f1_macro_mean, "f1_micro_mean": col_f1_micro_mean,
                           "f1_macro_std": col_f1_macro_std, "f1_micro_std": col_f1_micro_std,
                           })
df_hp_svm = df_hp_svm.sort_values(["algorithm", "sample"], ascending=False)
simple_algo_names_dic = {"SVM": "SVM", "microsoft/deberta-v3-base": "BERT-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli", "logistic": "logistic regression"}
df_hp_svm.algorithm = df_hp_svm.algorithm.map(simple_algo_names_dic)
df_hp_svm = df_hp_svm.drop(columns=["method", "f1_macro_mean", "f1_micro_mean", "f1_macro_std", "f1_micro_std", "algorithm"])
df_hp_svm = df_hp_svm.rename(columns={"hypothesis": "hypothesis/context"})
df_hp_svm.C = df_hp_svm.C.round(2)
df_hp_svm.coef0 = df_hp_svm.coef0.round(2)
df_hp_svm.context = ["yes" if string == "template_not_nli_context" else np.nan if pd.isna(string) else "no" for string in df_hp_svm.context]

print(df_hp_svm)



#### Logistic Regression
# get all optuna study file paths
path_files_lst = [os.path.join(path, name) for path, subdirs, files in os.walk("results/") for name in files if any(string not in name for string in [".DS_Store", "experiment"])]
path_files_lst = [path_files for path_files in path_files_lst if ".DS_Store" not in path_files]
path_files_lst = [path_files for path_files in path_files_lst if "experiment" not in path_files]
# exclude/only specific algo?
path_files_lst = [path_files for path_files in path_files_lst if "logistic" in path_files]

# add path name as key again to distinguish between datasets
hp_study_dic = {}
for path_file in path_files_lst:
  hp_study_dic.update({path_file.replace("results/", ""): joblib.load(path_file)})

#### extract relevant information from optuna study object and overall dictionary
## Transformers
col_dataset = []
col_algo = []
col_method = []
col_sample = []
col_ngram = []
col_max_df = []
col_min_df = []
col_solver = []
col_c = []
col_class_weight = []
col_max_iter = []
col_warm_start = []
col_hypo_template = []
col_f1_macro_mean = []
col_f1_micro_mean = []
col_f1_macro_std = []
col_f1_micro_std = []
for study_key, study_value in hp_study_dic.items():
    study_value = study_value[list(study_value.keys())[0]]
    col_dataset.append(study_value["dataset"])
    col_algo.append(study_value["algorithm"])  # .split("/")[-1]
    col_method.append(study_value["method"])
    col_sample.append(study_value["n_max_sample_class"])
    col_ngram.append(study_value["optuna_study"].best_params["ngram_range"])
    col_max_df.append(study_value["optuna_study"].best_params["max_df"])
    col_min_df.append(study_value["optuna_study"].best_params["min_df"])
    col_solver.append(study_value["optuna_study"].best_params["solver"])
    col_c.append(study_value["optuna_study"].best_params["C"])
    col_class_weight.append(study_value["optuna_study"].best_params["class_weight"])
    col_max_iter.append(study_value["optuna_study"].best_params["max_iter"])
    col_warm_start.append(study_value["optuna_study"].best_params["warm_start"])
    col_f1_macro_mean.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_macro_mean"])
    col_f1_micro_mean.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_micro_mean"])
    col_f1_macro_std.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_macro_std"])
    col_f1_micro_std.append(study_value["optuna_study"].best_trial.user_attrs["metric_details"]["F1_micro_std"])
    if "hypothesis_template" in study_value["optuna_study"].best_params.keys():
        col_hypo_template.append(study_value["optuna_study"].best_params["hypothesis_template"])
    else:
        col_hypo_template.append(np.nan)

df_hp_lr = pd.DataFrame(data={"algorithm": col_algo, "dataset": col_dataset, "method": col_method, "sample": col_sample,
                            "ngram": col_ngram, "max_df": col_max_df, "min_df": col_min_df, "solver": col_solver,
                            "C": col_c, "class_weight": col_class_weight, "max_iter": col_max_iter, "warm_start": col_warm_start,
                           "context": col_hypo_template,
                           "f1_macro_mean": col_f1_macro_mean, "f1_micro_mean": col_f1_micro_mean,
                           "f1_macro_std": col_f1_macro_std, "f1_micro_std": col_f1_micro_std,
                           })
df_hp_lr = df_hp_lr.sort_values(["algorithm", "sample"], ascending=False)
simple_algo_names_dic = {"SVM": "SVM", "microsoft/deberta-v3-base": "BERT-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli", "logistic": "logistic regression"}
df_hp_lr.algorithm = df_hp_lr.algorithm.map(simple_algo_names_dic)
df_hp_lr = df_hp_lr.drop(columns=["method", "f1_macro_mean", "f1_micro_mean", "f1_macro_std", "f1_micro_std", "algorithm"])
df_hp_lr.context = ["yes" if string == "template_not_nli_context" else np.nan if pd.isna(string) else "no" for string in df_hp_lr.context]
df_hp_lr.C = df_hp_lr.C.round(2)

print(df_hp_lr)





##### compare different runs

dataset_to_inspect = "manifesto-protectionism"  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
sample_to_inspect = "2500"  #["100", "500", "1000", "2500"]

## optimisation history
for key_dataset_algo, value_dataset_algo in hp_study_dic.items():
    #if sample_to_inspect in key_dataset_algo:
        if dataset_to_inspect in key_dataset_algo:
            for study_key, study_value in value_dataset_algo.items():
              print(f"Study {study_key}:")
              plot_optimization_history(study_value["optuna_study"]).show()

## good plot to see if hyperparameter ranges are adequate
for key_dataset_algo, value_dataset_algo in hp_study_dic.items():
    #if sample_to_inspect in key_dataset_algo:
        if dataset_to_inspect in key_dataset_algo:
            for study_key, study_value in value_dataset_algo.items():
                print(f"Study {study_key}:")
                plot_slice(study_value["optuna_study"]).show()

## old notes on hp-search 100 & 500 with BERT variants:
# sentiment-news-econ: hps good, 15 runs adequate
# coronanet: standard_dl: could reduce num_train_epochs to 20 (especially for bigger samples); nli: num_train_epochs could be higher than 11, batch size could try higher than 32
# cap-us-court: standard_dl: could reduce num_train_epochs to 20 (especially for bigger samples, maybe even 10); nli: num_train_epochs should be higher than 11, batch size could try higher than 32
# cap-sotu: standard_dl: fine (for 100_samp, epochs could be higher than 100);  nli: seems good
# manifesto-8: standard_dl:  seems fine  ; nli:  seems fine (maybe slightly higher epochs for low n_sample
# manifesto-military:  standard_dl:  good  ; nli:  good
# manifesto-protectionism:  standard_dl:  good  ; nli:  good (epochs could be higher for 100_samp)
# manifesto-morality:  standard_dl:  x  ; nli:  x

## old notes for SVM: 100, 500, 1000, 2500, 5000, 10000
# sentiment-news-econ: good. hardly improvements after 25
# coronanet: good. hardly improvements after 30
# cap-us-court: good. hardly improvements after 15-30
# cap-sotu: good. hardly improvements after 20 - 50
# manifesto-8: good. hardly improvements after 20  (one small improvement at 60)
# manifesto-military: good. no improvements after 30
# manifesto-protectionism: good. hardly improvements after 25 (two small improvements at 40 & 50)
# manifesto-morality: good. hardly improvements after 30 (two small improvements at 40)
# => seems reasonable to reduce search to max 50 with sampling at 30 in the future

for key_dataset_algo, value_dataset_algo in hp_study_dic.items():
    for study_key, study_value in value_dataset_algo.items():
        try:
            print(f"Study {study_key}: {optuna.importance.get_param_importances(study_value['optuna_study'])}")
        except Exception as e:
            print(e)

#optuna.visualization.plot_param_importances(
#    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
#)

#plot_contour(study)


## best run details
for key_dataset_algo, value_dataset_algo in hp_study_dic.items():
    for study_key, study_value in value_dataset_algo.items():
      print(f"Study {study_key}:")
      print("Best hyperparameters: ", study_value["optuna_study"].best_params)
      print("Best hyperparameters all: ", study_value["optuna_study"].best_trial.user_attrs["hyperparameters_all"])
      print("Best performance: ", study_value["optuna_study"].best_value)
      print("Best performance details: ", study_value["optuna_study"].best_trial.user_attrs["metric_details"])
      #print("Best trial full info: ", study_value["optuna_study"].best_trial, "\n")
      #print(study.trials)


