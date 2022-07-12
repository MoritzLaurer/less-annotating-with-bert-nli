
# optuna import issue
#no module named sqlalchemy.dialects.postgresql
# updating sqlalchemy from 1.4.35 to 1.4.39 latest
# new error: "AttributeError: partially initialized module 'optuna' has no attribute 'logging' (most likely due to a circular import)"

import pandas as pd
import numpy as np
import os
import joblib
import optuna
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
#from IPython.display import Markdown, display
#pd.set_option("display.notebook_repr_html", True)
#nb_setup.setup_pandas(escape_latex = False)
SEED_GLOBAL = 42

print(os.getcwd())
if os.getcwd() != "/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments":
    os.chdir("./NLI-experiments")

## !! fix necessary: remove multiple results from different dates from results folder (final best option), or make this appendix script handle multiple dates


##### 1. appendix

### Manifesto

## Manifesto-8 data distribution
df_cl_manifesto = pd.read_csv("./data_clean/df_manifesto_all.csv", index_col="idx")
df_train_manifesto = pd.read_csv("./data_clean/df_manifesto_train.csv", index_col="idx")
df_test_manifesto = pd.read_csv("./data_clean/df_manifesto_test.csv", index_col="idx")
df_train_test_distribution_manifesto = pd.DataFrame([df_train_manifesto.label_domain_text.value_counts().rename("train"), df_test_manifesto.label_domain_text.value_counts().rename("test"),
                                           df_cl_manifesto.label_domain_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_manifesto.index.name = "labels"
df_train_test_distribution_manifesto.to_csv("./appendix/data-distribution-manifesto-8.csv")

## Manifesto-military data distribution
df_cl_military = pd.read_csv("./data_clean/df_manifesto_military_cl.csv", index_col="idx")
df_train_military = pd.read_csv("./data_clean/df_manifesto_military_train.csv", index_col="idx")
df_test_military = pd.read_csv("./data_clean/df_manifesto_military_test.csv", index_col="idx")
df_train_test_distribution_military = pd.DataFrame([df_train_military.label_text.value_counts().rename("train"), df_test_military.label_text.value_counts().rename("test"),
                                                   df_cl_military.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_military.index.name = "labels"
df_train_test_distribution_military.to_csv("./appendix/data-distribution-manifesto-military.csv")

## Manifesto-protectionism data distribution
df_cl_protectionism = pd.read_csv("./data_clean/df_manifesto_protectionism_cl.csv", index_col="idx")
df_train_protectionism = pd.read_csv("./data_clean/df_manifesto_protectionism_train.csv", index_col="idx")
df_test_protectionism = pd.read_csv("./data_clean/df_manifesto_protectionism_test.csv", index_col="idx")
df_train_test_distribution_protectionism = pd.DataFrame([df_train_protectionism.label_text.value_counts().rename("train"), df_test_protectionism.label_text.value_counts().rename("test"),
                                                   df_cl_protectionism.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_protectionism.index.name = "labels"
df_train_test_distribution_protectionism.to_csv("./appendix/data-distribution-manifesto-protectionism.csv")

## Manifesto-morality data distribution
df_cl_morality = pd.read_csv("./data_clean/df_manifesto_morality_cl.csv", index_col="idx")
df_train_morality = pd.read_csv("./data_clean/df_manifesto_morality_train.csv", index_col="idx")
df_test_morality = pd.read_csv("./data_clean/df_manifesto_morality_test.csv", index_col="idx")
df_train_test_distribution_morality = pd.DataFrame([df_train_morality.label_text.value_counts().rename("train"), df_test_morality.label_text.value_counts().rename("test"),
                                                   df_cl_morality.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_morality.index.name = "labels"
df_train_test_distribution_morality.to_csv("./appendix/data-distribution-manifesto-morality.csv")


### Sentiment Economy
df_cl_senti = pd.read_csv("./data_clean/df_sentiment_news_econ_all.csv", index_col="idx")
df_train_senti = pd.read_csv("./data_clean/df_sentiment_news_econ_train.csv", index_col="idx")
df_test_senti = pd.read_csv("./data_clean/df_sentiment_news_econ_test.csv", index_col="idx")
df_train_test_distribution_senti = pd.DataFrame([df_train_senti.label_text.value_counts().rename("train"), df_test_senti.label_text.value_counts().rename("test"),
                                                 df_cl_senti.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_senti.index.name = "labels"
df_train_test_distribution_senti.to_csv("./appendix/data-distribution-sentiment.csv")


### Comparative Agendas Project (CAP) - US State of the Union Speeches (SotU)
df_cl_cap_sotu = pd.read_csv("./data_clean/df_cap_sotu_all.csv", index_col="idx")
df_train_cap_sotu = pd.read_csv("./data_clean/df_cap_sotu_train.csv", index_col="idx")
df_test_cap_sotu = pd.read_csv("./data_clean/df_cap_sotu_test.csv", index_col="idx")
df_train_test_distribution_cap_sotu = pd.DataFrame([df_train_cap_sotu.label_text.value_counts().rename("train"), df_test_cap_sotu.label_text.value_counts().rename("test"),
                                                 df_cl_cap_sotu.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_cap_sotu.index.name = "labels"
df_train_test_distribution_cap_sotu.to_csv("./appendix/data-distribution-cap-sotu.csv")


### Comparative Agendas Project (CAP) - US Court Cases
df_cl_cap_uscourt = pd.read_csv("./data_clean/df_cap_us_court_all.csv", index_col="idx")
df_train_cap_uscourt = pd.read_csv("./data_clean/df_cap_us_court_train.csv", index_col="idx")
df_test_cap_uscourt = pd.read_csv("./data_clean/df_cap_us_court_test.csv", index_col="idx")
df_train_test_distribution_cap_uscourt = pd.DataFrame([df_train_cap_uscourt.label_text.value_counts().rename("train"), df_test_cap_uscourt.label_text.value_counts().rename("test"),
                                                 df_cl_cap_uscourt.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_cap_uscourt.index.name = "labels"
df_train_test_distribution_cap_sotu.to_csv("./appendix/data-distribution-cap-us-court.csv")


### CoronaNet
df_cl_coronanet = pd.read_csv("./data_clean/df_coronanet_20220124_all.csv", index_col="idx")
df_train_coronanet = pd.read_csv("./data_clean/df_coronanet_20220124_train.csv", index_col="idx")
df_test_coronanet = pd.read_csv("./data_clean/df_coronanet_20220124_test.csv", index_col="idx")
df_train_test_distribution_coronanet = pd.DataFrame([df_train_coronanet.label_text.value_counts().rename("train"), df_test_coronanet.label_text.value_counts().rename("test"),
                                                     df_cl_coronanet.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_coronanet.index.name = "labels"
df_train_test_distribution_coronanet.to_csv("./appendix/data-distribution-coronanet.csv")





##### appendix 2. Hyperparameters and Pre-processing

### Best Hyperparameters for DeBERTa-base & DeBERTa-nli
# get all optuna study file paths
path_files_lst = [os.path.join(path, name) for path, subdirs, files in os.walk("results/") for name in files if any(string not in name for string in [".DS_Store", "experiment"])]
path_files_lst = [path_files for path_files in path_files_lst if ".DS_Store" not in path_files]
path_files_lst = [path_files for path_files in path_files_lst if "experiment" not in path_files]
# exclude/include specific algo
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

## extract relevant information from optuna study object and overall dictionary
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
simple_algo_names_dic = {"SVM": "SVM", "microsoft/deberta-v3-base": "DeBERTa-v3-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "DeBERTa-v3-nli", "logistic": "logistic regression"}
df_hp.algorithm = df_hp.algorithm.map(simple_algo_names_dic)
df_hp.learning_rate_importance = df_hp.learning_rate_importance.round(2)
df_hp.epochs_importance = df_hp.epochs_importance.round(2)
df_hp.batch_size_importance = df_hp.batch_size_importance.round(2)
df_hp.hypothesis_importance = df_hp.hypothesis_importance.round(2)
df_hp.lr_warmup_ratio_importance = df_hp.lr_warmup_ratio_importance.round(2)

df_hp_deberta_base = df_hp[df_hp.algorithm == "DeBERTa-v3-base"].drop(columns=["algorithm", "lr_warmup_ratio", "lr_warmup_ratio_importance"])
df_hp_deberta_nli = df_hp[df_hp.algorithm == "DeBERTa-v3-nli"].drop(columns=["algorithm"])

df_hp_deberta_base.to_csv("./appendix/hyperparams-deberta-base.csv")
df_hp_deberta_nli.to_csv("./appendix/hyperparams-deberta-nli.csv")



### Best Hyperparameters for SVM
# get all optuna study file paths
path_files_lst = [os.path.join(path, name) for path, subdirs, files in os.walk("results/") for name in files if any(string not in name for string in [".DS_Store", "experiment"])]
path_files_lst = [path_files for path_files in path_files_lst if ".DS_Store" not in path_files]
path_files_lst = [path_files for path_files in path_files_lst if "experiment" not in path_files]
# exclude/only specific algo?
path_files_lst = [path_files for path_files in path_files_lst if ("SVM_tfidf" in path_files) and ("20220712" not in path_files)]

# add path name as key again to distinguish between datasets
hp_study_dic = {}
for path_file in path_files_lst:
  hp_study_dic.update({path_file.replace("results/", ""): joblib.load(path_file)})

## extract relevant information from optuna study object and overall dictionary
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
#simple_algo_names_dic = {"SVM": "SVM", "microsoft/deberta-v3-base": "BERT-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli", "logistic": "logistic regression"}
simple_algo_names_dic = {"logistic_tfidf": "logistic_tfidf", "logistic_embeddings": "logistic_embeddings",
                         "SVM_tfidf": "SVM_tfidf", "SVM_embeddings": "SVM_embeddings",
                         "deberta-v3-base": "BERT-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli",
                         }
df_hp_svm.algorithm = df_hp_svm.algorithm.map(simple_algo_names_dic)
df_hp_svm = df_hp_svm.drop(columns=["method", "f1_macro_mean", "f1_micro_mean", "f1_macro_std", "f1_micro_std", "algorithm"])
df_hp_svm = df_hp_svm.rename(columns={"hypothesis": "hypothesis/context"})
df_hp_svm.C = df_hp_svm.C.round(2)
df_hp_svm.coef0 = df_hp_svm.coef0.round(2)
df_hp_svm.context = ["yes" if "context" in str(string) else np.nan if pd.isna(string) else "no" for string in df_hp_svm.context]

df_hp_svm.to_csv("./appendix/hyperparams-svm-tfidf.csv")



### Best Hyperparameters for Logistic Regression
# get all optuna study file paths
path_files_lst = [os.path.join(path, name) for path, subdirs, files in os.walk("results/") for name in files if any(string not in name for string in [".DS_Store", "experiment"])]
path_files_lst = [path_files for path_files in path_files_lst if ".DS_Store" not in path_files]
path_files_lst = [path_files for path_files in path_files_lst if "experiment" not in path_files]
# exclude/only specific algo?
path_files_lst = [path_files for path_files in path_files_lst if ("logistic_tfidf" in path_files) and ("20220712" not in path_files)]  # and ("20220712" in path_files)

# add path name as key again to distinguish between datasets
hp_study_dic = {}
for path_file in path_files_lst:
  hp_study_dic.update({path_file.replace("results/", ""): joblib.load(path_file)})

#### extract relevant information from optuna study object and overall dictionary
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
#simple_algo_names_dic = {"SVM": "SVM", "microsoft/deberta-v3-base": "BERT-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli", "logistic": "logistic regression"}
simple_algo_names_dic = {"logistic_tfidf": "logistic_tfidf", "logistic_embeddings": "logistic_embeddings",
                         "SVM_tfidf": "SVM_tfidf", "SVM_embeddings": "SVM_embeddings",
                         "deberta-v3-base": "BERT-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli",
                         }
df_hp_lr.algorithm = df_hp_lr.algorithm.map(simple_algo_names_dic)
df_hp_lr = df_hp_lr.drop(columns=["method", "f1_macro_mean", "f1_micro_mean", "f1_macro_std", "f1_micro_std", "algorithm"])
df_hp_lr.context = ["yes" if "context" in str(string) else np.nan if pd.isna(string) else "no" for string in df_hp_lr.context]
df_hp_lr.C = df_hp_lr.C.round(2)

df_hp_lr.to_csv("./appendix/hyperparams-logistic-tfidf.csv")





#### appendix 3 - Metrics per algorithm per sample size

### extract metrics
DATASET_NAME_LST = ["sentiment-news-econ", "coronanet", "cap-sotu", "cap-us-court", "manifesto-8",
                    "manifesto-military", "manifesto-protectionism", "manifesto-morality"]

def load_latest_experiment_dic(method_name="SVM_tfidf", dataset_name=None):
    # get latest experiment for each method for the respective dataset - experiments take a long time and many were conducted
    path_dataset = f"./results/{dataset_name}"
    file_names_lst = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]
    # print(dataset_name)
    # print(method_name)
    # print(file_names_lst)
    ## ! need to manually make sure that experiments for same dataset and method have same latest date
    # experiment_dates = [int(file_name.split("_")[-1].replace(".pkl", "")) for file_name in file_names_lst if method_name in file_name]
    experiment_dates = [int(file_name.split("_")[-1].replace(".pkl", "")) for file_name in file_names_lst if (method_name in file_name) and ("experiment" in file_name)]
    if method_name in ["SVM_tfidf", "logistic_tfidf"]:
        experiment_dates = [date for date in experiment_dates if date != 20220712]  # there seems to be a bug for this run
    if len(experiment_dates) > 0:  # in case no experiment for method available yet
        latest_experiment_date = np.max(experiment_dates)
        # get only file names for latest experiment and respective method - ordered starting with smalles experiment
        file_names = np.sort([file_name for file_name in file_names_lst if all(x in file_name for x in [str(latest_experiment_date), "experiment", method_name])])
        # create compile sample experiments into single dic
        experiment_dic = {}
        [experiment_dic.update(joblib.load(f"./results/{dataset_name}/{file_name}")) for file_name in file_names]
        return experiment_dic
    else:
        return None

## load results
experiment_details_dic_all_methods_dataset = {}
for dataset_name in DATASET_NAME_LST:
    # dataset_name = "cap-us-court"

    experiment_details_dic_all_methods = {dataset_name: {}}
    for method_name in ["logistic_tfidf", "SVM_tfidf", "logistic_embeddings", "SVM_embeddings",
                        "deberta-v3-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
                       ]:
        experiment_dic = load_latest_experiment_dic(method_name=method_name, dataset_name=dataset_name)
        if experiment_dic != None:  # to catch cases where not experiment data for method available yet
            experiment_details_dic_all_methods[dataset_name].update({method_name: experiment_dic})

    experiment_details_dic_all_methods_dataset.update(experiment_details_dic_all_methods)

## print experiment dics to check results
"""for key_dataset in experiment_details_dic_all_methods_dataset:
  print(key_dataset)
  for key_method in experiment_details_dic_all_methods_dataset[key_dataset]:
    print("  ", key_method)
  print("")"""

# experiment_details_dic_all_methods_dataset["manifesto-military"]["xtremedistil-l6-h256-uncased"]


## Data preparation
dataset_n_class_dic = {"sentiment-news-econ": 2, "coronanet": 20, "cap-sotu": 22, "cap-us-court": 20, "manifesto-8": 8, "manifesto-44": 44,
                       "manifesto-military": 3, "manifesto-protectionism": 3, "manifesto-morality": 3}

## iterate over all dataset experiment dics to extract metrics for viz
visual_data_dic_datasets = {}
for key_dataset_name, experiment_details_dic_all_methods in experiment_details_dic_all_methods_dataset.items():
    n_classes = dataset_n_class_dic[key_dataset_name]

    n_max_sample = []
    n_total_samples = []
    for key, value in experiment_details_dic_all_methods[list(experiment_details_dic_all_methods.keys())[0]].items():  # experiment_details_dic_all_methods["SVM"].items():
        n_max_sample.append(value["n_max_sample"])
        n_total_samples.append(value["n_train_total"])

    if n_max_sample[-1] < 10000:
        n_max_sample[-1] = f"{n_max_sample[-1]} (all)"
    x_axis_values = [f"{n_sample}" for n_sample in n_max_sample]
    # x_axis_values = ["0" if n_per_class == 0 else f"{str(n_total)} (all)" if n_per_class >= 1001 else f"{str(n_total)}"  for n_per_class, n_total in zip(n_sample_per_class, n_total_samples)]
    # if key_dataset_name == "cap-us-court":  # CAP-us-court reaches max dataset with 320 samples per class
    #  x_axis_values = ["0" if n_per_class == 0 else f"{str(n_total)} (all)" if n_per_class >= 320 else f"{str(n_total)}"  for n_per_class, n_total in zip(n_sample_per_class, n_total_samples)]

    ## specific data for indiv approaches
    visual_data_dic = {}
    for key_method in experiment_details_dic_all_methods:
        f1_macro_mean_lst = []
        f1_micro_mean_lst = []
        f1_macro_std_lst = []
        f1_micro_std_lst = []
        for key_step in experiment_details_dic_all_methods[key_method]:
            # experiment_details_dic_all_methods[key_method][key_step]["model"]  # method, model
            f1_macro_mean_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_macro_mean"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
            f1_micro_mean_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_micro_mean"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
            f1_macro_std_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_macro_std"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
            f1_micro_std_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_micro_std"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
        dic_method = {key_method: {"f1_macro_mean": f1_macro_mean_lst, "f1_micro_mean": f1_micro_mean_lst, "f1_macro_std": f1_macro_std_lst, "f1_micro_std": f1_micro_std_lst,
                                   "n_classes": n_classes, "x_axis_values": x_axis_values}}  # "n_max_sample": n_sample_per_class, "n_total_samples": n_total_samples,
        visual_data_dic.update(dic_method)

    visual_data_dic_datasets.update({key_dataset_name: visual_data_dic})




### Disaggregated metrics per dataset
import copy
visual_data_dic_datasets_cl = copy.deepcopy(visual_data_dic_datasets)

metrics_all_dic = {}
for key_dataset, value_dataset in visual_data_dic_datasets_cl.items():
    # delete unnecessary column
    [visual_data_dic_datasets_cl[key_dataset][key_method].pop("n_classes", None) for key_method, value_method in visual_data_dic_datasets_cl[key_dataset].items()]
    # round numbers
    for key_method, value_method in visual_data_dic_datasets_cl[key_dataset].items():
        for key_metric, value_metric in visual_data_dic_datasets_cl[key_dataset][key_method].items():
            if key_metric != "x_axis_values":
                visual_data_dic_datasets_cl[key_dataset][key_method][key_metric] = [round(value, 3) for value in value_metric]
    # build df
    df_metrics_all_dataset = pd.DataFrame(data=visual_data_dic_datasets_cl[key_dataset]).round(3)
    df_metrics_all_dataset = df_metrics_all_dataset.rename(index={'x_axis_values': 'n_sample'})
    df_metrics_all_dataset = df_metrics_all_dataset.reindex(["n_sample", "f1_macro_mean", "f1_micro_mean", "f1_macro_std", "f1_micro_std"])

    n_sample = df_metrics_all_dataset.loc["n_sample"][0]
    df_metrics_all_dataset = df_metrics_all_dataset.drop("n_sample")
    df_metrics_all_dataset = df_metrics_all_dataset.explode(list(df_metrics_all_dataset.columns))
    df_metrics_all_dataset["n_sample"] = 4 * n_sample
    df_metrics_all_dataset = df_metrics_all_dataset[['n_sample', 'logistic_tfidf', 'SVM_tfidf', 'logistic_embeddings', 'SVM_embeddings', 'deberta-v3-base', 'DeBERTa-v3-base-mnli-fever-docnli-ling-2c']]
    df_metrics_all_dataset = df_metrics_all_dataset.rename(columns={'DeBERTa-v3-base-mnli-fever-docnli-ling-2c': 'deberta-v3-nli'})
    metrics_all_dic.update({key_dataset: df_metrics_all_dataset})

for dataset in metrics_all_dic:
    metrics_all_dic[dataset].to_csv(f"./appendix/metrics_all_{dataset}.csv")





### Aggregate performance difference
simple_algo_names_dic = {"logistic_tfidf": "logistic_tfidf", "logistic_embeddings": "logistic_embeddings",
                         "SVM_tfidf": "SVM_tfidf", "SVM_embeddings": "SVM_embeddings",
                         "deberta-v3-base": "BERT-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli",
                         }

## extract metrics to create df comparing performance per dataset per algo
# ! careful: not all datasets have 2500 data points, so if it says 2500, this includes 2116 for protectionism (and less full samples for higher intervals)

df_metrics_lst = []
df_std_lst = []
for metric in ["f1_macro", "f1_micro"]:
    col_dataset = []
    col_algo = []
    col_f1_macro = []
    cols_metrics_dic = {"0 (8 datasets)": [], "100 (8 datasets)": [], "500 (8 datasets)": [], "1000 (8 datasets)": [], "2500 (8 datasets)": [], "5000 (4 datasets)": [],
                        "10000 (3 datasets)": []}
    cols_std_dic = {"0 (8 datasets)": [], "100 (8 datasets)": [], "500 (8 datasets)": [], "1000 (8 datasets)": [], "2500 (8 datasets)": [], "5000 (4 datasets)": [],
                        "10000 (3 datasets)": []}
    for key_dataset in visual_data_dic_datasets:
        #if key_dataset in datasets_selection:
          for key_algo in visual_data_dic_datasets[key_dataset]:
            col_dataset.append(key_dataset)
            col_algo.append(simple_algo_names_dic[key_algo])
            for i, k in enumerate(cols_metrics_dic.keys()):
                if len(visual_data_dic_datasets[key_dataset][key_algo][f"{metric}_mean"]) > i:
                    cols_metrics_dic[k].append(visual_data_dic_datasets[key_dataset][key_algo][f"{metric}_mean"][i])
                    cols_std_dic[k].append(visual_data_dic_datasets[key_dataset][key_algo][f"{metric}_std"][i])
                else:
                    cols_metrics_dic[k].append(np.nan)
                    cols_std_dic[k].append(np.nan)

    ## create aggregate metric dfs
    df_metrics = pd.DataFrame(data={"dataset": col_dataset, "algorithm": col_algo, **cols_metrics_dic})
    df_std = pd.DataFrame(data={"dataset": col_dataset, "algorithm": col_algo, **cols_std_dic})
    df_metrics_lst.append(df_metrics)
    df_std_lst.append(df_std)



## subset average metrics by dataset size
datasets_all = ["sentiment-news-econ", "cap-us-court", "manifesto-military", "manifesto-protectionism", "manifesto-morality", "coronanet", "cap-sotu", "manifesto-8"]
datasets_5000 = ["cap-us-court", "coronanet", "cap-sotu", "manifesto-8"]
datasets_10000 = ["coronanet", "cap-sotu", "manifesto-8"]

df_metrics_mean_dic = {}
for i, metric in enumerate(["f1_macro", "f1_micro"]):
    df_metrics_mean_all = df_metrics_lst[i][df_metrics_lst[i].dataset.isin(datasets_all)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["0 (8 datasets)", "100 (8 datasets)", "500 (8 datasets)", "1000 (8 datasets)", "2500 (8 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_metrics_mean_medium = df_metrics_lst[i][df_metrics_lst[i].dataset.isin(datasets_5000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["5000 (4 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_metrics_mean_large = df_metrics_lst[i][df_metrics_lst[i].dataset.isin(datasets_10000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["10000 (3 datasets)"]]
    #df_metrics_mean_all = df_metrics.groupby(by="algorithm", as_index=True).apply(np.mean).round(4)
    df_metrics_mean = pd.concat([df_metrics_mean_all, df_metrics_mean_medium, df_metrics_mean_large], axis=1)
    # add row with best classical algo value
    df_metrics_mean.loc["classical-best-tfidf"] = [max(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_tfidf"], df_metrics_mean.loc["logistic_tfidf"])]
    df_metrics_mean.loc["classical-best-embeddings"] = [max(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_embeddings"], df_metrics_mean.loc["logistic_embeddings"])]
    # order rows
    order_algos = ["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings", "classical-best-tfidf", "classical-best-embeddings", "BERT-base", "BERT-base-nli"]
    df_metrics_mean = df_metrics_mean.reindex(order_algos)
    df_metrics_mean.index.name = "Sample size / Algorithm"
    df_metrics_mean_dic.update({metric: df_metrics_mean.round(3)})


## difference in performance
df_metrics_difference_dic = {}
for i, metric in enumerate(["f1_macro", "f1_micro"]):
    df_metrics_difference = pd.DataFrame(data={
        #"BERT-base vs. SVM": df_metrics_mean_dic[metric].loc["BERT-base"] - df_metrics_mean_dic[metric].loc["SVM"],
        #"BERT-base vs. Log. Reg.": df_metrics_mean_dic[metric].loc["BERT-base"] - df_metrics_mean_dic[metric].loc["logistic regression"],
        "BERT-base vs. classical-best-tfidf": df_metrics_mean_dic[metric].loc["BERT-base"] - df_metrics_mean_dic[metric].loc["classical-best-tfidf"],
        "BERT-base vs. classical-best-embeddings": df_metrics_mean_dic[metric].loc["BERT-base"] - df_metrics_mean_dic[metric].loc["classical-best-embeddings"],
        #"BERT-base-nli vs. SVM": df_metrics_mean_dic[metric].loc["BERT-base-nli"] - df_metrics_mean_dic[metric].loc["SVM"],
        #"BERT-base-nli vs. Log. Reg.": df_metrics_mean_dic[metric].loc["BERT-base-nli"] - df_metrics_mean_dic[metric].loc["logistic regression"],
        "BERT-base-nli vs. classical-best-tfidf": df_metrics_mean_dic[metric].loc["BERT-base-nli"] - df_metrics_mean_dic[metric].loc["classical-best-tfidf"],
        "BERT-base-nli vs. classical-best-embeddings": df_metrics_mean_dic[metric].loc["BERT-base-nli"] - df_metrics_mean_dic[metric].loc["classical-best-embeddings"],
        "BERT-base-nli vs. BERT-base": df_metrics_mean_dic[metric].loc["BERT-base-nli"] - df_metrics_mean_dic[metric].loc["BERT-base"],
       #"Transformer-Mini-NLI vs. SVM": df_metrics_mean_all.loc["Transformer-Mini-NLI"] - df_metrics_mean_all.loc["SVM"],
       #"Transformer-Mini-NLI vs. Transformer-Mini": df_metrics_mean_all.loc["Transformer-Mini-NLI"] - df_metrics_mean_all.loc["Transformer-Mini"]
       }).transpose()
    #df_metrics_difference = df_metrics_difference.applymap(lambda x: f"+{round(x, 2)}" if x > 0 else round(x, 2))
    #df_metrics_difference = df_metrics_difference.applymap(lambda x: round(x, 2))
    df_metrics_difference.index.name = "Sample size"
    df_metrics_difference_dic.update({metric: df_metrics_difference.round(3)})

df_metrics_mean_dic["f1_macro"].to_excel("./appendix/f1-macro-mean.xlsx")
df_metrics_mean_dic["f1_micro"].to_excel("./appendix/f1-micro-mean.xlsx")
df_metrics_difference_dic["f1_macro"].to_excel("./appendix/f1-macro-mean-difference.xlsx")
df_metrics_difference_dic["f1_micro"].to_excel("./appendix/f1-micro-mean-difference.xlsx")



##### Training time
### comparison in run-tune speeds
col_train_time = []
col_time_sample_size = []
col_time_algo_name = []
col_time_dataset_name = []
col_eval_per_sec = []
for key_name_dataset, value_dataset in experiment_details_dic_all_methods_dataset.items():
    for key_name_algo, value_algo in experiment_details_dic_all_methods_dataset[key_name_dataset].items():
        for key_name_sample_run, value_sample_run in experiment_details_dic_all_methods_dataset[key_name_dataset][key_name_algo].items():
            if experiment_details_dic_all_methods_dataset[key_name_dataset][key_name_algo][key_name_sample_run]["n_max_sample"] in [100, 500, 1000, 2500, 5000, 10000]:
                col_time_dataset_name.append(key_name_dataset)
                col_time_algo_name.append(key_name_algo)
                col_time_sample_size.append(experiment_details_dic_all_methods_dataset[key_name_dataset][key_name_algo][key_name_sample_run]["n_max_sample"])
                col_train_time.append(round(experiment_details_dic_all_methods_dataset[key_name_dataset][key_name_algo][key_name_sample_run]["train_eval_time_per_model"] / 60, 0))
                #col_eval_per_sec.append(experiment_details_dic_all_methods_dataset[key_name_dataset][key_name_algo][key_name_sample_run]["eval_samples_per_second"])

col_hardware = ["CPU (AMD Rome 7H12)" if any(algo in algo_name for algo in ["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings"]) else "GPU (A100)" for algo_name in col_time_algo_name]

df_speed = pd.DataFrame(data={"dataset": col_time_dataset_name, "algorithm": col_time_algo_name,
                              "sample size": col_time_sample_size, "minutes training": col_train_time,
                              "hardware": col_hardware})

df_speed.algorithm = df_speed.algorithm.map(simple_algo_names_dic)  # simplify algorithm names

df_speed_mean = df_speed.groupby(by=["algorithm", "sample size"], as_index=False).apply(np.mean).round(2)
df_speed_mean["hardware"] = ["CPU (AMD Rome 7H12)" if algo in ["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings"] else "GPU (A100)" for algo in df_speed_mean.algorithm]

# sort values via categorical
df_speed_mean.algorithm = pd.Categorical(df_speed_mean.algorithm, categories=["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings", "BERT-base-nli", "BERT-base"])
df_speed_mean = df_speed_mean.sort_values(["algorithm", "sample size"])

df_speed_mean.to_csv("./appendix/training-time.csv")




#####  appendix 5 - Details on text formatting for NLI
hypothesis_template = "The quote is about {}."

### manifesto-8
# short explicit labels
explicit_labels_dic_short = OrderedDict({
    "Economy": "economy, or technology, or infrastructure, or free market",
    "External Relations": "international relations, or foreign policy, or military",
    "Fabric of Society": "law and order, or multiculturalism, or national way of life, or traditional morality",
    "Freedom and Democracy": "democracy, or freedom, or human rights, or constitutionalism",
    "Political System": "governmental efficiency, or political authority, or decentralisation, or corruption",
    "Social Groups": "agriculture, or social groups, or labour groups, or minorities",
    "Welfare and Quality of Life": "welfare, or education, or environment, or equality, or culture",
    "No other category applies": "something other than the topics economy, international relations, society, freedom and democracy, political system, social groups, welfare. It is about non of these topics"
})
# long explicit labels
explicit_labels_dic_long = OrderedDict({
    "Economy": "economy, free market economy, incentives, market regulation, economic planning, cooperation of government, employers and unions, protectionism, economic growth, technology and infrastructure, nationalisation, neoliberalism, marxism, sustainability",
    "External Relations": "international relations, foreign policy, anti-imperialism, military, peace, internationalism, European Union",
    "Fabric of Society": "society, national way of life, immigration, traditional morality, law and order, civic mindedness, solidarity, multiculturalism, diversity",
    "Freedom and Democracy": "democracy, freedom, human rights, constitutionalism, representative or direct democracy",
    "Political System": "political system, centralisation, governmental and administrative efficiency, political corruption, political authority",
    "Social Groups": "social groups, labour groups, agriculture and farmers, middle class and professional groups, minority groups, women, students, old people",
    "Welfare and Quality of Life": "welfare and quality of life, environmental protection, culture, equality, welfare state, education",
    "No other category applies": "something other than the topics economy, international relations, society, freedom and democracy, political system, social groups, welfare. It is about non of these topics"
})

hypo_short_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_short.items():
    hypo_short_dic.update({key_label: hypothesis_template.format(value_label)})

hypo_long_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_long.items():
    hypo_long_dic.update({key_label: hypothesis_template.format(value_label)})

df_hypo_short_manifesto_8 = pd.DataFrame(data={"label": hypo_short_dic.keys(), "hypotheses_short": hypo_short_dic.values()})
df_hypo_long_manifesto_8 = pd.DataFrame(data={"label": hypo_long_dic.keys(), "hypotheses_long": hypo_long_dic.values()})
df_hypo_manifesto_8 = pd.merge(df_hypo_short_manifesto_8, df_hypo_long_manifesto_8, on="label")

df_hypo_manifesto_8.to_csv("./appendix/hypotheses-manifesto-8.csv")


### manifest-military
hypothesis_hyperparams_dic = OrderedDict({
    "template_quote":
        {"Military: Positive": "The quote is positive towards the military",
         "Military: Negative": "The quote is negative towards the military",
         "Other": "The quote is not about military or defense"
         },
    "template_quote_2":  # ! performed best for most train sizes
        {"Military: Positive": "The quote is positive towards the military, for example for military spending, defense, military treaty obligations.",
         "Military: Negative": "The quote is negative towards the military, for example against military spending, for disarmament, against conscription.",
         "Other": "The quote is not about military or defense"
}})
df_hypo_short_manifesto_military = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_quote"].keys(), "hypotheses_short": hypothesis_hyperparams_dic["template_quote"].values()})
df_hypo_long_manifesto_military = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_quote_2"].keys(), "hypotheses_long": hypothesis_hyperparams_dic["template_quote_2"].values()})
df_hypo_manifesto_military = pd.merge(df_hypo_short_manifesto_military, df_hypo_long_manifesto_military, on="label")

df_hypo_manifesto_military.to_csv("./appendix/hypotheses-manifesto-military.csv")


### manifesto-protectionism
hypothesis_hyperparams_dic = OrderedDict({
    "template_quote":
        {"Protectionism: Positive": "The quote is positive towards protectionism, for example protection of internal markets through tariffs or subsidies",
         "Protectionism: Negative": "The quote is negative towards protectionism, for example in favour of free trade or open markets",
         "Other": "The quote is not about protectionism or free trade"  # , free trade, tariffs
         },
    "template_quote_2":
        {"Protectionism: Positive": "The quote is positive towards protectionism, for example in favour of protection of internal markets through tariffs or export subsidies or quotas",
         "Protectionism: Negative": "The quote is negative towards protectionism, for example in favour of free trade or open international markets",
         "Other": "The quote is not about protectionism or free trade"  # , free trade, tariffs
}})

df_hypo_short_manifesto_protectionism = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_quote"].keys(), "hypotheses_short": hypothesis_hyperparams_dic["template_quote"].values()})
df_hypo_long_manifesto_protectionism = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_quote_2"].keys(), "hypotheses_long": hypothesis_hyperparams_dic["template_quote_2"].values()})
df_hypo_manifesto_protectionism = pd.merge(df_hypo_short_manifesto_protectionism, df_hypo_long_manifesto_protectionism, on="label")

df_hypo_manifesto_protectionism.to_csv("./appendix/hypotheses-manifesto-protectionism.csv")


### manifesto-morality
hypothesis_hyperparams_dic = OrderedDict({
    "template_quote":
        {"Traditional Morality: Positive": "The quote is positive towards traditional morality",
         "Traditional Morality: Negative": "The quote is negative towards traditional morality",
         "Other": "The quote is not about traditional morality"
         },
    "template_quote_2":
        {
        "Traditional Morality: Positive": "The quote is positive towards traditional morality, for example in favour of traditional family values, religious institutions, or against unseemly behaviour",
        "Traditional Morality: Negative": "The quote is negative towards traditional morality, for example in favour of divorce or abortion, modern families, separation of church and state, modern values",
        "Other": "The quote is not about traditional morality, for example not about family values, abortion or religion"
        }
})
df_hypo_short_manifesto_morality = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_quote"].keys(), "hypotheses_short": hypothesis_hyperparams_dic["template_quote"].values()})
df_hypo_long_manifesto_morality = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_quote_2"].keys(), "hypotheses_long": hypothesis_hyperparams_dic["template_quote_2"].values()})
df_hypo_manifesto_morality = pd.merge(df_hypo_short_manifesto_morality, df_hypo_long_manifesto_morality, on="label")

df_hypo_manifesto_morality.to_csv("./appendix/hypotheses-manifesto-morality.csv")



#### Sentiment Economy
hypothesis_hyperparams_dic = OrderedDict({
    "template_quote":
        {"positive": "The quote is overall positive",
         "negative": "The quote is overall negative",
         },
    "template_complex":
        {"positive": "The economy is performing well overall",
         "negative": "The economy is performing badly overall",
         }
})
df_hypo_quote_senti = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_quote"].keys(), "hypotheses_quote": hypothesis_hyperparams_dic["template_quote"].values()})
df_hypo_complex_senti = pd.DataFrame(data={"label": hypothesis_hyperparams_dic["template_complex"].keys(), "hypotheses_complex": hypothesis_hyperparams_dic["template_complex"].values()})
df_hypo_senti = pd.merge(df_hypo_quote_senti, df_hypo_complex_senti, on="label")

df_hypo_senti.to_csv("./appendix/hypotheses-sentiment.csv")



#### CAP-SotU
explicit_labels_dic_short = OrderedDict({
    'Agriculture': "agriculture",
    'Culture': "cultural policy",
    'Civil Rights': "civil rights, or minorities, or civil liberties",
    'Defense': "defense, or military",
    'Domestic Commerce': "banking, or finance, or commerce",
    'Education': "education",
    'Energy': "energy, or electricity, or fossil fuels",
    'Environment': "the environment, or water, or waste, or pollution",
    'Foreign Trade': "foreign trade",
    'Government Operations': "government operations, or administration",
    'Health': "health",
    'Housing': "community development, or housing issues",
    'Immigration': "migration",
    'International Affairs': "international affairs, or foreign aid",
    'Labor': "employment, or labour",
    'Law and Crime': "law, crime, or family issues",
    'Macroeconomics': "macroeconomics",
    'Other': "other, miscellaneous",
    'Public Lands': "public lands, or water management",
    'Social Welfare': "social welfare",
    'Technology': "space, or science, or technology, or communications",
    'Transportation': "transportation",
})
explicit_labels_dic_long = OrderedDict({
    'Agriculture': "agriculture, for example: agricultural foreign trade, or subsidies to farmers, or food inspection and safety, or agricultural marketing, or animal and crop disease, or fisheries, or R&D",
    'Culture': "cultural policy",
    'Civil Rights': "civil rights, for example: minority/gender/age/handicap discrimination, or voting rights, or freedom of speech, or privacy",
    'Defense': "defense, for example: defense alliances, or military intelligence, or military readiness, or nuclear arms, or military aid, or military personnel issues, or military procurement, or reserve forces, or hazardous waste, or civil defense and terrorism, or contractors, or foreign operations, or R&D",
    'Domestic Commerce': "domestic commerce, for example: banking, or securities and commodities, or consumer finance, or insurance regulation, or bankruptcy, or corporate management, or small businesses, or copyrights and patents, or disaster relief, or tourism, or consumer safety, or sports regulation, or R&D",
    'Education': "education, for example: higher education, or education finance, or schools, or education of underprivileged, or vocational education, or education for handicapped, or excellence, or R&D",
    'Energy': "energy, for example: nuclear energy and safety, or electricity, or natural gas & oil, or coal, or alternative and renewable energy, or conservation, or R&D",
    'Environment': "the environment, for example: drinking water, or waste disposal, or hazardous waste, or air pollution, or recycling, or species and forest protection, or conservation, or R&D",
    'Foreign Trade': "foreign trade, for example: trade agreements, or exports, or private investments, or competitiveness, or tariff and imports, or exchange rates",
    'Government Operations': "government operations, for example: intergovernmental relations, or agencies, or bureaucracy, or postal service, or civil employees, or appointments, or national currency, or government procurement, or government property management, or tax administration, or public scandals, or government branch relations, or political campaigns, or census, or capital city, or national holidays",
    'Health': "health, for example: health care reform, or health insurance, or drug industry, or medical facilities, or disease prevention, or infants and children, or mental health, or drug/alcohol/tobacco abuse, or R&D",
    'Housing': "housing, for example: community development, or urban development, or rural housing, low-income assistance for housing, housing for veterans/elderly/homeless, or R&D",
    'Immigration': "migration, for example: immigration, or refugees, or citizenship",
    'International Affairs': "international affairs, for example: foreign aid, or international resources exploitation, or developing countries, or international finance, or western Europe, or specific countries, or human rights, or international organisations, or international terrorism, or diplomats",
    'Labor': "labour, for example: worker safety, or employment training, or employee benefits, or labor unions, or fair labor standards, or youth employment, or migrant and seasonal workers",
    'Law and Crime': "law and crime, for example: law enforcement agencies, or white collar crime, or illegal drugs, or court administration, or prisons, or juvenile crime, or child abuse, or family issues, or criminal and civil code, or police",
    'Macroeconomics': "macroeconomics, for example: interest rates, or unemployment, or monetary policy, or national budget, or taxes, or industrial policy",
    'Other': "other things, miscellaneous",
    'Public Lands': "public lands, for example: national parks, or indigenous affairs, or public lands, or water resources, or dependencies and territories",
    'Social Welfare': "social welfare, for example: low-income assistance, or elderly assistance, or disabled assistance, or volunteer associations, or child care, or social welfare",
    'Technology': "technology, for example: government space programs, or commercial use of space, or science transfer, or telecommunications, or regulation of media, or weather science, or computers, or internet, or R&D",
    'Transportation': "transportation, for example: mass transportation, or highways, or air travel, or railroads, or maritime, or infrastructure, or R&D",
})
hypo_short_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_short.items():
    hypo_short_dic.update({key_label: hypothesis_template.format(value_label)})
hypo_long_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_long.items():
    hypo_long_dic.update({key_label: hypothesis_template.format(value_label)})
df_hypo_short_cap_sotu = pd.DataFrame(data={"label": hypo_short_dic.keys(), "hypotheses_short": hypo_short_dic.values()})
df_hypo_long_cap_sotu = pd.DataFrame(data={"label": hypo_long_dic.keys(), "hypotheses_long": hypo_long_dic.values()})

df_hypo_cap_sotu = pd.merge(df_hypo_short_cap_sotu, df_hypo_long_cap_sotu, on="label")

df_hypo_cap_sotu.to_csv("./appendix/hypotheses-cap-sotu.csv")



#### CAP-US-Court

explicit_labels_dic_short = OrderedDict({
    'Agriculture': "agriculture",
    #'Culture': "cultural policy",
    'Civil Rights': "civil rights, or minorities, or civil liberties",
    'Defense': "defense, or military",
    'Domestic Commerce': "banking, or finance, or commerce",
    'Education': "education",
    'Energy': "energy, or electricity, or fossil fuels",
    'Environment': "the environment, or water, or waste, or pollution",
    'Foreign Trade': "foreign trade",
    'Government Operations': "government operations, or administration",
    'Health': "health",
    'Housing': "community development, or housing issues",
    'Immigration': "migration",
    'International Affairs': "international affairs, or foreign aid",
    'Labor': "employment, or labour",
    'Law and Crime': "law, crime, or family issues",
    'Macroeconomics': "macroeconomics",
    # 'Other': "other, miscellaneous",
    'Public Lands': "public lands, or water management",
    'Social Welfare': "social welfare",
    'Technology': "space, or science, or technology, or communications",
    'Transportation': "transportation",
})
explicit_labels_dic_long = OrderedDict({
    'Agriculture': "agriculture, for example: agricultural foreign trade, or subsidies to farmers, or food inspection and safety, or agricultural marketing, or animal and crop disease, or fisheries, or R&D",
    # 'Culture': "cultural policy",
    'Civil Rights': "civil rights, for example: minority/gender/age/handicap discrimination, or voting rights, or freedom of speech, or privacy",
    'Defense': "defense, for example: defense alliances, or military intelligence, or military readiness, or nuclear arms, or military aid, or military personnel issues, or military procurement, or reserve forces, or hazardous waste, or civil defense and terrorism, or contractors, or foreign operations, or R&D",
    'Domestic Commerce': "domestic commerce, for example: banking, or securities and commodities, or consumer finance, or insurance regulation, or bankruptcy, or corporate management, or small businesses, or copyrights and patents, or disaster relief, or tourism, or consumer safety, or sports regulation, or R&D",
    'Education': "education, for example: higher education, or education finance, or schools, or education of underprivileged, or vocational education, or education for handicapped, or excellence, or R&D",
    'Energy': "energy, for example: nuclear energy and safety, or electricity, or natural gas & oil, or coal, or alternative and renewable energy, or conservation, or R&D",
    'Environment': "the environment, for example: drinking water, or waste disposal, or hazardous waste, or air pollution, or recycling, or species and forest protection, or conservation, or R&D",
    'Foreign Trade': "foreign trade, for example: trade agreements, or exports, or private investments, or competitiveness, or tariff and imports, or exchange rates",
    'Government Operations': "government operations, for example: intergovernmental relations, or agencies, or bureaucracy, or postal service, or civil employees, or appointments, or national currency, or government procurement, or government property management, or tax administration, or public scandals, or government branch relations, or political campaigns, or census, or capital city, or national holidays",
    'Health': "health, for example: health care reform, or health insurance, or drug industry, or medical facilities, or disease prevention, or infants and children, or mental health, or drug/alcohol/tobacco abuse, or R&D",
    'Housing': "housing, for example: community development, or urban development, or rural housing, low-income assistance for housing, housing for veterans/elderly/homeless, or R&D",
    'Immigration': "migration, for example: immigration, or refugees, or citizenship",
    'International Affairs': "international affairs, for example: foreign aid, or international resources exploitation, or developing countries, or international finance, or western Europe, or specific countries, or human rights, or international organisations, or international terrorism, or diplomats",
    'Labor': "labour, for example: worker safety, or employment training, or employee benefits, or labor unions, or fair labor standards, or youth employment, or migrant and seasonal workers",
    'Law and Crime': "law and crime, for example: law enforcement agencies, or white collar crime, or illegal drugs, or court administration, or prisons, or juvenile crime, or child abuse, or family issues, or criminal and civil code, or police",
    'Macroeconomics': "macroeconomics, for example: interest rates, or unemployment, or monetary policy, or national budget, or taxes, or industrial policy",
    # 'Other': "other things, miscellaneous",
    'Public Lands': "public lands, for example: national parks, or indigenous affairs, or public lands, or water resources, or dependencies and territories",
    'Social Welfare': "social welfare, for example: low-income assistance, or elderly assistance, or disabled assistance, or volunteer associations, or child care, or social welfare",
    'Technology': "technology, for example: government space programs, or commercial use of space, or science transfer, or telecommunications, or regulation of media, or weather science, or computers, or internet, or R&D",
    'Transportation': "transportation, for example: mass transportation, or highways, or air travel, or railroads, or maritime, or infrastructure, or R&D",
})
hypo_short_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_short.items():
    hypo_short_dic.update({key_label: hypothesis_template.format(value_label)})
hypo_long_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_long.items():
    hypo_long_dic.update({key_label: hypothesis_template.format(value_label)})

df_hypo_short_cap_court = pd.DataFrame(data={"label": hypo_short_dic.keys(), "hypotheses_short": hypo_short_dic.values()})
df_hypo_long_cap_court = pd.DataFrame(data={"label": hypo_long_dic.keys(), "hypotheses_long": hypo_long_dic.values()})
df_hypo_cap_court = pd.merge(df_hypo_short_cap_court, df_hypo_long_cap_court, on="label")

df_hypo_long_cap_court.to_csv("./appendix/hypotheses-cap-us-court.csv")



#### CoronaNet
explicit_labels_dic_short = OrderedDict({
    'Anti-Disinformation Measures': "measures against disinformation",
    'COVID-19 Vaccines': "COVID-19 vaccines",
    'Closure and Regulation of Schools': "regulating schools",
    'Curfew': "a curfew",
    'Declaration of Emergency': "declaration of emergency",
    'External Border Restrictions': "external border restrictions",
    'Health Monitoring': "health monitoring",
    'Health Resources': "health resources, materials, infrastructure, personnel, mask purchases",
    'Health Testing': "health testing",
    'Hygiene': "hygiene",
    'Internal Border Restrictions': "internal border restrictions",
    'Lockdown': "a lockdown",
    'New Task Force, Bureau or Administrative Configuration': "a new administrative body",
    'Public Awareness Measures': "public awareness measures",
    'Quarantine': "quarantine",
    'Restriction and Regulation of Businesses': "restricting or regulating businesses",
    'Restriction and Regulation of Government Services': "restricting or regulating government services or public facilities",
    'Restrictions of Mass Gatherings': "restrictions of mass gatherings",
    'Social Distancing': "social distancing, reducing contact, mask wearing",
    "Other Policy Not Listed Above": "something other than regulation of businesses, government, gatherings, distancing, quarantine, lockdown, curfew, emergency, vaccine, disinformation, schools, borders or travel, testing, resources. It is not about any of these topics."
})
explicit_labels_dic_long = OrderedDict({
    'Anti-Disinformation Measures': "measures against disinformation: Efforts by the government to limit the spread of false, inaccurate or harmful information",
    'COVID-19 Vaccines': "COVID-19 vaccines. A policy regarding the research and development, or regulation, or production, or purchase, or distribution of a vaccine.",
    'Closure and Regulation of Schools': "regulating schools and educational establishments. For example closing an educational institution, or allowing educational institutions to open with or without certain conditions.",
    'Curfew': "a curfew: Domestic freedom of movement is limited during certain times of the day",
    'Declaration of Emergency': "declaration of a state of national emergency",
    'External Border Restrictions': "external border restrictions: The ability to enter or exit country borders is reduced.",
    'Health Monitoring': "health monitoring of individuals who are likely to be infected.",
    'Health Resources': "health resources: For example medical equipment, number of hospitals, health infrastructure, personnel (e.g. doctors, nurses), mask purchases",
    'Health Testing': "health testing of large populations regardless of their likelihood of being infected.",
    'Hygiene': "hygiene: Promotion of hygiene in public spaces, for example disinfection in subways or burials.",
    'Internal Border Restrictions': "internal border restrictions: The ability to move freely within the borders of a country is reduced.",
    'Lockdown': "a lockdown: People are obliged shelter in place and are only allowed to leave their shelter for specific reasons",
    'New Task Force, Bureau or Administrative Configuration': "a new administrative body, for example a new task force, bureau or administrative configuration.",
    'Public Awareness Measures': "public awareness measures or efforts to disseminate or gather reliable information, for example information on health prevention.",
    'Quarantine': "quarantine. People are obliged to isolate themselves if they are infected.",
    'Restriction and Regulation of Businesses': "restricting or regulating businesses, private commercial activities: For example closing down commercial establishments, or allowing commercial establishments to open with or without certain conditions.",
    'Restriction and Regulation of Government Services': "restricting or regulating government services or public facilities: For example closing down government services, or allowing government services to operate with or without certain conditions.",
    'Restrictions of Mass Gatherings': "restrictions of mass gatherings: The number of people allowed to congregate in a place is limited",
    'Social Distancing': "social distancing, reducing contact between individuals in public spaces, mask wearing.",
    "Other Policy Not Listed Above": "something other than regulation of businesses, government, gatherings, distancing, quarantine, lockdown, curfew, emergency, vaccines, disinformation, schools, borders or travel, testing, health resources. It is not about any of these topics."
})

hypo_short_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_short.items():
    hypo_short_dic.update({key_label: hypothesis_template.format(value_label)})
hypo_long_dic = OrderedDict()
for key_label, value_label in explicit_labels_dic_long.items():
    hypo_long_dic.update({key_label: hypothesis_template.format(value_label)})

df_hypo_short_coronanet = pd.DataFrame(data={"label": hypo_short_dic.keys(), "hypotheses_short": hypo_short_dic.values()})
df_hypo_long_coronanet = pd.DataFrame(data={"label": hypo_long_dic.keys(), "hypotheses_long": hypo_long_dic.values()})
df_hypo_coronanet = pd.merge(df_hypo_short_coronanet, df_hypo_long_coronanet, on="label")

df_hypo_coronanet.to_csv("./appendix/hypotheses-coronanet.csv")


print("Script done.")


