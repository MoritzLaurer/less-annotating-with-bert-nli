#!/usr/bin/env python
# coding: utf-8

# ## Install and load packages
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import joblib

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

## determine metric to use for figure with performance per dataset disaggregated
METRIC = "f1_macro"  # options: ['f1_macro', 'accuracy/f1_micro', 'accuracy_balanced', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro',  'cohen_kappa', 'matthews_corrcoef']




# ## Data loading
#set wd
print(os.getcwd())
if "NLI-experiments" not in os.getcwd():
    os.chdir("./NLI-experiments")
print(os.getcwd())

DATASET_NAME_LST = ["sentiment-news-econ", "coronanet", "cap-sotu", "cap-us-court", "manifesto-8",
                    "manifesto-military", "manifesto-protectionism", "manifesto-morality"]

def load_latest_experiment_dic(method_name="SVM_tfidf", dataset_name=None):
  # get latest experiment for each method for the respective dataset - experiments take a long time and many were conducted
  path_dataset = f"./results/{dataset_name}"
  file_names_lst = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]

  experiment_dates = [int(file_name.split("_")[-1].replace(".pkl", "")) for file_name in file_names_lst if (method_name in file_name) and ("experiment" in file_name)]
  #if method_name in ["SVM_tfidf", "logistic_tfidf"]:
  #   experiment_dates = [date for date in experiment_dates if date == 20220700]  # in case specific run needs to be selected

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
  experiment_details_dic_all_methods = {dataset_name: {}}
  for method_name in ["logistic_tfidf", "SVM_tfidf", "logistic_embeddings", "SVM_embeddings",
                      "deberta-v3-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
                      ]:
    experiment_dic = load_latest_experiment_dic(method_name=method_name, dataset_name=dataset_name)
    if experiment_dic != None:  # to catch cases where not experiment data for method available yet
      experiment_details_dic_all_methods[dataset_name].update({method_name: experiment_dic})

  experiment_details_dic_all_methods_dataset.update(experiment_details_dic_all_methods)


## print experiment dics to check results
for key_dataset in experiment_details_dic_all_methods_dataset:
  print(key_dataset)
  for key_method in experiment_details_dic_all_methods_dataset[key_dataset]:
    print("  ", key_method)
  print("")
#experiment_details_dic_all_methods_dataset["manifesto-military"]["xtremedistil-l6-h256-uncased"]


# testing different metrics from sklearn https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef, roc_auc_score
top_xth = 4
def compute_metrics(label_pred, label_gold, label_text_alphabetical=None):
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_pred, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_pred, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_gold, label_pred)
    #acc_not_balanced = accuracy_score(label_gold, label_pred)  # same as F1-micro
    cohen_kappa = cohen_kappa_score(label_gold, label_pred)
    matthews = matthews_corrcoef(label_gold, label_pred)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef
    #roc_auc_macro = roc_auc_score(label_gold, label_pred, average='macro', multi_class='ovo')  # no possible because requires probabilites for predictions, but categorical ints, see https://stackoverflow.com/questions/61288972/axiserror-axis-1-is-out-of-bounds-for-array-of-dimension-1-when-calculating-auc;  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

    ## manual calculation of per-class-average accuracy and per intervals (thirds)
    if not isinstance(label_gold, pd.core.series.Series):  # some label arrays are already series for some reason
        eval_gold_df = pd.DataFrame(pd.Series(label_gold, name="labels"))
    else:
        eval_gold_df = pd.DataFrame(data={"labels": label_gold.reset_index(drop=True)})
    if not isinstance(label_pred, pd.core.series.Series):  # some label arrays are already series for some reason
        eval_pred_df = pd.DataFrame(pd.Series(label_pred, name="labels"))
    else:
        eval_pred_df = pd.DataFrame(data={"labels": label_pred.reset_index(drop=True)})
    # calculate balanced accuracy manually - same as recall-macro
    """accuracy_per_class_dic = {}
    for group_name, group_df in eval_gold_df.groupby(by="labels"):
        label_gold_class_n = group_df
        label_pred_class_n = eval_pred_df[eval_pred_df.index.isin(group_df.index)]
        accuracy_per_class_dic.update({str(group_name): accuracy_score(label_gold_class_n, label_pred_class_n)})
    accuracy_balanced_manual = np.mean(list(accuracy_per_class_dic.values()))"""
    # calculate non-balanced accuracy for top 3rd and bottom two 3rd
    n_class_topshare = int(len(np.unique(eval_gold_df)) / top_xth)
    if n_class_topshare == 0: n_class_topshare = 1  # if only two classes, then n_class_topshare is 0. then set it to 1
    # top 3rd
    labels_topshare = [weird_tuple[0] for weird_tuple in eval_gold_df.value_counts()[:n_class_topshare].index.values.tolist()]
    eval_gold_df_topshare = eval_gold_df[eval_gold_df.labels.isin(labels_topshare)]
    eval_pred_df_topshare = eval_pred_df[eval_pred_df.index.isin(eval_gold_df_topshare.index)]
    accuracy_topshare = accuracy_score(eval_gold_df_topshare, eval_pred_df_topshare)
    # bottom two thirds
    labels_bottomrest = [weird_tuple[0] for weird_tuple in eval_gold_df.value_counts()[n_class_topshare:].index.values.tolist()]
    eval_gold_df_bottomrest = eval_gold_df[eval_gold_df.labels.isin(labels_bottomrest)]
    eval_pred_df_bottomrest = eval_pred_df[eval_pred_df.index.isin(eval_gold_df_bottomrest.index)]
    accuracy_bottomrest = accuracy_score(eval_gold_df_bottomrest, eval_pred_df_bottomrest)

    ## calculate F1 for high N classes (top 3rd) and low N class (bottom two 3rds) separately to see impact of class size on metrics
    class_report = classification_report(label_gold, label_pred, output_dict=True, digits=2)
    class_report = {k: v for k, v in class_report.items() if k not in ["accuracy", "macro avg", "weighted avg"]}  # remove overall aggregate metrics and only maintain per-class metrics
    class_report = dict(sorted(class_report.items(), key=lambda item: item[1]["support"], reverse=True))  # order report from highest to lowest support (examples per class)
    #class_report = {"0": {"precision": "test", "support": 1}, "1": {"precision": "test", "support": 9}, "2": {"precision": "test", "support": 5}, "3": {"precision": "test", "support": 99}}
    #class_report = dict(sorted(class_report.items(), key=lambda item: item[1]["support"]))
    # add per-class accuracy - is equivalent to per-class recall
    #class_report = {key_class_name: {**value_class_metrics, "accuracy": accuracy_per_class_dic[key_class_name]} for key_class_name, value_class_metrics in class_report.items()}
    n_class_topshare = int(len(class_report) / top_xth)
    if n_class_topshare == 0: n_class_topshare = 1  # if only two classes, then n_class_topshare is 0. then set it to 1
    class_report_topshare = {k: class_report[k] for k in list(class_report)[:n_class_topshare]}
    class_report_bottomrest = {k: class_report[k] for k in list(class_report)[n_class_topshare:]}
    f1_macro_topshare = np.mean([value_class_metrics["f1-score"] for key_class, value_class_metrics in class_report_topshare.items()])
    f1_macro_bottomrest = np.mean([value_class_metrics["f1-score"] for key_class, value_class_metrics in class_report_bottomrest.items()])
    recall_macro_topshare = np.mean([value_class_metrics["recall"] for key_class, value_class_metrics in class_report_topshare.items()])
    recall_macro_bottomrest = np.mean([value_class_metrics["recall"] for key_class, value_class_metrics in class_report_bottomrest.items()])
    precision_macro_topshare = np.mean([value_class_metrics["precision"] for key_class, value_class_metrics in class_report_topshare.items()])
    precision_macro_bottomrest = np.mean([value_class_metrics["precision"] for key_class, value_class_metrics in class_report_bottomrest.items()])

    metrics = {'f1_macro': f1_macro,
               f'f1_macro_top{top_xth}th': f1_macro_topshare,
               f'f1_macro_rest': f1_macro_bottomrest,
               'accuracy/f1_micro': f1_micro,
               #'accuracy': acc_not_balanced,
               'accuracy_balanced': acc_balanced,
               f"accuracy_top{top_xth}th": accuracy_topshare,
               "accuracy_rest": accuracy_bottomrest,
               #'accuracy_balanced_manual': accuracy_balanced_manual,  # confirmed that same as sklearn
               'recall_macro': recall_macro,
               'recall_micro': recall_micro,
               f'recall_macro_top{top_xth}th': recall_macro_topshare,
               'recall_macro_rest': recall_macro_bottomrest,
               'precision_macro': precision_macro,
               'precision_micro': precision_micro,
               f'precision_macro_top{top_xth}th': precision_macro_topshare,
               'precision_macro_rest': precision_macro_bottomrest,
               'cohen_kappa': cohen_kappa,
               'matthews_corrcoef': matthews,
               #'roc_auc_macro': roc_auc_macro
               }

    return metrics

metrics_all_name = ['f1_macro', f"f1_macro_top{top_xth}th", "f1_macro_rest",  'accuracy/f1_micro', 'accuracy_balanced',
                    'recall_macro', 'recall_micro', f'recall_macro_top{top_xth}th', 'recall_macro_rest',   # 'accuracy_balanced_manual',
                    'precision_macro', 'precision_micro', f'precision_macro_top{top_xth}th', 'precision_macro_rest',
                    f"accuracy_top{top_xth}th", "accuracy_rest",
                    'cohen_kappa', 'matthews_corrcoef']


### Adding mean balanced accuracy metric for all datasets, algos, sample sizes to "metrics_mean" sub-dictionary
# based on reviewer feedback
for key_dataset_name, experiment_details_dic_all_methods in experiment_details_dic_all_methods_dataset.items():
    for method in experiment_details_dic_all_methods.keys():
        for sample_size in experiment_details_dic_all_methods[method].keys():
            results_per_seed_dic = {key: value for key, value in experiment_details_dic_all_methods[method][sample_size].items() if "metrics_seed" in key}

            # balanced accuracy
            #accuracy_balanced_per_seed = [value_metrics["eval_accuracy_balanced"] if "eval_accuracy_balanced" in value_metrics.keys() else value_metrics["accuracy_balanced"] for key_metrics_seed, value_metrics in results_per_seed_dic.items()]  # if else, because zero-shot dicts use word "accuracy_balanced", while full runs use word "eval_accuracy_balanced"
            #accuracy_balanced_mean = np.mean(accuracy_balanced_per_seed)
            #accuracy_balanced_std = np.std(accuracy_balanced_per_seed)
            #experiment_details_dic_all_methods[method][sample_size]["metrics_mean"].update({"accuracy_balanced_mean": accuracy_balanced_mean, "accuracy_balanced_std": accuracy_balanced_std})

            ## recalculate several additional metrics based on reviewer feedback
            for metric in metrics_all_name:  #['f1_macro', 'f1_micro', 'accuracy_balanced', 'accuracy_not_b', 'precision_macro', 'recall_macro', 'precision_micro', 'recall_micro', 'cohen_kappa']:
                metric_per_seed = [compute_metrics(value_metrics["eval_label_predicted_raw"], value_metrics["eval_label_gold_raw"])[metric] if len(value_metrics.keys()) >= 8 else 0 for key_metrics_seed, value_metrics in results_per_seed_dic.items()]  # if else to distinguish between 0-shot runs for non-NLI algos and actual runs for all algos
                #metric_per_seed = [print(len(value_metrics.keys())) if len(value_metrics.keys()) >= 8 else print("pups") for key_metrics_seed, value_metrics in results_per_seed_dic.items()]  # if else to distinguish between 0-shot runs for non-NLI algos and actual runs for all algos
                metric_mean = np.mean(metric_per_seed)
                metric_std = np.std(metric_per_seed)
                experiment_details_dic_all_methods[method][sample_size]["metrics_mean"].update({f"{metric}_mean": metric_mean, f"{metric}_std": metric_std})
    print("Dataset done: ", key_dataset_name)



# ## Data preparation
dataset_n_class_dic = {"sentiment-news-econ": 2, "coronanet": 20, "cap-sotu": 22, "cap-us-court": 20, "manifesto-8": 8, "manifesto-44": 44,
                        "manifesto-military": 3, "manifesto-protectionism": 3, "manifesto-morality": 3}

#### iterate over all dataset experiment dics to extract metrics for viz
visual_data_dic_datasets = {}
for key_dataset_name, experiment_details_dic_all_methods in experiment_details_dic_all_methods_dataset.items():
  ### for one dataset iterate over each approach
  ## overall data for all approaches
  n_classes = dataset_n_class_dic[key_dataset_name]
  
  n_max_sample = []
  n_total_samples = []
  for key, value in experiment_details_dic_all_methods[list(experiment_details_dic_all_methods.keys())[0]].items():  #experiment_details_dic_all_methods["SVM"].items():
    n_max_sample.append(value["n_max_sample"])
    n_total_samples.append(value["n_train_total"])

  if n_max_sample[-1] < 10000:
      n_max_sample[-1] = f"{n_max_sample[-1]} (all)"
  x_axis_values = [f"{n_sample}" for n_sample in n_max_sample]
  #x_axis_values = ["0" if n_per_class == 0 else f"{str(n_total)} (all)" if n_per_class >= 1001 else f"{str(n_total)}"  for n_per_class, n_total in zip(n_sample_per_class, n_total_samples)]
  #if key_dataset_name == "cap-us-court":  # CAP-us-court reaches max dataset with 320 samples per class
  #  x_axis_values = ["0" if n_per_class == 0 else f"{str(n_total)} (all)" if n_per_class >= 320 else f"{str(n_total)}"  for n_per_class, n_total in zip(n_sample_per_class, n_total_samples)]

  ## old deletable code unnesting specific metrics individually for better viz format
  """
  visual_data_dic = {}
  for key_method in experiment_details_dic_all_methods:
    f1_macro_mean_lst = []
    f1_micro_mean_lst = []
    f1_macro_std_lst = []
    f1_micro_std_lst = []
    accuracy_balanced_mean_lst = []
    accuracy_balanced_std_lst = []
    for key_step in experiment_details_dic_all_methods[key_method]:
      f1_macro_mean_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_macro_mean"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
      f1_micro_mean_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_micro_mean"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
      f1_macro_std_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_macro_std"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
      f1_micro_std_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["f1_micro_std"])  # f1_macro_mean, f1_macro_std, f1_micro_mean, f1_micro_std
      # added accuracy balanced based on reviewer feedback
      accuracy_balanced_mean_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["accuracy_balanced_mean"])
      accuracy_balanced_std_lst.append(experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"]["accuracy_balanced_std"])
    dic_method = { key_method: {"f1_macro_mean": f1_macro_mean_lst, "f1_micro_mean": f1_micro_mean_lst, "f1_macro_std": f1_macro_std_lst, "f1_micro_std": f1_micro_std_lst,
                                "accuracy_balanced_mean": accuracy_balanced_mean_lst, "accuracy_balanced_std": accuracy_balanced_std_lst,
                                "n_classes": n_classes, "x_axis_values": x_axis_values} }  #"n_max_sample": n_sample_per_class, "n_total_samples": n_total_samples,
    """
  ## unnest metric results in better format for visualisation
  # generalised code for any metric
  visual_data_dic = {}
  for key_method in experiment_details_dic_all_methods:
    name_second_step = list(experiment_details_dic_all_methods[key_method].items())[2][0]
    # create empty list per metric to write/append to
    metrics_dic = {key: [] for key in experiment_details_dic_all_methods[key_method][name_second_step]["metrics_mean"].keys()}
    #metric_std_dic = {key: [] for key in experiment_details_dic_all_methods[key_method][name_second_step]["metrics_mean"].keys()}
    for key_step in experiment_details_dic_all_methods[key_method]:
        # metrics per step
        for key_metric_name, value_metric in experiment_details_dic_all_methods[key_method][key_step]["metrics_mean"].items():
            metrics_dic[key_metric_name].append(value_metric)
    dic_method = {key_method: {"n_classes": n_classes, "x_axis_values": x_axis_values, **metrics_dic} }
    #dic_method[key_method].update({key_metric_name: value_metric})
    visual_data_dic.update(dic_method)

  visual_data_dic_datasets.update({key_dataset_name: visual_data_dic})

[print(key, " viz info: ", value, "") for key, value in visual_data_dic_datasets.items()]



#### add random and majority baseline
df_test_dic = {}
for dataset_name in DATASET_NAME_LST:
  if dataset_name == "cap-us-court":
    df_test = pd.read_csv("./data_clean/df_cap_us_court_test.csv", index_col="idx")
  elif dataset_name == "sentiment-news-econ":
    df_test = pd.read_csv("./data_clean/df_sentiment_news_econ_test.csv", index_col="idx")
  elif dataset_name == "cap-sotu":
    df_test = pd.read_csv("./data_clean/df_cap_sotu_test.csv", index_col="idx")
  elif "manifesto-8" in dataset_name:
    df_test = pd.read_csv("./data_clean/df_manifesto_test.csv", index_col="idx")
  elif dataset_name == "manifesto-complex":
    df_test = pd.read_csv("./data_clean/df_manifesto_complex_test.csv", index_col="idx")
  elif dataset_name == "coronanet":
    df_test = pd.read_csv("./data_clean/df_coronanet_20220124_test.csv", index_col="idx")
  elif dataset_name == "manifesto-military":
    df_test = pd.read_csv("./data_clean/df_manifesto_military_test.csv", index_col="idx")
  elif dataset_name == "manifesto-protectionism":
    df_test = pd.read_csv("./data_clean/df_manifesto_protectionism_test.csv", index_col="idx")
  elif dataset_name == "manifesto-morality":
    df_test = pd.read_csv("./data_clean/df_manifesto_morality_test.csv", index_col="idx")
  else:
    raise Exception(f"Dataset name not found: {dataset_name}")
  df_test_dic.update({dataset_name: df_test})



### create metrics for random and majority base-line
# unnecessarily complicated way of getting all names for all metrics
#import re
#metrics_all_name = experiment_details_dic_all_methods[key_method][name_second_step]["metrics_mean"].keys()
#metrics_all_name = pd.unique([re.sub("_mean|_std", "",  metric_name) for metric_name in metrics_all_name]).tolist()

metrics_baseline_dic = {}
for key_dataset_name, value_df_test in df_test_dic.items():
  np.random.seed(SEED_GLOBAL)

  ## get random metrics averaged over several seeds
  #f1_macro_random_mean_lst = []
  #f1_micro_random_mean_lst = []
  #accuracy_balanced_random_mean_lst = []
  metric_random_dic = {metric_name: [] for metric_name in metrics_all_name}
  for seed in np.random.choice(range(1000), 10):
    np.random.seed(seed)  # to mix up the labels_random
    # label column different depending on dataset
    if "manifesto-8" in key_dataset_name:
      labels_random = np.random.choice(value_df_test.label_domain_text, len(value_df_test))
      labels_gold = value_df_test.label_domain_text
      metrics_random = compute_metrics(labels_random, labels_gold, label_text_alphabetical=np.sort(value_df_test.label_domain_text.unique()))
      #f1_macro_random_mean_lst.append(metrics_random["f1_macro"])
      #f1_micro_random_mean_lst.append(metrics_random["f1_micro"])
      #accuracy_balanced_random_mean_lst.append(metrics_random["accuracy_balanced"])
      #
      for metric in metrics_all_name:
          metric_random_dic[metric].append(metrics_random[metric])
    else:
      labels_random = np.random.choice(value_df_test.label_text, len(value_df_test))
      labels_gold = value_df_test.label_text
      metrics_random = compute_metrics(labels_random, labels_gold, label_text_alphabetical=np.sort(value_df_test.label_text.unique()))
      #f1_macro_random_mean_lst.append(metrics_random["f1_macro"])
      #f1_micro_random_mean_lst.append(metrics_random["f1_micro"])
      #accuracy_balanced_random_mean_lst.append(metrics_random["accuracy_balanced"])
      for metric in metrics_all_name:
          metric_random_dic[metric].append(metrics_random[metric])
    ## generalised version for any metric

  ## random mean per metric
  #f1_macro_random_mean = np.mean(f1_macro_random_mean_lst)
  #f1_micro_random_mean = np.mean(f1_micro_random_mean_lst)
  #accuracy_balanced_random_mean = np.mean(accuracy_balanced_random_mean_lst)
  metric_random_mean_dic = {}
  for metric in metrics_all_name:
      metric_random_mean_dic.update({f"{metric}_random": np.mean(metric_random_dic[metric])})

  ## get majority metrics
  # label column different depending on dataset
  metric_majority_dic = {}
  if "manifesto-8" in key_dataset_name:
    labels_majority = [value_df_test.label_domain_text.value_counts().idxmax()] * len(value_df_test)
    labels_gold = value_df_test.label_domain_text
    metrics_majority = compute_metrics(labels_majority, labels_gold, label_text_alphabetical=np.sort(value_df_test.label_domain_text.unique()))
    #f1_macro_majority = metrics_majority["f1_macro"]
    #f1_micro_majority = metrics_majority["f1_micro"]
    #accuracy_balanced_majority = metrics_majority["accuracy_balanced"]
    for metric in metrics_all_name:
        metric_majority_dic.update({f"{metric}_majority": metrics_majority[metric]})
  else:
    labels_majority = [value_df_test.label_text.value_counts().idxmax()] * len(value_df_test)
    labels_gold = value_df_test.label_text
    metrics_majority = compute_metrics(labels_majority, labels_gold, label_text_alphabetical=np.sort(value_df_test.label_text.unique()))
    #f1_macro_majority = metrics_majority["f1_macro"]
    #f1_micro_majority = metrics_majority["f1_micro"]
    #accuracy_balanced_majority = metrics_majority["accuracy_balanced"]
    for metric in metrics_all_name:
        metric_majority_dic.update({f"{metric}_majority": metrics_majority[metric]})

  """metrics_baseline_dic.update({key_dataset_name: {"f1_macro_random": f1_macro_random_mean, "f1_micro_random": f1_micro_random_mean,
                                                  "f1_macro_majority": f1_macro_majority, "f1_micro_majority": f1_micro_majority,
                                                  "accuracy_balanced_random": accuracy_balanced_random_mean, "accuracy_balanced_majority": accuracy_balanced_majority,
                                                  }
                               })"""
  metrics_baseline_dic.update({key_dataset_name: {**metric_random_mean_dic, **metric_majority_dic}})
np.random.seed(SEED_GLOBAL)  # rest seed to global seed


metrics_baseline_dic




# ## Visualisation

### visualisation   # https://plotly.com/python/continuous-error-bars/
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots  # https://plotly.com/python/subplots/

#metric = "f1_macro" #"f1_macro"  #"f1_micro", 
colors_hex = ["#45a7d9", "#4451c4", "#45a7d9", "#4451c4", "#7EAB55", "#FFC000"]  # order: logistic_tfidf, SVM_tfidf, logistic_embeddings, SVM_embeddings, deberta, deberta-nli   # must have same order as visual_data_dic
simple_algo_names_dic = {"logistic_tfidf": "logistic_tfidf", "logistic_embeddings": "logistic_embeddings",
                         "SVM_tfidf": "SVM_tfidf", "SVM_embeddings": "SVM_embeddings",
                         "deberta-v3-base": "BERT-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-base-nli",
                         }

### iterate over all datasets
i = 0
subplot_titles = ["Sentiment News (2 class)", "CoronaNet (20 class)", "CAP SotU (22 class)", "CAP US Court (20 class)", "Manifesto (8 class)", #"Manifesto Simple (44 class)",
                  "Manifesto Military (3 class)", "Manifesto Protectionism (3 class)", "Manifesto Morality (3 class)"]
fig = make_subplots(rows=3, cols=3, start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                    subplot_titles=subplot_titles,
                    x_title="Number of random training examples", y_title=METRIC)

for key_dataset_name, visual_data_dic in visual_data_dic_datasets.items():
  ## for one dataset, create one figure
  # determine position of subplot
  i += 1
  if i in [1, 2, 3]:
    i_row = 1
  elif i in [4, 5, 6]:
    i_row = 2
  elif i in [7, 8, 9]:
    i_row = 3
  if i in [1, 4, 7]:
    i_col = 1
  elif i in [2, 5, 8]:
    i_col = 2
  elif i in [3, 6, 9]:
    i_col = 3

  ## add random and majority baseline data
  algo_string = list(visual_data_dic.keys())[0]  # get name string for one algo to add standard info (x axis value etc.) to plot
  fig.add_trace(go.Scatter(
        name=f"random baseline ({METRIC})",
        x=visual_data_dic[algo_string]["x_axis_values"],
        y=[metrics_baseline_dic[key_dataset_name][f"{METRIC}_random"]] * len(visual_data_dic[algo_string]["x_axis_values"]),
        mode='lines',
        #line=dict(color="grey"), 
        line_dash="dot", line_color="grey",
        showlegend=False if i != 5 else True
        ), 
        row=i_row, col=i_col
  )
  fig.add_trace(go.Scatter(
        name=f"majority baseline ({METRIC})",
        x=visual_data_dic[algo_string]["x_axis_values"],
        y=[metrics_baseline_dic[key_dataset_name][f"{METRIC}_majority"]] * len(visual_data_dic[algo_string]["x_axis_values"]),
        mode='lines',
        #line=dict(color="grey"), 
        line_dash="dashdot", line_color="grey",  # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        showlegend=False if i != 5 else True
        ), 
        row=i_row, col=i_col
  )
  ## add universal politicsnli markers for indiv data point per dataset  # https://plotly.com/python/marker-style/
  """fig.add_trace(go.Scatter(
        name=f"Transformer-Mini-NLI-Politics",
        # x-axis 0 for held-out data, otherwise equivalent of 320 samp
        x=[0] if key_dataset_name in ["sentiment-news-econ", "cap-us-court", "manifesto-protectionism"] else [visual_data_dic["xtremedistil-l6-h256-uncased"]["x_axis_values"][5]], 
        y=visual_data_dic["xtremedistil-l6-h256-mnli-fever-anli-ling-politicsnli"][f"{METRIC}_mean"],
        mode='markers',
        marker_symbol="circle-open-dot",
        marker_line_color=colors_hex[-1], marker_color=colors_hex[-1],
        marker_line_width=2, marker_size=15,
        #line=dict(color="grey"), 
        #line_dash="dashdot", line_color="grey",  # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        showlegend=False if i != 5 else True
        ), 
        row=i_row, col=i_col
  )"""

  ## iterate for each method in dic and add respective line + std
  for key_algo, hex in zip(visual_data_dic, colors_hex):
    fig.add_trace(go.Scatter(
          name=simple_algo_names_dic[key_algo],
          x=visual_data_dic[key_algo]["x_axis_values"] if "nli" in key_algo else visual_data_dic[key_algo]["x_axis_values"][1:],
          y=visual_data_dic[key_algo][f"{METRIC}_mean"] if "nli" in key_algo else visual_data_dic[key_algo][f"{METRIC}_mean"][1:],
          mode='lines',
          line=dict(color=hex),
          line_dash="dash" if key_algo in ["SVM_tfidf", "logistic_tfidf"] else "solid",  #['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
          showlegend=False if i != 5 else True
          ), 
          row=i_row, col=i_col
    )
    upper_bound_y = pd.Series(visual_data_dic[key_algo][f"{METRIC}_mean"]) + pd.Series(visual_data_dic[key_algo][f"{METRIC}_std"])
    fig.add_trace(go.Scatter(
          name=f'Upper Bound {key_algo}',
          x=visual_data_dic[key_algo]["x_axis_values"] if "nli" in key_algo else visual_data_dic[key_algo]["x_axis_values"][1:],
          y=upper_bound_y if "nli" in key_algo else upper_bound_y[1:],  #pd.Series(metric_mean_nli) + pd.Series(metric_std_nli),
          mode='lines',
          marker=dict(color="#444"),
          line=dict(width=0),
          showlegend=False
          ), 
          row=i_row, col=i_col
    )
    lower_bound_y = pd.Series(visual_data_dic[key_algo][f"{METRIC}_mean"]) - pd.Series(visual_data_dic[key_algo][f"{METRIC}_std"])
    fig.add_trace(go.Scatter(
          name=f'Lower Bound {key_algo}',
          x=visual_data_dic[key_algo]["x_axis_values"] if "nli" in key_algo else visual_data_dic[key_algo]["x_axis_values"][1:],
          y=lower_bound_y if "nli" in key_algo else lower_bound_y[1:],  #pd.Series(metric_mean_nli) - pd.Series(metric_std_nli),
          marker=dict(color="#444"),
          line=dict(width=0),
          mode='lines',
          fillcolor='rgba(68, 68, 68, 0.13)',
          fill='tonexty',
          showlegend=False
          ), 
          row=i_row, col=i_col
    )

  # update layout for individual subplots  # https://stackoverflow.com/questions/63580313/update-specific-subplot-axes-in-plotly
  fig['layout'][f'xaxis{i}'].update(
      #title_text=f'N random examples given {visual_data_dic[key_algo]["n_classes"]} classes',
      tickangle=-15,
      type='category',
  )
  fig['layout'][f'yaxis{i}'].update(
      #range=[0.2, pd.Series(visual_data_dic[key_algo][f"{METRIC}_mean"]).iloc[-1] + pd.Series(visual_data_dic[key_algo][f"{METRIC}_std"]).iloc[-1] + 0.1]
      dtick=0.1
  )

# update layout for overall plot
fig.update_layout(
    title_text=f"Performance ({METRIC}) vs. Training Data Size", title_x=0.5,
    #paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)',
    template="none",  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]  # https://plotly.com/python/templates/
    height=800,
    font={"family": "verdana"}  # https://plotly.com/python/reference/layout/#layout-font
) 

fig.show(renderer="browser")
#fig.write_image("figures/fig1.png")






# ### Aggregate performance difference

## extract metrics to create df comparing performance per dataset per algo
# ! careful: not all datasets have 2500 data points, so if it says 2500, this includes 2116 for protectionism (and less full samples for higher intervals)

df_metrics_lst = []
df_std_lst = []
for metric in metrics_all_name:  #["f1_macro", "f1_micro", "accuracy_balanced"]:
    col_dataset = []
    col_algo = []
    #col_f1_macro = []
    cols_metrics_dic = {"0\n(8 datasets)": [], "100\n(8 datasets)": [], "500\n(8 datasets)": [], "1000\n(8 datasets)": [], "2500\n(8 datasets)": [], "5000\n(4 datasets)": [],
                        "10000\n(3 datasets)": []}
    cols_std_dic = {"0\n(8 datasets)": [], "100\n(8 datasets)": [], "500\n(8 datasets)": [], "1000\n(8 datasets)": [], "2500\n(8 datasets)": [], "5000\n(4 datasets)": [],
                        "10000\n(3 datasets)": []}
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

df_metrics_mean_lst = []
for i in range(len(metrics_all_name)):
    df_metrics_mean_all = df_metrics_lst[i][df_metrics_lst[i].dataset.isin(datasets_all)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["0\n(8 datasets)", "100\n(8 datasets)", "500\n(8 datasets)", "1000\n(8 datasets)", "2500\n(8 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_metrics_mean_medium = df_metrics_lst[i][df_metrics_lst[i].dataset.isin(datasets_5000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["5000\n(4 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_metrics_mean_large = df_metrics_lst[i][df_metrics_lst[i].dataset.isin(datasets_10000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["10000\n(3 datasets)"]]
    #df_metrics_mean_all = df_metrics.groupby(by="algorithm", as_index=True).apply(np.mean).round(4)
    df_metrics_mean = pd.concat([df_metrics_mean_all, df_metrics_mean_medium, df_metrics_mean_large], axis=1)
    # add row with best classical algo value
    df_metrics_mean.loc["classical-best-tfidf"] = [max(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_tfidf"], df_metrics_mean.loc["logistic_tfidf"])]
    df_metrics_mean.loc["classical-best-embeddings"] = [max(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_embeddings"], df_metrics_mean.loc["logistic_embeddings"])]
    # order rows
    order_algos = ["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings", "classical-best-tfidf", "classical-best-embeddings", "BERT-base", "BERT-base-nli"]
    df_metrics_mean = df_metrics_mean.reindex(order_algos)
    df_metrics_mean.index.name = "Sample size /\nAlgorithm"
    df_metrics_mean_lst.append(df_metrics_mean)

# average standard deviation
df_std_mean_lst = []
for i in range(len(metrics_all_name)):
    df_std_mean_all = df_std_lst[i][df_std_lst[i].dataset.isin(datasets_all)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["0\n(8 datasets)", "100\n(8 datasets)", "500\n(8 datasets)", "1000\n(8 datasets)", "2500\n(8 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_std_mean_medium = df_std_lst[i][df_std_lst[i].dataset.isin(datasets_5000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["5000\n(4 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_std_mean_large = df_std_lst[i][df_std_lst[i].dataset.isin(datasets_10000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["10000\n(3 datasets)"]]
    #df_std_mean_all = df_std.groupby(by="algorithm", as_index=True).apply(np.mean).round(4)
    df_std_mean = pd.concat([df_std_mean_all, df_std_mean_medium, df_std_mean_large], axis=1)
    # add std for best classical algo. need to go into df_metrics_mean
    df_std_mean.loc["classical-best-tfidf"] = [svm_std if max(svm_metric, lr_metric) == svm_metric else lr_std for svm_metric, lr_metric, svm_std, lr_std in zip(df_metrics_mean_lst[i].loc["SVM_tfidf"], df_metrics_mean_lst[i].loc["logistic_tfidf"], df_std_mean.loc["SVM_tfidf"], df_std_mean.loc["logistic_tfidf"])]
    df_std_mean.loc["classical-best-embeddings"] = [svm_std if max(svm_metric, lr_metric) == svm_metric else lr_std for svm_metric, lr_metric, svm_std, lr_std in zip(df_metrics_mean_lst[i].loc["SVM_embeddings"], df_metrics_mean_lst[i].loc["logistic_embeddings"], df_std_mean.loc["SVM_embeddings"], df_std_mean.loc["logistic_embeddings"])]
    # order rows
    order_algos = ["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings", "classical-best-tfidf", "classical-best-embeddings", "BERT-base", "BERT-base-nli"]
    df_std_mean = df_std_mean.reindex(order_algos)
    df_std_mean.index.name = "Sample size /\nAlgorithm"
    df_std_mean_lst.append(df_std_mean)


## difference in performance
df_metrics_difference_lst = []
for i in range(len(metrics_all_name)):
    df_metrics_difference = pd.DataFrame(data={
        #"BERT-base vs. SVM": df_metrics_mean_lst[i].loc["BERT-base"] - df_metrics_mean_lst[i].loc["SVM"],
        #"BERT-base vs. Log. Reg.": df_metrics_mean_lst[i].loc["BERT-base"] - df_metrics_mean_lst[i].loc["logistic regression"],
        "BERT-base vs. classical-best-tfidf": df_metrics_mean_lst[i].loc["BERT-base"] - df_metrics_mean_lst[i].loc["classical-best-tfidf"],
        "BERT-base vs. classical-best-embeddings": df_metrics_mean_lst[i].loc["BERT-base"] - df_metrics_mean_lst[i].loc["classical-best-embeddings"],
        #"BERT-base-nli vs. SVM": df_metrics_mean_lst[i].loc["BERT-base-nli"] - df_metrics_mean_lst[i].loc["SVM"],
        #"BERT-base-nli vs. Log. Reg.": df_metrics_mean_lst[i].loc["BERT-base-nli"] - df_metrics_mean_lst[i].loc["logistic regression"],
        "BERT-base-nli vs. classical-best-tfidf": df_metrics_mean_lst[i].loc["BERT-base-nli"] - df_metrics_mean_lst[i].loc["classical-best-tfidf"],
        "BERT-base-nli vs. classical-best-embeddings": df_metrics_mean_lst[i].loc["BERT-base-nli"] - df_metrics_mean_lst[i].loc["classical-best-embeddings"],
        "BERT-base-nli vs. BERT-base": df_metrics_mean_lst[i].loc["BERT-base-nli"] - df_metrics_mean_lst[i].loc["BERT-base"],
       }).transpose()
    #df_metrics_difference = df_metrics_difference.applymap(lambda x: f"+{round(x, 2)}" if x > 0 else round(x, 2))
    #df_metrics_difference = df_metrics_difference.applymap(lambda x: round(x, 2))
    df_metrics_difference.index.name = "Sample size /\nComparison"
    df_metrics_difference_lst.append(df_metrics_difference)





#### Visualisation of overall average performance
colors_hex = ["#45a7d9", "#4451c4", "#7EAB55", "#FFC000"]  # order: logistic_tfidf, SVM_tfidf, logistic_embeddings, SVM_embeddings, deberta, deberta-nli   # must have same order as visual_data_dic
algo_names_comparison = ["classical-best-tfidf", "classical-best-embeddings", "BERT-base", "BERT-base-nli"]

### average random baseline, changes depending on sample size, because less datasets included in higher sample size
## majority
"""
f1_macro_majority_average_all = np.mean([value["f1_macro_majority"] for key, value in metrics_baseline_dic.items()])
f1_micro_majority_average_all = np.mean([value["f1_micro_majority"] for key, value in metrics_baseline_dic.items()])
accuracy_balanced_majority_average_all = np.mean([value["accuracy_balanced_majority"] for key, value in metrics_baseline_dic.items()])
f1_macro_majority_average_5000 = np.mean([value["f1_macro_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])
f1_micro_majority_average_5000 = np.mean([value["f1_micro_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])
accuracy_balanced_majority_average_5000 = np.mean([value["accuracy_balanced_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])
f1_macro_majority_average_10000 = np.mean([value["f1_macro_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_10000])
f1_micro_majority_average_10000 = np.mean([value["f1_micro_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_10000])
accuracy_balanced_majority_average_10000 = np.mean([value["accuracy_balanced_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_10000])

metrics_majority_dic = {
    "f1_macro": [f1_macro_majority_average_all] * 5 + [f1_macro_majority_average_5000, f1_macro_majority_average_10000],
    "f1_micro": [f1_micro_majority_average_all] * 5 + [f1_micro_majority_average_5000, f1_micro_majority_average_10000],
    "accuracy_balanced": [accuracy_balanced_majority_average_all] * 5 + [accuracy_balanced_majority_average_5000, accuracy_balanced_majority_average_10000]
}
"""

metric_majority_average_all_dic = {}
metric_majority_average_5000_dic = {}
metric_majority_average_10000_dic = {}
for metric in metrics_all_name:
    metric_majority_average_all_dic.update({metric: np.mean([value[f"{metric}_majority"] for key, value in metrics_baseline_dic.items()])})
    metric_majority_average_5000_dic.update({metric: np.mean([value[f"{metric}_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])})
    metric_majority_average_10000_dic.update({metric: np.mean([[value[f"{metric}_majority"] for key, value in metrics_baseline_dic.items() if key in datasets_10000]])})

metrics_majority_dic = {}
for metric in metrics_all_name:
    metrics_majority_dic.update({metric: [metric_majority_average_all_dic[metric]] * 5 + [metric_majority_average_5000_dic[metric], metric_majority_average_10000_dic[metric]]})


################

## random
"""
f1_macro_random_average_all = np.mean([value["f1_macro_random"] for key, value in metrics_baseline_dic.items()])
f1_micro_random_average_all = np.mean([value["f1_micro_random"] for key, value in metrics_baseline_dic.items()])
accuracy_balanced_random_average_all = np.mean([value["accuracy_balanced_random"] for key, value in metrics_baseline_dic.items()])
f1_macro_random_average_5000 = np.mean([value["f1_macro_random"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])
f1_micro_random_average_5000 = np.mean([value["f1_micro_random"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])
accuracy_balanced_random_average_5000 = np.mean([value["accuracy_balanced_random"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])
f1_macro_random_average_10000 = np.mean([value["f1_macro_random"] for key, value in metrics_baseline_dic.items() if key in datasets_10000])
f1_micro_random_average_10000 = np.mean([value["f1_micro_random"] for key, value in metrics_baseline_dic.items() if key in datasets_10000])
accuracy_balanced_random_average_10000 = np.mean([value["accuracy_balanced_random"] for key, value in metrics_baseline_dic.items() if key in datasets_10000])

metrics_random_dic = {
    "f1_macro": [f1_macro_random_average_all] * 5 + [f1_macro_random_average_5000, f1_macro_random_average_10000],
    "f1_micro": [f1_micro_random_average_all] * 5 + [f1_micro_random_average_5000, f1_micro_random_average_10000],
    "accuracy_balanced": [accuracy_balanced_random_average_all] * 5 + [accuracy_balanced_random_average_5000, accuracy_balanced_random_average_10000],
}
"""

metric_random_average_all_dic = {}
metric_random_average_5000_dic = {}
metric_random_average_10000_dic = {}
for metric in metrics_all_name:
    metric_random_average_all_dic.update({metric: np.mean([value[f"{metric}_random"] for key, value in metrics_baseline_dic.items()])})
    metric_random_average_5000_dic.update({metric: np.mean([value[f"{metric}_random"] for key, value in metrics_baseline_dic.items() if key in datasets_5000])})
    metric_random_average_10000_dic.update({metric: np.mean([[value[f"{metric}_random"] for key, value in metrics_baseline_dic.items() if key in datasets_10000]])})

metrics_random_dic = {}
for metric in metrics_all_name:
    metrics_random_dic.update({metric: [metric_random_average_all_dic[metric]] * 5 + [metric_random_average_5000_dic[metric], metric_random_average_10000_dic[metric]]})



### create plot
# ! removing sample size above 2500 because not comparable and visual more confusing
metrics_all_name = ['f1_macro', f"f1_macro_top{top_xth}th", "f1_macro_rest",
                    'accuracy/f1_micro', f"accuracy_top{top_xth}th", "accuracy_rest",  #'accuracy_balanced',
                    'recall_macro', f'recall_macro_top{top_xth}th', 'recall_macro_rest',  # 'recall_micro',
                    'precision_macro', f'precision_macro_top{top_xth}th', 'precision_macro_rest',  #'precision_micro',
                    #'cohen_kappa', 'matthews_corrcoef'
                    ]

subplot_titles_compare = metrics_all_name  #["f1_macro", "accuracy/f1_micro", "accuracy_balanced"]
# determine max number of rows
i_row = 0
i_col = 0
col_max = 3
for i, metric_i in enumerate(metrics_all_name):   #["f1_macro", "f1_micro", "accuracy_balanced"]
    i_col += 1
    if i % col_max == 0:
        i_row += 1
        if i > 0:
            i_col = 1
    #print("row: ", i_row)
    #print("col: ", i_col)
print("row max: ", i_row)
print("col max: ", col_max)
fig_compare = make_subplots(rows=i_row, cols=col_max, start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                            subplot_titles=subplot_titles_compare, x_title="Number of random training examples")  #y_title="f1 score",
marker_symbols = ["circle", "circle", "circle", "circle"]  # "triangle-down", "triangle-up", "star-triangle-up", "star-square"

## create new sub-plot for each metric
i_row = 0
i_col = 0
for i, metric_i in enumerate(metrics_all_name):   #["f1_macro", "f1_micro", "accuracy_balanced"]
    # determine row and col position for each sub-figure
    i_col += 1
    if i % col_max == 0:
        i_row += 1
        if i > 0:
            i_col = 1

    fig_compare.add_trace(go.Scatter(
        name=f"majority baseline",
        x=[0, 100, 500, 1000, 2500],  #["0 (8 datasets)", "100 (8)", "500 (8)", "1000 (8)", "2500 (8)", "5000 (4)", "10000 (3)"],  #[0, 100, 500, 1000, 2500] + list(cols_metrics_dic.keys())[-2:],
        y=metrics_majority_dic[metric_i],  #[metrics_majority_average[i]] * len(list(cols_metrics_dic.keys())),
        mode='lines',
        #line=dict(color="grey"),
        line_dash="dashdot", line_color="grey", line=dict(width=3),
        showlegend=True if i == 1 else False,
        #font=dict(size=14),
        ),
        row=i_row, col=i_col
    )
    fig_compare.add_trace(go.Scatter(
        name=f"random baseline",
        x=[0, 100, 500, 1000, 2500],
        y=metrics_random_dic[metric_i],  #[metrics_random_average[i]] * len(list(cols_metrics_dic.keys())),
        mode='lines',
        #line=dict(color="grey"),
        line_dash="dot", line_color="grey", line=dict(width=3),
        showlegend=True if i == 1 else False,
        #font=dict(size=14),
        ),
        row=i_row, col=i_col
    )
    for algo, hex, marker in zip(algo_names_comparison, colors_hex, marker_symbols):
        fig_compare.add_trace(go.Scatter(
            name=algo,
            x=[0, 100, 500, 1000, 2500],
            y=df_metrics_mean_lst[i].loc[algo],  #df_metrics_mean_lst[i].loc[algo] if "nli" in algo else [np.nan] + df_metrics_mean_lst[i].loc[algo][1:].tolist(),
            mode='lines+markers',
            marker_symbol=marker,
            #marker_size=10,
            line=dict(color=hex, width=3),
            line_dash="solid",  # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
            showlegend=True if i == 1 else False,
            #font=dict(size=14),
            ),
            row=i_row, col=i_col
        )
        # add standard deviation
        upper_bound_y = pd.Series(df_metrics_mean_lst[i].loc[algo]) + pd.Series(df_std_mean_lst[i].loc[algo])
        fig_compare.add_trace(go.Scatter(
            name=f'Upper Bound {algo}',
            x=[0, 100, 500, 1000, 2500],  #visual_data_dic[algo]["x_axis_values"] if "nli" in algo else visual_data_dic[algo]["x_axis_values"][1:],
            y=upper_bound_y,  #upper_bound_y if "nli" in algo else upper_bound_y[1:],  # pd.Series(metric_mean_nli) + pd.Series(metric_std_nli),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
            ),
            row=i_row, col=i_col
        )
        lower_bound_y = pd.Series(df_metrics_mean_lst[i].loc[algo]) - pd.Series(df_std_mean_lst[i].loc[algo])
        fig_compare.add_trace(go.Scatter(
            name=f'Lower Bound {algo}',
            x=[0, 100, 500, 1000, 2500],  #visual_data_dic[algo]["x_axis_values"] if "nli" in algo else visual_data_dic[algo]["x_axis_values"][1:],
            y=lower_bound_y,  #lower_bound_y if "nli" in algo else lower_bound_y[1:],  # pd.Series(metric_mean_nli) - pd.Series(metric_std_nli),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.13)',
            fill='tonexty',
            showlegend=False
            ),
            row=i_row, col=i_col
        )
    #fig_compare.add_vline(x=4, line_dash="longdash", annotation_text="8 datasets", annotation_position="left", row=1, col=i+1)  # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'] https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_vline
    #fig_compare.add_vline(x=4, line_dash="dot", annotation_text="4 datasets", annotation_position="right", row=1, col=i+1)  # annotation=dict(font_size=20, font_family="Times New Roman")  # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_vline

    # update layout for individual subplots  # https://stackoverflow.com/questions/63580313/update-specific-subplot-axes-in-plotly
    fig_compare['layout'][f'xaxis{i+1}'].update(
        # title_text=f'N random examples given {visual_data_dic[algo]["n_classes"]} classes',
        tickangle=-10,
        type='category',
        title_font_size=16,
    )
    fig_compare['layout'][f'yaxis{i+1}'].update(
        # range=[0.2, pd.Series(visual_data_dic[algo][f"{metric}_mean"]).iloc[-1] + pd.Series(visual_data_dic[algo][f"{metric}_std"]).iloc[-1] + 0.1]
        title_text="accuracy/" + metric_i if metric_i == "f1_micro" else metric_i,
        title_font_size=16,
        dtick=0.1,
        range=[0, 0.82],
        #font=dict(size=14)
    )

# update layout for overall plot
fig_compare.update_layout(
    title_text=f"Aggregate Performance vs. Training Data Size", title_x=0.5,
    #paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)',
    template="none",  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]  # https://plotly.com/python/templates/
    margin={"l": 200},
    font=dict(size=16)
    #height=800,
)
fig_compare.show(renderer="browser")







### visualise performance difference
"""fig_difference = go.Figure()
algo_names_difference = ["BERT-base vs. classical-best-tfidf", "BERT-base-nli vs. classical-best-tfidf", "BERT-base vs. classical-best-embeddings", "BERT-base-nli vs. classical-best-embeddings", "BERT-base-nli vs. BERT-base"]
colors_hex_difference = ["#16bfb4", "#168fbf", "#bfa616", "#bf7716", "#bf16bb"]


for algo, hex in zip(algo_names_difference, colors_hex_difference):
    fig_difference.add_trace(go.Scatter(
        name=algo,
        x=list(cols_metrics_dic.keys()),
        y=df_metrics_difference.loc[algo],
        mode='lines',
        line=dict(color=hex),
        line_dash="solid",  # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        showlegend=True
        )
    )
# update layout for individual subplots  # https://stackoverflow.com/questions/63580313/update-specific-subplot-axes-in-plotly
fig_difference['layout'][f'xaxis'].update(
    # title_text=f'N random examples given {visual_data_dic[key_algo]["n_classes"]} classes',
    tickangle=-15,
    type='category',
)
fig_difference['layout'][f'yaxis'].update(
    # range=[0.2, pd.Series(visual_data_dic[key_algo][f"{metric}_mean"]).iloc[-1] + pd.Series(visual_data_dic[key_algo][f"{metric}_std"]).iloc[-1] + 0.1]
    dtick=0.1
)
# update layout for overall plot
fig_difference.update_layout(
    title_text=f"Performance difference ({metric}) of different algorithms", title_x=0.5,
    #paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)',
    template="none",  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]  # https://plotly.com/python/templates/
    #height=800,
)
fig_difference.show(renderer="browser")

"""



### deletable script for renaming file names (when "tfidf" to classical ml files)
"""import os
DATASET_NAME_LST = ["sentiment-news-econ", "coronanet", "cap-sotu", "cap-us-court", "manifesto-8",
                    "manifesto-military", "manifesto-protectionism", "manifesto-morality"]

for dataset_name in DATASET_NAME_LST:
    path = f"./results/{dataset_name}"
    files = os.listdir(path)
    for file_name in files:
        for method in ["SVM", "logistic"]:
            if (method in file_name) and ("embedding" not in file_name):
                file_name_new = file_name.replace(method, f"{method}_tfidf")
                #os.rename(os.path.join(path, file_name), os.path.join(path, file_name_new))
"""


