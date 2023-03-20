
### This script has two purposes:
# 1. It re-calculates all the metrics based on the raw output from the analysis scripts
# 2. It creates visualisations based on these metrics

# import relevant packages
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import joblib
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
from pathlib import Path

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

# setting working directory for local runs
"""
print(os.getcwd())
if "snellius" in os.getcwd():
    os.chdir("./NLI-experiments")
print(os.getcwd())
"""

# create the results/figures and results/appendix directory if it does not already exist - for code ocean
Path("../results/figures/").mkdir(parents=True, exist_ok=True)
Path("../results/appendix/").mkdir(parents=True, exist_ok=True)




### Data loading

DATASET_NAME_LST = ["sentiment-news-econ", "coronanet", "cap-sotu", "cap-us-court", "manifesto-8",
                    "manifesto-military", "manifesto-protectionism", "manifesto-morality"]

## load raw results
def load_latest_experiment_dic(method_name="SVM_tfidf", dataset_name=None):
  # get latest experiment for each method for the respective dataset - experiments take a long time and many were conducted
  path_dataset = f"./results-raw/{dataset_name}"
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
    [experiment_dic.update(joblib.load(f"./results-raw/{dataset_name}/{file_name}")) for file_name in file_names]
    return experiment_dic
  else: 
    return None


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


# calculating different metrics from sklearn https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
top_xth = 4  # defines share of classes for which to calculate additional metrics. e.g. with a value of 4, additional metrics are calculated for the largest 4th of classes
def compute_metrics(label_pred, label_gold, label_text_alphabetical=None):
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_pred, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_pred, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_gold, label_pred)
    #acc_not_balanced = accuracy_score(label_gold, label_pred)  # same as F1-micro
    cohen_kappa = cohen_kappa_score(label_gold, label_pred)
    matthews = matthews_corrcoef(label_gold, label_pred)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef
    #roc_auc_macro = roc_auc_score(label_gold, label_pred, average='macro', multi_class='ovo')  # no possible because requires probabilites for predictions, but categorical ints, see https://stackoverflow.com/questions/61288972/axiserror-axis-1-is-out-of-bounds-for-array-of-dimension-1-when-calculating-auc;  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

    ## manual calculation of per-class-average accuracy and per intervals (e.g. thirds)
    if not isinstance(label_gold, pd.core.series.Series):  # some label arrays are already series for some reason
        eval_gold_df = pd.DataFrame(pd.Series(label_gold, name="labels"))
    else:
        eval_gold_df = pd.DataFrame(data={"labels": label_gold.reset_index(drop=True)})
    if not isinstance(label_pred, pd.core.series.Series):  # some label arrays are already series for some reason
        eval_pred_df = pd.DataFrame(pd.Series(label_pred, name="labels"))
    else:
        eval_pred_df = pd.DataFrame(data={"labels": label_pred.reset_index(drop=True)})
    # calculate balanced accuracy manually - same as recall-macro
    accuracy_per_class_dic = {}
    for group_name, group_df in eval_gold_df.groupby(by="labels"):
        label_gold_class_n = group_df
        label_pred_class_n = eval_pred_df[eval_pred_df.index.isin(group_df.index)]
        accuracy_per_class_dic.update({str(group_name): accuracy_score(label_gold_class_n, label_pred_class_n)})
    accuracy_balanced_manual = np.mean(list(accuracy_per_class_dic.values()))
    # calculate non-balanced accuracy for top 3rd and bottom two 3rd
    n_class_topshare = int(len(np.unique(eval_gold_df)) / top_xth)
    if n_class_topshare == 0: n_class_topshare = 1  # if only two classes, then n_class_topshare is 0. then set it to 1
    # top xth
    labels_topshare = [weird_tuple[0] for weird_tuple in eval_gold_df.value_counts()[:n_class_topshare].index.values.tolist()]
    eval_gold_df_topshare = eval_gold_df[eval_gold_df.labels.isin(labels_topshare)]
    eval_pred_df_topshare = eval_pred_df[eval_pred_df.index.isin(eval_gold_df_topshare.index)]
    accuracy_topshare = accuracy_score(eval_gold_df_topshare, eval_pred_df_topshare)
    # bottom xth
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

    ## calculate standard deviation of per-class metrics
    accuracy_crossclass_std = np.std(list(accuracy_per_class_dic.values()))
    f1_crossclass_std = np.std([value_class_metrics["f1-score"] for key_class, value_class_metrics in class_report.items()])
    recall_crossclass_std = np.std([value_class_metrics["recall"] for key_class, value_class_metrics in class_report.items()])
    precision_crossclass_std = np.std([value_class_metrics["precision"] for key_class, value_class_metrics in class_report.items()])

    metrics = {'F1 Macro': f1_macro,
               f'f1_macro_top{top_xth}th': f1_macro_topshare,
               f'f1_macro_rest': f1_macro_bottomrest,
               'Accuracy/F1 Micro': f1_micro,
               #'accuracy': acc_not_balanced,
               'Balanced Accuracy': acc_balanced,
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
               'accuracy_crossclass_std': accuracy_crossclass_std,
               'f1_crossclass_std': f1_crossclass_std,
               'recall_crossclass_std': recall_crossclass_std,
               'precision_crossclass_std': precision_crossclass_std
               }

    return metrics

metrics_all_name = ['F1 Macro', f"f1_macro_top{top_xth}th", "f1_macro_rest",  'Accuracy/F1 Micro', 'Balanced Accuracy',
                    'recall_macro', 'recall_micro', f'recall_macro_top{top_xth}th', 'recall_macro_rest',   # 'Balanced Accuracy_manual',
                    'precision_macro', 'precision_micro', f'precision_macro_top{top_xth}th', 'precision_macro_rest',
                    f"accuracy_top{top_xth}th", "accuracy_rest",
                    'cohen_kappa', 'matthews_corrcoef',
                    'accuracy_crossclass_std', 'f1_crossclass_std', 'recall_crossclass_std', 'precision_crossclass_std']


### Adding mean balanced accuracy metric for all datasets, algos, sample sizes to "metrics_mean" sub-dictionary
# based on reviewer feedback
for key_dataset_name, experiment_details_dic_all_methods in experiment_details_dic_all_methods_dataset.items():
    for method in experiment_details_dic_all_methods.keys():
        for sample_size in experiment_details_dic_all_methods[method].keys():
            results_per_seed_dic = {key: value for key, value in experiment_details_dic_all_methods[method][sample_size].items() if "metrics_seed" in key}
            ## recalculate several additional metrics based on reviewer feedback
            for metric in metrics_all_name:  #['f1_macro', 'f1_micro', 'accuracy_balanced', 'accuracy_not_b', 'precision_macro', 'recall_macro', 'precision_micro', 'recall_micro', 'cohen_kappa']:
                metric_per_seed = [compute_metrics(value_metrics["eval_label_predicted_raw"], value_metrics["eval_label_gold_raw"])[metric] if len(value_metrics.keys()) >= 8 else 0 for key_metrics_seed, value_metrics in results_per_seed_dic.items()]  # if else to distinguish between 0-shot runs for non-NLI algos and actual runs for all algos
                #metric_per_seed = [print(len(value_metrics.keys())) if len(value_metrics.keys()) >= 8 else print("pups") for key_metrics_seed, value_metrics in results_per_seed_dic.items()]  # if else to distinguish between 0-shot runs for non-NLI algos and actual runs for all algos
                metric_mean = np.mean(metric_per_seed)
                metric_std = np.std(metric_per_seed)
                experiment_details_dic_all_methods[method][sample_size]["metrics_mean"].update({f"{metric}_mean": metric_mean, f"{metric}_std": metric_std})
    print("Dataset done: ", key_dataset_name)



### Data preparation
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



### add random and majority baseline
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
  #elif dataset_name == "manifesto-complex":
  #  df_test = pd.read_csv("./data_clean/df_manifesto_complex_test.csv", index_col="idx")
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



## create metrics for random and majority base-line
metrics_baseline_dic = {}
for key_dataset_name, value_df_test in df_test_dic.items():
  np.random.seed(SEED_GLOBAL)

  ## get random metrics averaged over several seeds
  metric_random_dic = {metric_name: [] for metric_name in metrics_all_name}
  for seed in np.random.choice(range(1000), 10):
    np.random.seed(seed)  # to mix up the labels_random
    # label column different depending on dataset
    if "manifesto-8" in key_dataset_name:
      labels_random = np.random.choice(value_df_test.label_domain_text, len(value_df_test))
      labels_gold = value_df_test.label_domain_text
      metrics_random = compute_metrics(labels_random, labels_gold, label_text_alphabetical=np.sort(value_df_test.label_domain_text.unique()))
      for metric in metrics_all_name:
          metric_random_dic[metric].append(metrics_random[metric])
    else:
      labels_random = np.random.choice(value_df_test.label_text, len(value_df_test))
      labels_gold = value_df_test.label_text
      metrics_random = compute_metrics(labels_random, labels_gold, label_text_alphabetical=np.sort(value_df_test.label_text.unique()))
      for metric in metrics_all_name:
          metric_random_dic[metric].append(metrics_random[metric])

  ## random mean per metric
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
    for metric in metrics_all_name:
        metric_majority_dic.update({f"{metric}_majority": metrics_majority[metric]})
  else:
    labels_majority = [value_df_test.label_text.value_counts().idxmax()] * len(value_df_test)
    labels_gold = value_df_test.label_text
    metrics_majority = compute_metrics(labels_majority, labels_gold, label_text_alphabetical=np.sort(value_df_test.label_text.unique()))
    for metric in metrics_all_name:
        metric_majority_dic.update({f"{metric}_majority": metrics_majority[metric]})

  metrics_baseline_dic.update({key_dataset_name: {**metric_random_mean_dic, **metric_majority_dic}})

np.random.seed(SEED_GLOBAL)  # rest seed to global seed








##### Visualisation
# visualisation   # https://plotly.com/python/continuous-error-bars/
import plotly.graph_objs as go
from plotly.subplots import make_subplots  # https://plotly.com/python/subplots/

# same as above (only copied here to avoid re-run bugs)
metrics_all_name = ['F1 Macro', f"f1_macro_top{top_xth}th", "f1_macro_rest",  'Accuracy/F1 Micro', 'Balanced Accuracy',
                    'recall_macro', 'recall_micro', f'recall_macro_top{top_xth}th', 'recall_macro_rest',   # 'Balanced Accuracy_manual',
                    'precision_macro', 'precision_micro', f'precision_macro_top{top_xth}th', 'precision_macro_rest',
                    f"accuracy_top{top_xth}th", "accuracy_rest",
                    'cohen_kappa', 'matthews_corrcoef',
                    'accuracy_crossclass_std', 'f1_crossclass_std', 'recall_crossclass_std', 'precision_crossclass_std']



#### disaggregate visualisation for all datasets
colors_hex = ["#45a7d9", "#4451c4", "#45a7d9", "#4451c4", "#7EAB55", "#FFC000"]  # order: logistic_tfidf, SVM_tfidf, logistic_embeddings, SVM_embeddings, deberta, deberta-nli   # must have same order as visual_data_dic
simple_algo_names_dic = {"logistic_tfidf": "logistic_tfidf", "logistic_embeddings": "logistic_embeddings",
                         "SVM_tfidf": "SVM_tfidf", "SVM_embeddings": "SVM_embeddings",
                         "deberta-v3-base": "BERT-base", "DeBERTa-v3-base-mnli-fever-docnli-ling-2c": "BERT-NLI",
                         }

### iterate over all datasets
def plot_per_dataset(metric_func=None):

    subplot_titles = ["Sentiment News (2 class)", "CoronaNet (20 class)", "CAP SotU (22 class)", "CAP US Court (20 class)", "Manifesto (8 class)", #"Manifesto Simple (44 class)",
                      "Manifesto Military (3 class)", "Manifesto Protectionism (3 class)", "Manifesto Morality (3 class)"]
    fig = make_subplots(rows=3, cols=3, start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                        subplot_titles=subplot_titles,
                        x_title="* Number of random training examples", y_title="* " + metric_func)
    i = 0
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

      # attempt to accomodate reviewer's axis harmonisation request. Unfortunately makes figures much harder to read
      """x_axis_harmonised = visual_data_dic[algo_string]["x_axis_values"]
      if "5000" not in visual_data_dic[algo_string]["x_axis_values"]:
          x_axis_harmonised = x_axis_harmonised + ["5000"]
      if "10000" not in visual_data_dic[algo_string]["x_axis_values"]:
          x_axis_harmonised = x_axis_harmonised + ["10000"]
      y_axis_harmonised = [metrics_baseline_dic[key_dataset_name][f"{METRIC}_random"]] * len(visual_data_dic[algo_string]["x_axis_values"])
      if "5000" not in visual_data_dic[algo_string]["x_axis_values"]:
          y_axis_harmonised = y_axis_harmonised + [y_axis_harmonised[-1]]
      if "10000" not in visual_data_dic[algo_string]["x_axis_values"]:
          y_axis_harmonised = y_axis_harmonised + [y_axis_harmonised[-1]]"""

      fig.add_trace(go.Scatter(
            name=f"random baseline ({metric_func})",
            x=visual_data_dic[algo_string]["x_axis_values"],
            y=[metrics_baseline_dic[key_dataset_name][f"{metric_func}_random"]] * len(visual_data_dic[algo_string]["x_axis_values"]),
            mode='lines',
            #line=dict(color="grey"),
            line_dash="dot", line_color="grey",
            showlegend=False if i != 5 else True
            ),
            row=i_row, col=i_col
      )
      fig.add_trace(go.Scatter(
            name=f"majority baseline ({metric_func})",
            x=visual_data_dic[algo_string]["x_axis_values"],
            y=[metrics_baseline_dic[key_dataset_name][f"{metric_func}_majority"]] * len(visual_data_dic[algo_string]["x_axis_values"]),
            mode='lines',
            #line=dict(color="grey"),
            line_dash="dashdot", line_color="grey",  # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
            showlegend=False if i != 5 else True
            ),
            row=i_row, col=i_col
      )

      ## iterate for each method in dic and add respective line + std
      for key_algo, hex in zip(visual_data_dic, colors_hex):
        fig.add_trace(go.Scatter(
              name=simple_algo_names_dic[key_algo],
              x=visual_data_dic[key_algo]["x_axis_values"] if "nli" in key_algo else visual_data_dic[key_algo]["x_axis_values"][1:],
              y=visual_data_dic[key_algo][f"{metric_func}_mean"] if "nli" in key_algo else visual_data_dic[key_algo][f"{metric_func}_mean"][1:],
              mode='lines',
              line=dict(color=hex),
              line_dash="dash" if key_algo in ["SVM_tfidf", "logistic_tfidf"] else "solid",  #['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
              showlegend=False if i != 5 else True
              ),
              row=i_row, col=i_col
        )
        upper_bound_y = pd.Series(visual_data_dic[key_algo][f"{metric_func}_mean"]) + pd.Series(visual_data_dic[key_algo][f"{metric_func}_std"])
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
        lower_bound_y = pd.Series(visual_data_dic[key_algo][f"{metric_func}_mean"]) - pd.Series(visual_data_dic[key_algo][f"{metric_func}_std"])
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
          #range=[0.2, pd.Series(visual_data_dic[key_algo][f"{metric_func}_mean"]).iloc[-1] + pd.Series(visual_data_dic[key_algo][f"{metric_func}_std"]).iloc[-1] + 0.1]
          dtick=0.1
      )

    # update layout for overall plot
    fig.update_layout(
        title_text=f"Performance ({metric_func}) vs. Training Data Size", title_x=0.5,
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
        template="none",  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]  # https://plotly.com/python/templates/
        height=595*1.5, width=842*1.7,
        font={"family": "verdana"}  # https://plotly.com/python/reference/layout/#layout-font
    )

    return fig


fig_per_dataset_macro = plot_per_dataset(metric_func="F1 Macro")
fig_per_dataset_micro = plot_per_dataset(metric_func="Accuracy/F1 Micro")

#fig_per_dataset_macro.show(renderer="browser")
fig_per_dataset_macro.write_image("../results/figures/3-figure-performance-per-dataset-f1macro.svg")
fig_per_dataset_macro.write_image("../results/figures/3-figure-performance-per-dataset-f1macro.pdf")


#fig_per_dataset_micro.show(renderer="browser")
fig_per_dataset_micro.write_image("../results/appendix/6-figure-performance-per-dataset-f1micro.svg")
fig_per_dataset_micro.write_image("../results/appendix/6-figure-performance-per-dataset-f1micro.pdf")







#### Aggregate performance difference

## extract metrics to create df comparing performance per dataset per algo
# ! careful: not all datasets have 2500 data points, so if it says 2500, this includes 2116 for protectionism (and less full samples for higher intervals)
df_metrics_dic = {}
df_std_dic = {}
for metric_name in metrics_all_name:  #["f1_macro", "f1_micro", "accuracy_balanced"]:
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
                if len(visual_data_dic_datasets[key_dataset][key_algo][f"{metric_name}_mean"]) > i:
                    cols_metrics_dic[k].append(visual_data_dic_datasets[key_dataset][key_algo][f"{metric_name}_mean"][i])
                    cols_std_dic[k].append(visual_data_dic_datasets[key_dataset][key_algo][f"{metric_name}_std"][i])
                else:
                    cols_metrics_dic[k].append(np.nan)
                    cols_std_dic[k].append(np.nan)

    ## create aggregate metric dfs
    df_metrics = pd.DataFrame(data={"dataset": col_dataset, "algorithm": col_algo, **cols_metrics_dic})
    df_std = pd.DataFrame(data={"dataset": col_dataset, "algorithm": col_algo, **cols_std_dic})
    df_metrics_dic.update({metric_name: df_metrics})
    df_std_dic.update({metric_name: df_std})


## subset average metrics by dataset size
datasets_all = ["sentiment-news-econ", "cap-us-court", "manifesto-military", "manifesto-protectionism", "manifesto-morality", "coronanet", "cap-sotu", "manifesto-8"]
datasets_5000 = ["cap-us-court", "coronanet", "cap-sotu", "manifesto-8"]
datasets_10000 = ["coronanet", "cap-sotu", "manifesto-8"]

df_metrics_mean_dic = {}
for i, metric_name in enumerate(metrics_all_name):
    df_metrics_mean_all = df_metrics_dic[metric_name][df_metrics_dic[metric_name].dataset.isin(datasets_all)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["0\n(8 datasets)", "100\n(8 datasets)", "500\n(8 datasets)", "1000\n(8 datasets)", "2500\n(8 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_metrics_mean_medium = df_metrics_dic[metric_name][df_metrics_dic[metric_name].dataset.isin(datasets_5000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["5000\n(4 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_metrics_mean_large = df_metrics_dic[metric_name][df_metrics_dic[metric_name].dataset.isin(datasets_10000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["10000\n(3 datasets)"]]
    #df_metrics_mean_all = df_metrics.groupby(by="algorithm", as_index=True).apply(np.mean).round(4)
    df_metrics_mean = pd.concat([df_metrics_mean_all, df_metrics_mean_medium, df_metrics_mean_large], axis=1)
    # add row with best classical algo value
    if metric_name not in ['accuracy_crossclass_std', 'f1_crossclass_std', 'recall_crossclass_std', 'precision_crossclass_std']:
        df_metrics_mean.loc["classical-best-tfidf"] = [max(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_tfidf"], df_metrics_mean.loc["logistic_tfidf"])]
        df_metrics_mean.loc["classical-best-embed"] = [max(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_embeddings"], df_metrics_mean.loc["logistic_embeddings"])]
    else: # minimum value for cross-class standard deviation
        df_metrics_mean.loc["classical-best-tfidf"] = [min(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_tfidf"], df_metrics_mean.loc["logistic_tfidf"])]
        df_metrics_mean.loc["classical-best-embed"] = [min(svm_metric, lr_metric) for svm_metric, lr_metric in zip(df_metrics_mean.loc["SVM_embeddings"], df_metrics_mean.loc["logistic_embeddings"])]
    # order rows
    order_algos = ["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings", "classical-best-tfidf", "classical-best-embed", "BERT-base", "BERT-NLI"]
    df_metrics_mean = df_metrics_mean.reindex(order_algos)
    df_metrics_mean.index.name = "Sample size /\nAlgorithm"
    df_metrics_mean_dic.update({metric_name: df_metrics_mean})

# average standard deviation
df_std_mean_dic = {}
for i, metric_name in enumerate(metrics_all_name):
    df_std_mean_all = df_std_dic[metric_name][df_std_dic[metric_name].dataset.isin(datasets_all)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["0\n(8 datasets)", "100\n(8 datasets)", "500\n(8 datasets)", "1000\n(8 datasets)", "2500\n(8 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_std_mean_medium = df_std_dic[metric_name][df_std_dic[metric_name].dataset.isin(datasets_5000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["5000\n(4 datasets)"]]   #.iloc[:,:-1]  # drop last column, is only us-court
    df_std_mean_large = df_std_dic[metric_name][df_std_dic[metric_name].dataset.isin(datasets_10000)].groupby(by="algorithm", as_index=True).apply(np.mean).round(4)[["10000\n(3 datasets)"]]
    #df_std_mean_all = df_std.groupby(by="algorithm", as_index=True).apply(np.mean).round(4)
    df_std_mean = pd.concat([df_std_mean_all, df_std_mean_medium, df_std_mean_large], axis=1)
    # add std for best classical algo. need to go into df_metrics_mean
    if metric_name not in ['accuracy_crossclass_std', 'f1_crossclass_std', 'recall_crossclass_std', 'precision_crossclass_std']:
        df_std_mean.loc["classical-best-tfidf"] = [svm_std if max(svm_metric, lr_metric) == svm_metric else lr_std for svm_metric, lr_metric, svm_std, lr_std in zip(df_metrics_mean_dic[metric_name].loc["SVM_tfidf"], df_metrics_mean_dic[metric_name].loc["logistic_tfidf"], df_std_mean.loc["SVM_tfidf"], df_std_mean.loc["logistic_tfidf"])]
        df_std_mean.loc["classical-best-embed"] = [svm_std if max(svm_metric, lr_metric) == svm_metric else lr_std for svm_metric, lr_metric, svm_std, lr_std in zip(df_metrics_mean_dic[metric_name].loc["SVM_embeddings"], df_metrics_mean_dic[metric_name].loc["logistic_embeddings"], df_std_mean.loc["SVM_embeddings"], df_std_mean.loc["logistic_embeddings"])]
    else: # min value for cross-class std
        df_std_mean.loc["classical-best-tfidf"] = [svm_std if min(svm_metric, lr_metric) == svm_metric else lr_std for svm_metric, lr_metric, svm_std, lr_std in
                                                   zip(df_metrics_mean_dic[metric_name].loc["SVM_tfidf"], df_metrics_mean_dic[metric_name].loc["logistic_tfidf"], df_std_mean.loc["SVM_tfidf"],
                                                       df_std_mean.loc["logistic_tfidf"])]
        df_std_mean.loc["classical-best-embed"] = [svm_std if min(svm_metric, lr_metric) == svm_metric else lr_std for svm_metric, lr_metric, svm_std, lr_std in
                                                        zip(df_metrics_mean_dic[metric_name].loc["SVM_embeddings"], df_metrics_mean_dic[metric_name].loc["logistic_embeddings"],
                                                            df_std_mean.loc["SVM_embeddings"], df_std_mean.loc["logistic_embeddings"])]
    # order rows
    order_algos = ["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings", "classical-best-tfidf", "classical-best-embed", "BERT-base", "BERT-NLI"]
    df_std_mean = df_std_mean.reindex(order_algos)
    df_std_mean.index.name = "Sample size /\nAlgorithm"
    df_std_mean_dic.update({metric_name: df_std_mean})


## difference in performance
df_metrics_difference_dic = {}
for i, metric_name in enumerate(metrics_all_name):
    df_metrics_difference = pd.DataFrame(data={
        "BERT-base vs. classical-best-tfidf": df_metrics_mean_dic[metric_name].loc["BERT-base"] - df_metrics_mean_dic[metric_name].loc["classical-best-tfidf"],
        "BERT-base vs. classical-best-embed": df_metrics_mean_dic[metric_name].loc["BERT-base"] - df_metrics_mean_dic[metric_name].loc["classical-best-embed"],
        "BERT-NLI vs. classical-best-tfidf": df_metrics_mean_dic[metric_name].loc["BERT-NLI"] - df_metrics_mean_dic[metric_name].loc["classical-best-tfidf"],
        "BERT-NLI vs. classical-best-embed": df_metrics_mean_dic[metric_name].loc["BERT-NLI"] - df_metrics_mean_dic[metric_name].loc["classical-best-embed"],
        "BERT-NLI vs. BERT-base": df_metrics_mean_dic[metric_name].loc["BERT-NLI"] - df_metrics_mean_dic[metric_name].loc["BERT-base"],
       }).transpose()
    #df_metrics_difference = df_metrics_difference.applymap(lambda x: f"+{round(x, 2)}" if x > 0 else round(x, 2))
    #df_metrics_difference = df_metrics_difference.applymap(lambda x: round(x, 2))
    df_metrics_difference.index.name = "Sample size /\nComparison"
    df_metrics_difference_dic.update({metric_name: df_metrics_difference})




#### Visualisation of overall average performance
colors_hex = ["#45a7d9", "#4451c4", "#7EAB55", "#FFC000"]  # order: logistic_tfidf, SVM_tfidf, logistic_embeddings, SVM_embeddings, deberta, deberta-nli   # must have same order as visual_data_dic
algo_names_comparison = ["classical-best-tfidf", "classical-best-embed", "BERT-base", "BERT-NLI"]

### average random baseline, changes depending on sample size, because less datasets included in higher sample size
## majority
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

## random
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



## create plot
def plot_aggregate_metrics(metrics_all_name=None, height=None):
    subplot_titles_compare = metrics_all_name  #["f1_macro", "Accuracy/F1 Micro", "Balanced Accuracy"]
    # determine max number of rows
    i_row = 0
    i_col = 0
    col_max = 3
    for i, metric_i in enumerate(metrics_all_name):   #["f1_macro", "f1_micro", "Balanced Accuracy"]
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
    for i, metric_i in enumerate(metrics_all_name):   #["f1_macro", "f1_micro", "Balanced Accuracy"]
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
                y=df_metrics_mean_dic[metric_i].loc[algo],  #df_metrics_mean_dic[metric_i].loc[algo] if "nli" in algo else [np.nan] + df_metrics_mean_dic[metric_i].loc[algo][1:].tolist(),
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
            upper_bound_y = pd.Series(df_metrics_mean_dic[metric_i].loc[algo]) + pd.Series(df_std_mean_dic[metric_i].loc[algo])
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
            lower_bound_y = pd.Series(df_metrics_mean_dic[metric_i].loc[algo]) - pd.Series(df_std_mean_dic[metric_i].loc[algo])
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
            title_text=metric_i,  #"Accuracy/" + metric_i if metric_i == "F1 Micro" else metric_i,
            title_font_size=16,
            dtick=0.1,
            range=[0.10, 0.88] #[0.15, 0.8],
            #font=dict(size=14)
        )

    # update layout for overall plot
    fig_compare.update_layout(
        title_text=f"Aggregate Performance vs. Training Data Size", title_x=0.5,
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
        template="none",  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]  # https://plotly.com/python/templates/
        margin={"l": 200}, width=1200, height=height,
        font=dict(size=16)
        #height=800,
    )

    return fig_compare


# for main text
metrics_all_name = ['F1 Macro', 'Balanced Accuracy', 'Accuracy/F1 Micro']
fig_compare_main = plot_aggregate_metrics(metrics_all_name=metrics_all_name, height=800)
#fig_compare_main.show(renderer="browser")
fig_compare_main.write_image("../results/figures/2-figure-performance-aggregate.svg")
fig_compare_main.write_image("../results/figures/2-figure-performance-aggregate.pdf")


# for annex - displaying all possible metrics
metrics_all_name = ['F1 Macro', #f"f1_macro_top{top_xth}th", "f1_macro_rest",
                    'Accuracy/F1 Micro', 'Balanced Accuracy', #f"accuracy_top{top_xth}th", "accuracy_rest",
                    'recall_macro', 'recall_micro',  #f'recall_macro_top{top_xth}th', 'recall_macro_rest',  #
                    'precision_macro', 'precision_micro',  #f'precision_macro_top{top_xth}th', 'precision_macro_rest',  #
                    'cohen_kappa', 'matthews_corrcoef'
                    ]
fig_compare_all = plot_aggregate_metrics(metrics_all_name=metrics_all_name, height=800)
#fig_compare_all.show(renderer="browser")
fig_compare_all.write_image("../results/appendix/5-figure-performance-aggregate-many-metrics.svg")
fig_compare_all.write_image("../results/appendix/5-figure-performance-aggregate-many-metrics.pdf")


# for annex - comparison of metrics by top Xth vs. rest
# part one (because too long otherwise)
metrics_all_name = ['F1 Macro', f"f1_macro_top{top_xth}th", "f1_macro_rest",
                    'Accuracy/F1 Micro', f"accuracy_top{top_xth}th", "accuracy_rest",  #'Balanced Accuracy',
                    #'recall_macro', f'recall_macro_top{top_xth}th', 'recall_macro_rest',  # 'recall_micro',
                    #'precision_macro', f'precision_macro_top{top_xth}th', 'precision_macro_rest',  #'precision_micro',
                    #'cohen_kappa', 'matthews_corrcoef'
                    ]
fig_compare_topx = plot_aggregate_metrics(metrics_all_name=metrics_all_name, height=800)
#fig_compare_topx.show(renderer="browser")
fig_compare_topx.write_image("../results/appendix/4-figure-performance-aggregate-topxth-subplot1.svg")
fig_compare_topx.write_image("../results/appendix/4-figure-performance-aggregate-topxth-subplot1.pdf")
# part two (because too long otherwise)
metrics_all_name = [#'F1 Macro', f"f1_macro_top{top_xth}th", "f1_macro_rest",
                    #'Accuracy/F1 Micro', f"accuracy_top{top_xth}th", "accuracy_rest",  #'Balanced Accuracy',
                    'recall_macro', f'recall_macro_top{top_xth}th', 'recall_macro_rest',  # 'recall_micro',
                    'precision_macro', f'precision_macro_top{top_xth}th', 'precision_macro_rest',  #'precision_micro',
                    #'cohen_kappa', 'matthews_corrcoef'
                    ]
fig_compare_topx = plot_aggregate_metrics(metrics_all_name=metrics_all_name, height=800)
#fig_compare_topx.show(renderer="browser")
fig_compare_topx.write_image("../results/appendix/4-figure-performance-aggregate-topxth-subplot2.svg")
fig_compare_topx.write_image("../results/appendix/4-figure-performance-aggregate-topxth-subplot2.pdf")




### bar chart for displaying average standard deviation
# for figure 3 in appendix

# clean column names
df_metrics_mean_dic["f1_crossclass_std"].columns = [x.replace("\n", " ") for x in df_metrics_mean_dic["f1_crossclass_std"].columns.to_list()]
df_metrics_mean_dic["accuracy_crossclass_std"].columns = [x.replace("\n", " ") for x in df_metrics_mean_dic["accuracy_crossclass_std"].columns.to_list()]
# take mean of data intervals from 100 to 2500
f1_crossclass_std_avg = df_metrics_mean_dic["f1_crossclass_std"].drop(columns=['0 (8 datasets)', "5000 (4 datasets)", "10000 (3 datasets)"]).mean(axis=1)
accuracy_crossclass_std_avg = df_metrics_mean_dic["accuracy_crossclass_std"].drop(columns=['0 (8 datasets)', "5000 (4 datasets)", "10000 (3 datasets)"]).mean(axis=1)
# take lowest classical algo values
crossclass_std_avg_dict = {}
crossclass_std_avg_dict.update({"f1_crossclass_std": f1_crossclass_std_avg.drop(["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings"])})
crossclass_std_avg_dict.update({"accuracy_crossclass_std": accuracy_crossclass_std_avg.drop(["SVM_tfidf", "logistic_tfidf", "SVM_embeddings", "logistic_embeddings"])})

metrics_all_name = ['accuracy_crossclass_std', 'f1_crossclass_std']
#colors_hex = ["#45a7d9", "#4451c4", "#7EAB55", "#FFC000"]  # order: logistic_tfidf, SVM_tfidf, logistic_embeddings, SVM_embeddings, deberta, deberta-nli   # must have same order as visual_data_dic

subplot_titles_compare = metrics_all_name  #["f1_macro", "Accuracy/F1 Micro", "Balanced Accuracy"]
# determine max number of rows
i_row = 0
i_col = 0
col_max = 2
for i, metric_i in enumerate(metrics_all_name):   #["f1_macro", "f1_micro", "Balanced Accuracy"]
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
                            subplot_titles=subplot_titles_compare)  #y_title="f1 score", x_title="Number of random training examples"

## create new sub-plot for each metric
i_row = 0
i_col = 0
for i, metric_i in enumerate(metrics_all_name):   #["f1_macro", "f1_micro", "Balanced Accuracy"]
    # determine row and col position for each sub-figure
    i_col += 1
    if i % col_max == 0:
        i_row += 1
        if i > 0:
            i_col = 1

    fig_compare.add_trace(go.Bar(
        name=metric_i,
        x=crossclass_std_avg_dict[metric_i].index,
        y=crossclass_std_avg_dict[metric_i],  #df_metrics_mean_dic[metric_i].loc[algo] if "nli" in algo else [np.nan] + df_metrics_mean_dic[metric_i].loc[algo][1:].tolist(),
        #mode='lines+markers',
        #marker_symbol=marker,
        #marker_size=10,
        #line=dict(color=hex, width=3),
        marker_color=colors_hex,
        #line_dash="solid",  # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        showlegend=False,
        #font=dict(size=14),
        ),
        row=i_row, col=i_col
    )

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
        dtick=0.05,
        range=[0.1, 0.25],
        #font=dict(size=14)
    )

# update layout for overall plot
fig_compare.update_layout(
    title_text=f"Average cross-class standard deviation", title_x=0.5,
    #paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)',
    template="none",  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]  # https://plotly.com/python/templates/
    margin={"l": 200}, width=1200, height=800,
    font=dict(size=16)
    #height=800,
)
#fig_compare.show(renderer="browser")
fig_compare.write_image("../results/appendix/3-figure-standard-deviation.svg")
fig_compare.write_image("../results/appendix/3-figure-standard-deviation.pdf")




print("Script done.")

