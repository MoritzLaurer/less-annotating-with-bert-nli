
### This script contains functions used by the analysis scripts

# import relevant packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForNextSentencePrediction, Trainer, TrainingArguments
import torch
import datasets
import copy
import numpy as np
import gc

np.random.seed(42)


### reformat training data for NLI binary classification
def format_nli_trainset(df_train=None, hypo_label_dic=None, random_seed=42):
  print(f"\nFor NLI: Augmenting data by adding random not_entail examples to the train set from other classes within the train set.")
  print(f"Length of df_train before this step is: {len(df_train)}.\n")
  print(f"Max augmentation can be: len(df_train) * 2 = {len(df_train)*2}. Can also be lower, if there are more entail examples than not-entail for a majority class")

  df_train_lst = []
  for label_text, hypothesis in hypo_label_dic.items():
    ## entailment
    df_train_step = df_train[df_train.label_text == label_text].copy(deep=True)
    df_train_step["hypothesis"] = [hypothesis] * len(df_train_step)
    df_train_step["label"] = [0] * len(df_train_step)
    ## not_entailment
    df_train_step_not_entail = df_train[df_train.label_text != label_text].copy(deep=True)
    # could try weighing the sample texts for not_entail here. e.g. to get same n texts for each label
    df_train_step_not_entail = df_train_step_not_entail.sample(n=min(len(df_train_step), len(df_train_step_not_entail)), random_state=random_seed)  # can try sampling more not_entail here
    df_train_step_not_entail["hypothesis"] = [hypothesis] * len(df_train_step_not_entail)
    df_train_step_not_entail["label"] = [1] * len(df_train_step_not_entail)
    # append
    df_train_lst.append(pd.concat([df_train_step, df_train_step_not_entail]))
  df_train = pd.concat(df_train_lst)
  
  # shuffle
  df_train = df_train.sample(frac=1, random_state=random_seed)
  df_train["label"] = df_train.label.apply(int)
  print(f"For NLI:  not_entail training examples were added, which leads to an augmented training dataset of length {len(df_train)}.")

  return df_train.copy(deep=True)


### reformat test data for NLI binary classification 
def format_nli_testset(df_test=None, hypo_label_dic=None):
  ## explode test dataset for N hypotheses
  # hypotheses
  hypothesis_lst = [value for key, value in hypo_label_dic.items()]
  print("Number of hypotheses/classes: ", len(hypothesis_lst), "\n")

  # label lists with 0 at alphabetical position of their true hypo, 1 for other hypos
  label_text_label_dic_explode = {}
  for key, value in hypo_label_dic.items():
    label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
    label_text_label_dic_explode[key] = label_lst

  df_test["label"] = df_test.label_text.map(label_text_label_dic_explode)
  df_test["hypothesis"] = [hypothesis_lst] * len(df_test)
  print(f"For normal test, N classifications necessary: {len(df_test)}")
  
  # explode dataset to have K-1 additional rows with not_entail label and K-1 other hypotheses
  # ! after exploding, cannot sample anymore, because distorts the order to true label values, which needs to be preserved for evaluation code
  df_test = df_test.explode(["hypothesis", "label"])  # multi-column explode requires pd.__version__ >= '1.3.0'
  print(f"For NLI test, N classifications necessary: {len(df_test)}\n")

  return df_test.copy(deep=True)


### data preparation function for optuna. comprises sampling, text formatting, splitting, nli-formatting
def data_preparation(random_seed=42, hypothesis_template=None, hypo_label_dic=None, n_sample=None, df_train=None, df=None, format_text_func=None, method=None, embeddings=False):
  ## unrealistic oracle sample
  #df_train_samp = df_train.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), n_sample), random_state=random_seed))
  
  ## fully random sampling
  if n_sample == 999_999:
    df_train_samp = df_train.copy(deep=True)
  else:
    df_train_samp = df_train.sample(n=min(n_sample, len(df_train)), random_state=random_seed).copy(deep=True)
    # old code for filling up at least 3 examples for class
    #df_train_samp = random_sample_fill(df_train=df_train, n_sample_per_class=n_sample_per_class, random_seed=random_seed, df=df)
  print("Number of training examples after sampling: ", len(df_train_samp), " . (but before cross-validation split) ")

  # chose the text format depending on hyperparams (with /without context? delimiter strings for nli). does it both for nli and standard_dl/ml
  df_train_samp = format_text_func(df=df_train_samp, text_format=hypothesis_template, embeddings=embeddings)

  # ~50% split cross-val as recommended by https://arxiv.org/pdf/2109.12742.pdf
  df_train_samp, df_dev_samp = train_test_split(df_train_samp, test_size=0.40, shuffle=True, random_state=random_seed)
  print(f"Final train test length after cross-val split: len(df_train_samp) = {len(df_train_samp)}, len(df_dev_samp) {len(df_dev_samp)}.")

  # format train and dev set for NLI etc.
  if method == "nli":
    df_train_samp = format_nli_trainset(df_train=df_train_samp, hypo_label_dic=hypo_label_dic)  # hypo_label_dic_short , hypo_label_dic_long
    df_dev_samp = format_nli_testset(df_test=df_dev_samp, hypo_label_dic=hypo_label_dic)  # hypo_label_dic_short , hypo_label_dic_long
  
  return df_train_samp, df_dev_samp



def load_model_tokenizer(model_name=None, method=None, label_text_alphabetical=None):
    if method == "nli":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512);
        model = AutoModelForSequenceClassification.from_pretrained(model_name); 
    elif method == "nsp":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512);
        model = AutoModelForNextSentencePrediction.from_pretrained(model_name);
    elif method == "standard_dl":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512);
        # define config. label text to label id in alphabetical order
        label2id = dict(zip(np.sort(label_text_alphabetical), np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist())) # .astype(int).tolist()
        id2label = dict(zip(np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist(), np.sort(label_text_alphabetical)))
        config = AutoConfig.from_pretrained(model_name, label2id=label2id, id2label=id2label);
        # load model with config
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config);  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model.to(device);

    return model, tokenizer




### create HF datasets and tokenize data
def tokenize_datasets(df_train_samp=None, df_test=None, tokenizer=None, method=None, max_length=None, reverse=False):
    # train, val, test all in one datasetdict:
    dataset = datasets.DatasetDict({"train": datasets.Dataset.from_pandas(df_train_samp),
                                    "test": datasets.Dataset.from_pandas(df_test)})

    ### tokenize all elements in hf datasets dictionary object
    encoded_dataset = copy.deepcopy(dataset)

    def tokenize_func_nli(examples):
        return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, max_length=max_length)  # max_length=512,  padding=True

    def tokenize_func_mono(examples):
        return tokenizer(examples["text_prepared"], truncation=True, max_length=max_length)  # max_length=512,  padding=True

    # to test NSP-reverse or NLI-reverse order to text pair
    #if reverse == True:
    #    def tokenize_func_nli(examples):
    #        return tokenizer(examples["hypothesis"], examples["text_prepared"], truncation=True, max_length=max_length)  # max_length=512,  padding=True

    if method == "nli" or method == "nsp":
        encoded_dataset["train"] = dataset["train"].map(tokenize_func_nli, batched=True)  # batch_size=len(df_train)
        encoded_dataset["test"] = dataset["test"].map(tokenize_func_nli, batched=True)  # batch_size=len(df_train)

    if method == "standard_dl":
        encoded_dataset["train"] = dataset["train"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train)
        encoded_dataset["test"] = dataset["test"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train)

    return encoded_dataset



### load metrics from sklearn
# good literature review on best metrics for multiclass classification: https://arxiv.org/pdf/2008.05756.pdf
def compute_metrics_standard(eval_pred, label_text_alphabetical=None):
    labels = eval_pred.label_ids
    pred_logits = eval_pred.predictions
    preds_max = np.argmax(pred_logits, axis=1)  # argmax on each row (axis=1) in the tensor
    print(labels)
    print(preds_max)
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds_max, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds_max, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(labels, preds_max)
    acc_not_balanced = accuracy_score(labels, preds_max)

    metrics = {'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy_balanced': acc_balanced,
            'accuracy_not_b': acc_not_balanced,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'label_gold_raw': labels,
            'label_predicted_raw': preds_max
            }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without label lists
    print("Detailed metrics: ", classification_report(labels, preds_max, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")
    
    return metrics


def compute_metrics_nli_binary(eval_pred, label_text_alphabetical=None):
    predictions, labels = eval_pred
    #print("Predictions: ", predictions)
    #print("True labels: ", labels)
    #import pdb; pdb.set_trace()

    # split in chunks with predictions for each hypothesis for one unique premise
    def chunks(lst, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # for each chunk/premise, select the most likely hypothesis, either via raw logits, or softmax
    select_class_with_softmax = True  # tested this on two datasets - output is exactly (!) the same. makes no difference. 
    softmax = torch.nn.Softmax(dim=1)
    prediction_chunks_lst = list(chunks(predictions, len(set(label_text_alphabetical)) ))  # len(LABEL_TEXT_ALPHABETICAL)
    hypo_position_highest_prob = []
    for i, chunk in enumerate(prediction_chunks_lst):
        # if else makes no empirical difference. resulting metrics are exactly the same
        if select_class_with_softmax:
          # argmax on softmax values
          #if i < 2: print("Logit chunk before softmax: ", chunk)
          chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
          chunk_tensor = softmax(chunk_tensor).tolist()
          #if i < 2: print("Logit chunk after softmax: ", chunk_tensor)
          hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))  # only accesses the first column of the array, i.e. the entailment prediction logit of all hypos and takes the highest one
        else:
          # argmax on raw logits
          #if i < 2: print("Logit chunk without softmax: ", chunk)
          hypo_position_highest_prob.append(np.argmax(chunk[:, 0]))  # only accesses the first column of the array, i.e. the entailment prediction logit of all hypos and takes the highest one
   

    label_chunks_lst = list(chunks(labels, len(set(label_text_alphabetical)) ))
    label_position_gold = []
    for chunk in label_chunks_lst:
        label_position_gold.append(np.argmin(chunk))  # argmin to detect the position of the 0 among the 1s

    #print("Prediction chunks per permise: ", prediction_chunks_lst)
    #print("Label chunks per permise: ", label_chunks_lst)

    print("Highest probability prediction per premise: ", hypo_position_highest_prob)
    print("Correct label per premise: ", label_position_gold)

    #print(hypo_position_highest_prob)
    #print(label_position_gold)

    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
    metrics = {'f1_macro': f1_macro,
               'f1_micro': f1_micro,
               'accuracy_balanced': acc_balanced,
               'accuracy_not_b': acc_not_balanced,
               'precision_macro': precision_macro,
               'recall_macro': recall_macro,
               'precision_micro': precision_micro,
               'recall_micro': recall_micro,
               'label_gold_raw': label_position_gold,
               'label_predicted_raw': hypo_position_highest_prob
               }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without label lists
    print("Detailed metrics: ", classification_report(label_position_gold, hypo_position_highest_prob, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")
    return metrics


def compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=None):
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_pred, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_pred, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_gold, label_pred)
    acc_not_balanced = accuracy_score(label_gold, label_pred)

    metrics = {'eval_f1_macro': f1_macro,
            'eval_f1_micro': f1_micro,
            'eval_accuracy_balanced': acc_balanced,
            'eval_accuracy_not_b': acc_not_balanced,
            'eval_precision_macro': precision_macro,
            'eval_recall_macro': recall_macro,
            'eval_precision_micro': precision_micro,
            'eval_recall_micro': recall_micro,
            'eval_label_gold_raw': label_gold,
            'eval_label_predicted_raw': label_pred
            }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without label lists
    print("Detailed metrics: ", classification_report(label_gold, label_pred, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")
    return metrics



### Define trainer and hyperparameters
def set_train_args(hyperparams_dic=None, training_directory=None, disable_tqdm=False, **kwargs):
    # https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    
    train_args = TrainingArguments(
        #output_dir=f'./results/{training_directory}',
        #logging_dir=f'./logs/{training_directory}',
        output_dir=f'./{training_directory}',
        logging_dir=f'./{training_directory}',
        **hyperparams_dic,
        **kwargs,
        # num_train_epochs=4,
        # learning_rate=1e-5,
        # per_device_train_batch_size=8,
        # per_device_eval_batch_size=8,
        # warmup_steps=0,  # 1000, 0
        # warmup_ratio=0,  #0.1, 0.06, 0
        # weight_decay=0,  #0.1, 0
        #load_best_model_at_end=True,
        #metric_for_best_model="f1_macro",
        #fp16=True,
        #fp16_full_eval=True,
        #evaluation_strategy="no",  # "epoch"
        #seed=42,
        # eval_steps=300  # evaluate after n steps if evaluation_strategy!='steps'. defaults to logging_steps
        save_strategy="no",  # options: "no"/"steps"/"epoch"
        # save_steps=1_000_000,              # Number of updates steps before two checkpoint saves.
        save_total_limit=10,             # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
        logging_strategy="epoch",
        report_to="all",  # "all"
        disable_tqdm=disable_tqdm,
        # push_to_hub=False,
        # push_to_hub_model_id=f"{model_name}-finetuned-{task}",
    )
    # for n, v in best_run.hyperparameters.items():
    #    setattr(trainer.args, n, v)

    return train_args


def create_trainer(model=None, tokenizer=None, encoded_dataset=None, train_args=None, label_text_alphabetical=None, method=None):
  if method == "nli" or method == "nsp":
    compute_metrics = compute_metrics_nli_binary
  elif method == "standard_dl":
    compute_metrics = compute_metrics_standard
  else:
    raise Exception(f"Compute metrics for trainer not specified correctly: {method}")

  trainer = Trainer(
      model=model,
      tokenizer=tokenizer,
      args=train_args,
      train_dataset=encoded_dataset["train"],  # ["train"].shard(index=1, num_shards=100),  # https://huggingface.co/docs/datasets/processing.html#sharding-the-dataset-shard
      eval_dataset=encoded_dataset["test"],
      compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label_text_alphabetical=label_text_alphabetical)  # compute_metrics_nli_binary  # compute_metrics
  )
  return trainer


## cleaning memory in case of memory overload
def clean_memory():
  #del(model)
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()

  ## this could fully clear memory without restart ?
  #from numba import cuda
  #cuda.select_device(0)
  #cuda.close()
  #cuda.select_device(0)
  #torch.cuda.memory_summary(device=None, abbreviated=True)
  return print("Memory cleaned")



