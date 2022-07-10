#!/usr/bin/env python
# coding: utf-8

# ## Install and load packages

# In[1]:


get_ipython().system('pip install transformers[sentencepiece]==4.13')
get_ipython().system('pip install datasets==1.17')

get_ipython().system('pip install optuna==2.10')
get_ipython().system('pip install pandas==1.3.5  # for df.explode on multiple columns')

#!pip install spacy
#!python -m spacy download en_core_web_sm
#!pip install -U easynmt  # for data augmentation


# In[2]:


# benefits of colab pro: https://colab.research.google.com/signup#advantage
# https://colab.research.google.com/notebooks/pro.ipynb?authuser=1#scrollTo=65MSuHKqNeBZ
# info on GPU
get_ipython().system('nvidia-smi')
# info on available ram
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('\n\nYour runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))


# In[4]:



import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
import random
import os
import tqdm

from google.colab.data_table import DataTable
from google.colab import data_table
data_table.enable_dataframe_formatter() # https://colab.research.google.com/notebooks/data_table.ipynb#scrollTo=JgBtx0xFFv_i

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


# ## Load & prepare data

# In[5]:


## connect to google drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
#drive.flush_and_unmount()

#set wd
print(os.getcwd())
os.chdir("/content/drive/My Drive/Colab Notebooks")
print(os.getcwd())


# In[6]:


## load data
# PImPo codebook: https://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_codebook.pdf
df = pd.read_csv("https://raw.githubusercontent.com/farzamfan/Multilingual-StD-Pipeline/main/Plm%20Manifest/PImPo%20dataset/PImPo_verbatim.csv", sep=",",  #encoding='utf-8',  # low_memory=False  #lineterminator='\t',
                 on_bad_lines='skip'
                 )
print(df.columns)
print(len(df))


# In[7]:


### data cleaning
# code book https://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_codebook.pdf

# rn variable  "A running number sorting all quasi-sentences within each document into the order in which they appeared in the document."
# pos_corpus "Comment: Gives the position of the quasi-sentence within each document in the Manifesto Corpus. It can be used to merge the original verbatim of the quasi-sentences to the dataset. It is missing for 234 quasi-sentences from the Finnish National Coalition in 2007 and one quasi-sentence from the German Greens in 2013. For the crowd coding we have worked with the beta version of the Manifesto Corpus. The respective quasi- sentences were in the beta version, but are not in the publicly available Manifesto Corpus, because they were classified as text in margin by the Manifesto Project. The R-Script provided on the website makes it possible to add the verbatim from these quasi-sentences nonetheless."

### adding preceding / following text column
df_cl = df.rename(columns={"content": "text"}).copy(deep=True)

n_unique_doc_lst = []
n_unique_doc = 0
text_preceding = []
text_following = []
for name_group, df_group in df_cl.groupby(by=["party", "date"], sort=False):  # over each doc to avoid merging sentences accross manifestos
    n_unique_doc += 1
    df_group = df_group.reset_index(drop=True)  # reset index to enable iterating over index
    for i in range(len(df_group["text"])):
        if i > 0 and i < len(df_group["text"]) - 1:
            text_preceding.append(df_group["text"][i-1])
            text_following.append(df_group["text"][i+1])
        elif i == 0:  # for very first sentence of each manifesto
            text_preceding.append("")
            text_following.append(df_group["text"][i+1])
        elif i == len(df_group["text"]) - 1:  # for last sentence
            text_preceding.append(df_group["text"][i-1])
            text_following.append("")
        else:
          raise Exception("Issue: condition not in code")
    n_unique_doc_lst.append([n_unique_doc] * len(df_group["text"]))
n_unique_doc_lst = [item for sublist in n_unique_doc_lst for item in sublist]

# create new columns
df_cl["text_original"] = df_cl["text"]
df_cl = df_cl.drop(columns=["text"])
df_cl["text_preceding"] = text_preceding
df_cl["text_following"] = text_following
df_cl["doc_id"] = n_unique_doc_lst  # column with unique doc identifier




# In[ ]:


df_cl = df_cl[['country', "doc_id", # 'gs_1r', 'gs_2r', 'date', 'party', 'pos_corpus'
               #'gs_answer_1r', 'gs_answer_2q', 'gs_answer_3q', # 'num_codings_1r', 'num_codings_2r'
              'selection', 'certainty_selection', 'topic', 'certainty_topic',
              'direction', 'certainty_direction', 'rn', 'cmp_code',  # 'manually_coded'
              'text_original', "text_preceding", "text_following"]]

## exploring multilingual data
country_map = {
    11: "swe", 12: "nor", 13: "dnk", 14: "fin", 22: "nld",
    33: "esp", 41: "deu", 42: "aut", 43: "che",
    53: "irl", 61: "usa", 62: "can", 63: "aus", 64: "nzl"
}
country_names_col = [country_map[country_id] for country_id in df_cl.country]
df_cl["country_name"] = country_names_col
# languages ISO 639-2/T https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
languages_map = {
    11: "swe", 12: "nor", 13: "dan", 14: "fin", 22: "nld",
    33: "spa", 41: "deu", 42: "deu", 43: "deu",  # manually checked Switzerland, only has deu texts
    53: "eng", 61: "eng", 62: "fra", 63: "eng", 64: "eng"  # manually checked Canada, only has fra texts
}
languages_col = [languages_map[country_id] for country_id in df_cl.country]
df_cl["language"] = languages_col

## only english?
# Ireland, USA, Australia, New Zealand # Canada
#df_cl = df_cl[df_cl.country.isin([53, 61, 63, 64])]  # 62,

### key variables:
# df_cl.selection: "Variable takes the value of 1 if immigration or immigrant integration related, and zero otherwise."
# df_cl.topic: 1 == immigration, 2 == integration, NaN == not related
# df_cl.direction: -1 == sceptical, 0 == neutral, 1 == supportive, NaN == not related to topics. "Gives the direction of a quasi-sentences as either sceptical, neutral or supportive. Missing for quasi-sentence which are classified as not-related to immigration or integration."

### gold answer variables (usec to test crowd coders)
## df_cl.gs_answer_1r == gold answers whether related to immigration or integration  used for testing crowd coders
# 0 == not immi/inti related; 1 == Immi/inti related  # "Comment: The variable gives the value of how the respective gold-sentence was coded by the authors, i.e. 0 if it was classified as not related to immigration or integration and 1 if it was regarded as related to one of these. It is missing if a quasi-sentence was not a gold sentence in the first round."
## df_cl.gs_answer_2q == Gold answers whether it is about immigration or integration or none of the two
# 1 == immigration, 2 == integration, NaN == none of the two
## df_cl.gs_answer_3q == gold answers supportive or sceptical or neutral towards topic
# -1 == sceptical, 1 == supportive, NaN == none of the two (probably neutral if also about topic)
#df_cl_gold = df_cl[['gs_answer_1r', 'gs_answer_2q', 'gs_answer_3q']]

#df_cl = df_cl[['selection', 'topic', 'direction', "text_preceding", 'text_original', "text_following"]]

## only selected texts
#df_cl[df_cl.selection != 0]
#df_cl[df_cl.manually_coded == True]

df_cl = df_cl.reset_index(drop=True)


# In[ ]:


## add gold label text column
# ! gold answer can diverge from crowd answer (e.g. index 232735). 
# ! also: if sth was gold answer for r1, it's not necessarily gold answer for r2. no gold answer for topic neutral was provided, those three where only gold for r1 (!! one of them even has divergence between gold and crowd, index 232735)

label_text_gold_lst = []
for i, row in df_cl.iterrows():
  if row["selection"] == 0:   
    label_text_gold_lst.append("no_topic")
  elif row["selection"] == 1 and row["topic"] == 1 and row["direction"] == 1:   
    label_text_gold_lst.append("immigration_supportive")
  elif row["selection"] == 1 and row["topic"] == 1 and row["direction"] == -1:   
    label_text_gold_lst.append("immigration_sceptical")
  elif row["selection"] == 1 and row["topic"] == 1 and row["direction"] == 0:   
    label_text_gold_lst.append("immigration_neutral")
  elif row["selection"] == 1 and row["topic"] == 2 and row["direction"] == 1:   
    label_text_gold_lst.append("integration_supportive")
  elif row["selection"] == 1 and row["topic"] == 2 and row["direction"] == -1:   
    label_text_gold_lst.append("integration_sceptical")
  elif row["selection"] == 1 and row["topic"] == 2 and row["direction"] == 0:       
    label_text_gold_lst.append("integration_neutral")

df_cl["label_text"] = label_text_gold_lst

df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]


# In[ ]:


DataTable(df_cl[(df_cl.language != "asdf") & (df_cl.selection == 1)], num_rows_per_page=10)


# In[ ]:


## test how many sentences have same type as preceding / following sentence
#df_test = df_cl[df_cl.label_text != "no_topic"]

test_lst = []
test_lst2 = []
test_lst_after = []
for name_df, group_df in df_cl[df_cl.language == "eng"].groupby(by="doc_id", group_keys=False, as_index=False, sort=False):
  for i in range(len(group_df)):
    # one preceding text
    if i == 0 or group_df["label_text"].iloc[i] == "no_topic":
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1]:
      test_lst.append("same_before")
    else:
      test_lst.append(f"different label before: {group_df['label_text'].iloc[i-1]}")
    # two preceding texts
    """if i < 2 or group_df["label_text"].iloc[i] == "no_topic":
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1] == group_df["label_text"].iloc[i-2]:
      test_lst2.append("same_two_before")
    else:
      test_lst2.append("different_two_before")"""
    # for following texts
    if i >= len(group_df)-1 or group_df["label_text"].iloc[i] == "no_topic":
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i+1]:
      test_lst_after.append("same_after")
    else:
      test_lst_after.append(f"different label after: {group_df['label_text'].iloc[i+1]}")

print(pd.Series(test_lst).value_counts(normalize=True), "\n")  # SOTU: 75 % of sentences have the same type as the preceeding sentence
print(pd.Series(test_lst_after).value_counts(normalize=True), "\n")  
## English
# in 50% of cases, labeled text (excluding no_topic texts) is preceded or followed by no_topic text
# in 25% by same label and in 25% by different label (other than no_topic) => no unfair advantage through data leakage if random preceding and following text added!
## Multilingual: 
# in 38% of cases, labeled text (excluding no_topic texts) is preceded or followed by no_topic text
# in 34% by same label and in 28% by different label (other than no_topic) => no unfair advantage through data leakage if random preceding and following text added!


# In[ ]:


df_cl[df_cl.selection != 0].country_name.value_counts()

print(df_cl[df_cl.selection != 0].language.value_counts())

#print(df_cl[(df_cl.language != "asdf") & (df_cl.selection != 1234)].label_text.value_counts(), "\n")

for lang in df_cl.language.unique():
  print(lang)
  print(df_cl[(df_cl.language == lang) & (df_cl.selection != 1234)].label_text.value_counts(), "\n")



## XNLI: "English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu" https://arxiv.org/pdf/1809.05053.pdf
# not in XNLI, but in PImPo: nld, nor, dan, swe, fin  -  languages in XNLI & PImPo: deu, spa, eng, fra; 


# In[ ]:


#### decide on language
#df_cl = df_cl[df_cl.language == "eng"]


# In[ ]:


### separate train and test here

## use gold set from dataset creators as training set? Issue: three categories only 1-3 examples & for integration_neutral, one of two gold does not even agree with the crowd
#df_train = df_cl[~df_cl['gs_answer_1r'].isna() | ~df_cl['gs_answer_2q'].isna() | ~df_cl['gs_answer_3q'].isna()]  #'gs_answer_2q', 'gs_answer_3q'
#df_test = df_cl[~df_cl.index.isin(df_train.index)]

## random balanced sample - fixed
# only use max 20 per class for pimpo, because overall so few examples and can still simulate easily manually created dataset. can take 100/200 for no_topic
df_train = df_cl.groupby(by=["label_text", "language"], group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=(min(int(len(x)/2), 20)), random_state=SEED_GLOBAL) if x.label_text.iloc[0] != "no_topic" 
                                                                                              else x.sample(n=(min(int(len(x)/2), 50)), random_state=SEED_GLOBAL))
# ! train only on English?
#df_train = df_train[df_train.language == "eng"]

df_test = df_cl[~df_cl.index.isin(df_train.index)]

assert len(df_train) + len(df_test) == len(df_cl)

# sample of no-topic test set for faster testing
# ! need to introduce balancing of languages here somehow ?
df_test = df_test.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=(min(len(x), 1000)), random_state=SEED_GLOBAL))

# show train-test distribution
df_train_test_dist = pd.DataFrame([df_train.label_text.value_counts().rename("train"), df_test.label_text.value_counts().rename("test"), df_cl.label_text.value_counts().rename("data_all")]).transpose()
df_train_test_dist


# In[ ]:


# show label distribution across languages
test_language = df_train.groupby(by="language", group_keys=True, as_index=True, sort=False).apply(lambda x: x.label_text.value_counts())
DataTable(pd.DataFrame(test_language))


# In[ ]:


DataTable(df_cl[df_cl.label_text != "no_topic"], num_rows_per_page=5, max_rows=10_000)


# In[ ]:


### test certainty of coders per category
import matplotlib.pyplot as plt

for name_df, group_df in df_cl.groupby(by="label_text", group_keys=False, as_index=False, sort=False):
  group_df.certainty_selection.value_counts(bins=5).plot.bar()
  print(name_df)
  plt.show()

# coders were relatively certain, less so for immigrant integration neutral (& sceptical)


# In[ ]:


### Create alphabetically ordered hypothesis dictionary
# !! need to fix the ordering in all notebooks!
from collections import OrderedDict

hypothesis_hyperparams_dic = OrderedDict(
    {#"hypotheses_raw_v1":
     #   {"no_topic": "It is neither about immigration, nor about immigrant integration",
     #   "immigration_supportive": "Immigration is good", 
     #   "immigration_sceptical": "Immigration is bad",
     #   "immigration_neutral": "Immigration is neither good nor bad",
     #   "integration_supportive": "Immigrant integration is good", 
     #   "integration_sceptical": "Immigrant integration is bad", 
     #   "integration_neutral": "Immigrant integration is neither good nor bad",
     #   },
    "template_quote_three_text_separate_v1":
        {"no_topic": "The quote is neither about immigration, nor about immigrant integration",  #"The quote is not about immigration and the quote is not about immigrant integration",  
        "immigration_supportive": "The quote is supportive/favourable towards immigration", 
        "immigration_sceptical": "The quote is sceptical/disapproving towards immigration",
        "immigration_neutral": "The quote is neutral/descriptive towards immigration",  # the two neutral hypos could be symultaniously true. somehow prevent them from being sampled for non_entailment?
        "integration_supportive": "The quote is supportive/favourable towards immigrant integration", 
        "integration_sceptical": "The quote is sceptical/disapproving towards immigrant integration", 
        "integration_neutral": "The quote is neutral/descriptive towards immigrant integration", #  "or describes the status quo" could add this, but can produce random false positive if "status quo" is mentioned in no_topic text
        },
     "template_quote_three_text_separate_v2":
        {"no_topic": "The quote is neither about immigration, nor about immigrant integration",  
        "immigration_supportive": "The quote is supportive towards immigration", 
        "immigration_sceptical": "The quote is sceptical towards immigration",
        "immigration_neutral": "The quote is neutral towards immigration",  # the two neutral hypos could be symultaniously true. somehow prevent them from being sampled for non_entailment?
        "integration_supportive": "The quote is supportive towards immigrant integration", 
        "integration_sceptical": "The quote is sceptical towards immigrant integration", 
        "integration_neutral": "The quote is neutral towards immigrant integration", #  "or describes the status quo" could add this, but can produce random false positive if "status quo" is mentioned in no_topic text
        },
     "template_quote_two_text_separate_v1":
        {"no_topic": "The quote is neither about immigration, nor about immigrant integration",  
        "immigration_supportive": "The quote is supportive towards immigration", 
        "immigration_sceptical": "The quote is sceptical towards immigration",
        "immigration_neutral": "The quote is neutral towards immigration",  # the two neutral hypos could be symultaniously true. somehow prevent them from being sampled for non_entailment?
        "integration_supportive": "The quote is supportive towards immigrant integration", 
        "integration_sceptical": "The quote is sceptical towards immigrant integration", 
        "integration_neutral": "The quote is neutral towards immigrant integration",  #  "or describes the status quo" could add this, but can produce random false positive if "status quo" is mentioned in no_topic text
        },
      "template_quote_one_text":
        {"no_topic": "The quote is neither about immigration, nor about immigrant integration",  
        "immigration_supportive": "The quote is supportive towards immigration", 
        "immigration_sceptical": "The quote is sceptical towards immigration",
        "immigration_neutral": "The quote is neutral towards immigration",  # the two neutral hypos could be symultaniously true. somehow prevent them from being sampled for non_entailment?
        "integration_supportive": "The quote is supportive towards immigrant integration", 
        "integration_sceptical": "The quote is sceptical towards immigrant integration", 
        "integration_neutral": "The quote is neutral towards immigrant integration",  #  "or describes the status quo" could add this, but can produce random false positive if "status quo" is mentioned in no_topic text
        },
    }
)

print(hypothesis_hyperparams_dic)

## sort hypotheses alphabetically
for key_dic, value_dic in hypothesis_hyperparams_dic.items(): 
  hypothesis_hyperparams_dic.update({key_dic: dict(sorted(value_dic.items(), key=lambda x: x[0].lower()))})

print(hypothesis_hyperparams_dic)


# In[ ]:


def format_text(df=None, text_format=None):
  if text_format == 'hypotheses_raw_v1':
    df["text"] = df.text_original
  elif text_format == 'template_quote_one_text':
    df["text"] = 'The quote: "' + df.text_original + '".'
  elif text_format == 'template_quote_two_text_separate_v1':
    df["text"] = df.text_preceding + '. - The quote: "' + df.text_original + '".'
  elif text_format == 'template_quote_three_text_separate_v1' or text_format == 'template_quote_three_text_separate_v2':
    df["text"] = df.text_preceding + '. - The quote: "' + df.text_original + '". - ' + df.text_following
  else:
    raise Exception(f'Hypothesis template not found for: {text_format}')
  return df


# #### calculations for random and majority baseline

# In[ ]:


### calculations for random and majority baseline
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score, classification_report
np.random.seed(SEED_GLOBAL)

label_gold = df_test[df_test.language != "asdf"].label_text
# random baseline
label_predicted = np.random.choice(label_gold.unique(), size=len(label_gold))
# majority baseline
#label_predicted = [label_gold.value_counts().idxmax()] * len(label_gold) 

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_predicted, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_predicted, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
acc_balanced = balanced_accuracy_score(label_gold, label_predicted)
acc_not_balanced = accuracy_score(label_gold, label_predicted)
metrics = {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'accuracy_balanced': acc_balanced, 'accuracy_not_b': acc_not_balanced,
            'precision_macro': precision_macro, 'recall_macro': recall_macro, 'precision_micro': precision_micro, 'recall_micro': recall_micro}
print("Aggregate metrics: ", metrics)
print("Detailed metrics: ", classification_report(label_gold, label_predicted, labels=np.sort(label_gold.unique()), target_names=np.sort(label_gold.unique()), sample_weight=None, digits=2, output_dict=True, zero_division='warn'), "\n")





# #### Test PImPo classification outside of training pipeline

# In[ ]:


## load model and tokenizer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
# label2id mapping
#label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
#id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {"entailment": 0, "not_entailment": 1}
id2label = {0: "entailment", "not_entailment": 1}

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"  #"MoritzLaurer/MiniLM-L6-mnli-fever-docnli-ling-2c"  #"MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli" #"./results/nli-few-shot/mnli-2c/MiniLM-L6-mnli-binary/"  #"mamlong34/MiniLM-L6-snli_mnli_fever_anli_R1_R2_R3-nli" #"sentence-transformers/all-MiniLM-L6-v2"  #"sentence-transformers/paraphrase-distilroberta-base-v2" #"sentence-transformers/paraphrase-MiniLM-L6-v2" #"distilroberta-base"  #"microsoft/deberta-base-mnli"  #"ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"  # "huggingface/distilbert-base-uncased-finetuned-mnli"  # "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # model_max_length=512
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)  #  label2id=label2id, id2label=id2label, num_labels=len(label2id), num_labels=3

#print(model.config)


# In[ ]:


#### function for inference on batched input
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm


def classify_nli_batched(premise_lst=None, hypotheses_lst=None, batch_size=80, output_format=None):

  hypothesis_lst_multi = hypotheses_lst * len(premise_lst)
  premise_lst_multi = [[premise] * len(hypotheses_lst) for premise in premise_lst]
  premise_lst_multi = [item for sublist in premise_lst_multi for item in sublist]
  #prem_hy_pair = [[premise, hypothesis] for premise, hypothesis in zip(premise_lst_multi, hypothesis_lst_multi)]
  
  # use datasets to avoid OOM
  dataset_hy_prem = datasets.Dataset.from_dict({"premise": premise_lst_multi, "hypothesis": hypothesis_lst_multi})

  print("Tokenization time: ")
  def tokenize_func(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding='max_length', max_length=512, return_tensors="pt")  # ! max_length padding not optimal. for some reason dynamic padding during data loading did not work
  dataset_hy_prem = dataset_hy_prem.map(tokenize_func)  #batched=True
  dataset_hy_prem = dataset_hy_prem.remove_columns(column_names=["premise", "hypothesis"])
  dataset_hy_prem.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

  # could not make dynamic padding during data loading work ...  #for dynamic padding during data loading, not during tokanization. should make inference faster.
  #data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')  # https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorWithPadding
  dataloader = torch.utils.data.DataLoader(dataset_hy_prem, batch_size=batch_size, collate_fn=None)  # https://discuss.huggingface.co/t/are-dynamic-padding-and-smart-batching-in-the-library/10404

  print("Inference time: ")
  logits_lst_all = np.empty([0, len(model.config.id2label)])  # the last element is the number of output classes (e.g. 3 for NLI), the first number unclear
  for batch in tqdm.notebook.tqdm(dataloader):
      inputs = {'input_ids': batch["input_ids"][0].to(device), 'attention_mask': batch["attention_mask"][0].to(device)}
      with torch.no_grad():  # for speed https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/25   
          outputs = model(**inputs)
      logits = outputs[0]
      # stack new logits from batch on logits from previous batches
      logits_lst_all = np.vstack([logits_lst_all, logits.detach().cpu().numpy()])

  ## softmax on logits
  # softmax on all 3 logits
  prediction_lst_all = [torch.softmax(torch.from_numpy(logits), dim=-1).tolist() for logits in logits_lst_all]
  # softmax only on entailment and contradiction logit
  prediction_lst_2 = [torch.softmax(torch.from_numpy(logits), dim=-1).tolist() for logits in logits_lst_all[:, [0, -1]]]

  probs_dic_all = []
  for i in range(len(prediction_lst_all)):
    if output_format == "mnli":
      probs_dic_all.append({"entailment": prediction_lst_all[i][0], "neutral": prediction_lst_all[i][1], "contradiction": prediction_lst_all[i][2],  # ! label order from ALBERT
                            "hypothesis": hypothesis_lst_multi[i], "text": premise_lst_multi[i]})
    elif output_format == "binary":
      probs_dic_all.append({"entailment": prediction_lst_all[i][0], "not_entailment": prediction_lst_all[i][1],  # ! label binary model
                            "hypothesis": hypothesis_lst_multi[i], "text": premise_lst_multi[i]})
    else:
      raise Exception("Specify a different output format. Options: 'mnli', 'binary' ")

  return probs_dic_all


# In[ ]:


### run inference

df_cl_2 = df_cl[df_cl.selection > 0].copy(deep=True)

df_cl_2["text"] = df_cl_2.text_original

probs_dic = classify_nli_batched(premise_lst=df_cl_2.text.tolist(), hypotheses_lst=list(hypothesis_hyperparams_dic["hypotheses_complex"].values()), batch_size=128, 
                                 output_format="binary")


# In[ ]:


## merge probs dic with with meta data from df_effec_eval - only difference is entailment scores are now from new model
df_nli = pd.DataFrame(probs_dic)
print(len(df_nli))

# harmonise hypothesis texts to enable merge. (only merging on sentences does not work due to duplicate sentences)
df_nli = df_nli.merge(df_cl_2, how="left", left_on=["text"], right_on=["text"])
df_nli = df_nli.drop(columns=["country", "date", "party", "doc_id", "pos_corpus", "gs_answer_1r", "gs_answer_2q", "gs_answer_3q"])

# selecting predicted label for corresponding hypo
hypothesis_dic = hypothesis_hyperparams_dic["hypotheses_complex"]
df_nli["label_text_predicted"] = [list(hypothesis_dic.keys())[list(hypothesis_dic.values()).index(hypo)] for hypo in df_nli.hypothesis]

df_nli_cl = df_nli.groupby(by="text", group_keys=False, as_index=False, sort=False).apply(lambda x: x[x.entailment == x.entailment.max()])
print(len(df_nli_cl))


# In[ ]:


## Trying to use threshold to filter out neutrals
# ! issue: entailment is always very low, well below 0.1 on average

df_nli_cl_gold = df_nli_cl[df_nli_cl.label_text == df_nli_cl.label_text_predicted]
df_nli_cl_gold.label_text.value_counts()

for group_name, group_df in df_nli_cl_gold.groupby(by="label_text", group_keys=False, as_index=False, sort=False):
  group_df.entailment.plot.line(legend=True)


# In[ ]:


DataTable(df_nli_cl, num_rows_per_page=5)


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score, classification_report
label_gold = df_nli_cl.label_text
label_predicted = df_nli_cl.label_text_predicted

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_predicted, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_predicted, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
acc_balanced = balanced_accuracy_score(label_gold, label_predicted)
acc_not_balanced = accuracy_score(label_gold, label_predicted)
metrics = {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'accuracy_balanced': acc_balanced, 'accuracy_not_b': acc_not_balanced,
            'precision_macro': precision_macro, 'recall_macro': recall_macro, 'precision_micro': precision_micro, 'recall_micro': recall_micro}
print("Aggregate metrics: ", metrics)
print("Detailed metrics: ", classification_report(label_gold, label_predicted, labels=np.sort(label_gold.unique()), target_names=np.sort(label_gold.unique()), sample_weight=None, digits=2, output_dict=True, zero_division='warn'), "\n")



# In[ ]:


# sklearn plot https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
conf_mat = confusion_matrix(label_gold, label_predicted, normalize=None)  #labels=np.sort(df_cl.label_text.unique())  
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=np.sort(label_gold.unique()))
plt.rcParams["figure.figsize"] = (9,9)
disp.plot(xticks_rotation="vertical")
plt.show()

# ! issue: there is always~ one hypothesis that dominates the rest (esp integration_neutral = "immigrant integration is OK")


# In[ ]:


print(df_nli_cl.label_text.unique())
print(df_nli_cl.label_text_predicted.unique())

for label in df_nli_cl.label_text.unique():
  if not label in df_nli_cl.label_text_predicted.unique():
    print("Issue with label: ", label)


# # Other code from copied script

# #### Create df_nli with NLI-BERT

# In[ ]:


## !! improve this later with DeBERTa and probably text separation via two quotes


# In[ ]:


## load model and tokenizer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
# label2id mapping
label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
#label2id = {"entailment": 0, "not_entailment": 1}
#id2label = {0: "entailment", "not_entailment": 1}

model_name = "MoritzLaurer/MiniLM-L6-mnli" #"./results/nli-few-shot/mnli-2c/MiniLM-L6-mnli-binary/"  #"mamlong34/MiniLM-L6-snli_mnli_fever_anli_R1_R2_R3-nli" #"sentence-transformers/all-MiniLM-L6-v2"  #"sentence-transformers/paraphrase-distilroberta-base-v2" #"sentence-transformers/paraphrase-MiniLM-L6-v2" #"distilroberta-base"  #"microsoft/deberta-base-mnli"  #"ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"  # "huggingface/distilbert-base-uncased-finetuned-mnli"  # "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # model_max_length=512
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)  #  label2id=label2id, id2label=id2label, num_labels=len(label2id), num_labels=3

#print(model.config)


# In[ ]:


"""from transformers import DataCollatorWithPadding

dataset_hy_prem = datasets.Dataset.from_dict({"premise": df_cl.text_original[:500].tolist(), "hypothesis": df_cl.text_original[:500].tolist()})

def tokenize_func(examples):
  return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding='max_length', max_length=512, return_tensors="pt")  # max_length=512,  padding=True, padding='max_length'
dataset_hy_prem = dataset_hy_prem.map(tokenize_func, batched=True)  #batched=True
dataset_hy_prem = dataset_hy_prem.remove_columns(column_names=["premise", "hypothesis"])
dataset_hy_prem.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

# could not make dynamic padding during data loading work ...  #for dynamic padding during data loading, not during tokanization. should make inference faster.
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')  # https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorWithPadding
dataloader = torch.utils.data.DataLoader(dataset_hy_prem, batch_size=64, collate_fn=None)  # https://discuss.huggingface.co/t/are-dynamic-padding-and-smart-batching-in-the-library/10404

logits_lst_all = np.empty([0, len(model.config.id2label)])  # the last element is the number of output classes (e.g. 3 for NLI), the first number unclear
for batch in dataloader:
    inputs = {'input_ids': batch["input_ids"][0].to(device), 'attention_mask': batch["attention_mask"][0].to(device)}
    with torch.no_grad():  # for speed https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/25   
        outputs = model(**inputs)
    logits = outputs[0]
    # stack new logits from batch on logits from previous batches
    logits_lst_all = np.vstack([logits_lst_all, logits.detach().cpu().numpy()])

print(len(logits_lst_all))

prediction_lst_all = [torch.softmax(torch.from_numpy(logits), dim=-1).tolist() for logits in logits_lst_all]
print(len(prediction_lst_all))"""


# In[ ]:


#### function for inference on batched input
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm


def classify_nli_batched(premise_lst=None, hypotheses_lst=None, batch_size=80):

  hypothesis_lst_multi = hypotheses_lst * len(premise_lst)
  premise_lst_multi = [[premise] * len(hypotheses_lst) for premise in premise_lst]
  premise_lst_multi = [item for sublist in premise_lst_multi for item in sublist]
  #prem_hy_pair = [[premise, hypothesis] for premise, hypothesis in zip(premise_lst_multi, hypothesis_lst_multi)]
  
  # use datasets to avoid OOM
  dataset_hy_prem = datasets.Dataset.from_dict({"premise": premise_lst_multi, "hypothesis": hypothesis_lst_multi})

  print("Tokenization time: ")
  def tokenize_func(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding='max_length', max_length=512, return_tensors="pt")  # ! max_length padding not optimal. for some reason dynamic padding during data loading did not work
  dataset_hy_prem = dataset_hy_prem.map(tokenize_func)  #batched=True
  dataset_hy_prem = dataset_hy_prem.remove_columns(column_names=["premise", "hypothesis"])
  dataset_hy_prem.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

  # could not make dynamic padding during data loading work ...  #for dynamic padding during data loading, not during tokanization. should make inference faster.
  #data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')  # https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorWithPadding
  dataloader = torch.utils.data.DataLoader(dataset_hy_prem, batch_size=batch_size, collate_fn=None)  # https://discuss.huggingface.co/t/are-dynamic-padding-and-smart-batching-in-the-library/10404

  print("Inference time: ")
  logits_lst_all = np.empty([0, len(model.config.id2label)])  # the last element is the number of output classes (e.g. 3 for NLI), the first number unclear
  for batch in tqdm.notebook.tqdm(dataloader):
      inputs = {'input_ids': batch["input_ids"][0].to(device), 'attention_mask': batch["attention_mask"][0].to(device)}
      with torch.no_grad():  # for speed https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/25   
          outputs = model(**inputs)
      logits = outputs[0]
      # stack new logits from batch on logits from previous batches
      logits_lst_all = np.vstack([logits_lst_all, logits.detach().cpu().numpy()])

  ## softmax on logits
  # softmax on all 3 logits
  prediction_lst_all = [torch.softmax(torch.from_numpy(logits), dim=-1).tolist() for logits in logits_lst_all]
  # softmax only on entailment and contradiction logit
  prediction_lst_2 = [torch.softmax(torch.from_numpy(logits), dim=-1).tolist() for logits in logits_lst_all[:, [0, -1]]]

  probs_dic_all = []
  for i in range(len(prediction_lst_all)):
    probs_dic_all.append({#"entailment": prediction_lst_all[i][0], "not_entailment": prediction_lst_all[i][1],  # ! label binary model
                          "entailment": prediction_lst_all[i][0], "neutral": prediction_lst_all[i][1], "contradiction": prediction_lst_all[i][2],  # ! label order from ALBERT
                          #"entailment_soft3": prediction_lst_all[i][2], "contradiction_soft3": prediction_lst_all[i][0], "neutral_soft3": prediction_lst_all[i][1],  # ! label order from distilbert
                          #"entailment_soft2": prediction_lst_2[i][0], "contradiction_soft2": prediction_lst_2[i][1],
                          "hypothesis": hypothesis_lst_multi[i], "text": premise_lst_multi[i]})
  return probs_dic_all


#df_cl["text"] = 'The quote: "' + df_cl.text_preceding + ' ' + df_cl.text_original + '".'

#probs_dic = classify_nli_batched(premise_lst=df_cl.text.tolist(), hypotheses_lst=list(hypothesis_hyperparams_dic["template_quote_two_text_concat"].values()), batch_size=128)


# In[ ]:


## merge probs dic with with meta data from df_effec_eval - only difference is entailment scores are now from new model
df_nli = pd.DataFrame(probs_dic)
print(len(df_nli))

# harmonise hypothesis texts to enable merge. (only merging on sentences does not work due to duplicate sentences)
df_nli = df_nli.merge(df_cl, how="left", left_on=["text"], right_on=["text"])
df_nli = df_nli.drop(columns=["text", "year", "president", "pres_party", "id_original", "text_ext2"])

# selecting predicted label for corresponding hypo
hypothesis_dic = hypothesis_hyperparams_dic["template_quote_two_text_concat"]
df_nli["label_text_predicted"] = [list(hypothesis_dic.keys())[list(hypothesis_dic.values()).index(hypo)] for hypo in df_nli.hypothesis]


# In[ ]:


#compression_opts = dict(method='zip', archive_name='df_nli_3c.csv')  
#df_nli.to_csv("./datasets/CAP/df_nli_3c.zip", compression=compression_opts)


# #### Sampling code

# In[ ]:


df_nli = pd.read_csv("./datasets/CAP/df_nli_3c.zip", index_col="Unnamed: 0")
df_nli = df_nli.rename(columns={"premise": "text"})
#DataTable(df_nli, num_rows_per_page=5)


# In[ ]:


# remove texts in df_test from df_nli_train
df_nli_train = df_nli[~df_nli.text_original.isin(df_test.text_original)]
assert len(df_nli_train) == len(df_nli) - (len(df_test) * len(df_cl.label_text.unique()))


# In[ ]:


#### Smart sampling
import tqdm 
tqdm.notebook.tqdm.pandas(desc="Pandas apply")

# enables selecting texts/labels which model is certain about and then correct/reinforce it's zero-shot bias
# opens cool opportunities for model-in-the-loop training. could e.g. first zero-shot sample 50 per class, annotate, train model + evalutate, re-sample 100 with trained model 

## extract rows with two highest entailment prediction per text
df_nli_max_2 = df_nli_train.groupby(by="text", group_keys=False, as_index=False, sort=False).progress_apply(lambda x: x[x.entailment.isin(x.entailment.nlargest(2))])
print(len(df_nli_max_2))

## create now column indicating the max prediction for each premise
# facilitates smart samling below for low_n classes
premise_label_predicted_max_lst = []
for group_name, group_df in df_nli_max_2.groupby(by=["text"], group_keys=False, as_index=False, sort=False):
  text_label_pred_max = group_df[group_df.entailment == group_df.entailment.max()].label_text_predicted.tolist()[0]
  premise_label_predicted_max_lst = premise_label_predicted_max_lst + [text_label_pred_max] * len(group_df)

df_nli_max_2["premise_label_predicted_max"] = premise_label_predicted_max_lst


# In[ ]:


##### smart sampling function
## simplified version to avoid bugs
# ! resulting distribution does not necessarily seem to be better than random, lol (especially if I always add at least 3)
import tqdm 
tqdm.notebook.tqdm.pandas(desc="Pandas apply")

# sample most certain predictions based on absolute and relative certainty
# !! for iteration need above and below thresholds here !! Otherwise it resamples the same
certainty_minimum = [0.5]
certainty_distance_next = [0.10]

def conditioned_df_selection(x, certainty_interval=None):
  return x[(x.entailment == x.entailment.max()) & ((x.entailment.nlargest(2).iloc[0] - x.entailment.nlargest(2).iloc[1]) > certainty_distance_next[certainty_interval]) & (x.entailment.max() > certainty_minimum[certainty_interval])]

def smart_sampling_nli(df_nli_max_2=None, n_sample_per_class=None):
  ## take first best sample
  df_nli_max_best = df_nli_max_2.groupby(by="text", group_keys=False, as_index=False, sort=False).progress_apply(lambda x: conditioned_df_selection(x, certainty_interval=0))
  df_sample_smart = df_nli_max_best.groupby(by="hypothesis", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(min(len(x), n_sample_per_class_smart), random_state=SEED_GLOBAL))

  ## only maintain those predicted labels which correspond to cap gold label - simulates manual data cleaning
  df_sample_smart = df_sample_smart[df_sample_smart.label_text == df_sample_smart.label_text_predicted]

  # take fully random sample to fill rest
  n_to_fill = n_sample_per_class * len(df_nli_max_2.label_text.unique()) - len(df_sample_smart)
  step = 0
  while n_to_fill > 0:
    df_nli_train_deduplicated = df_nli_train[~df_nli_train.text.duplicated()][['text', 'label_cap2', 'label_cap4', 'label_cap2_text', 'label', 'label_text', 'text_original', 'text_preceding', 'doc_id']]  # ignoring irrelevant columns (all come from agriculture)
    df_sample_fill = df_nli_train_deduplicated.sample(n=n_to_fill, random_state=SEED_GLOBAL)  # can either sample from df_nli_train or df_nli_max_2
    df_sample_smart = pd.concat([df_sample_smart, df_sample_fill])
    df_sample_smart = df_sample_smart[~df_sample_smart.text.duplicated(keep='first')]
    n_to_fill = n_sample_per_class * len(df_nli_max_2.label_text.unique()) - len(df_sample_smart)
    step += 1
    if step == 10:
      break

  ## fill up to have at least 3 per class
  if sum(df_sample_smart.label_text.value_counts() >= 3) != len(df_cl.label_text.unique()):
    labels_count_insufficient_sample = df_sample_smart.label_text.value_counts().where(lambda x: x < 3).dropna()  # returns series with label_names and number of hits for predicted labels with less than 16 or n_sample_per_class hits
    # add labels and counts which don't appear a single time in first df_sample_smart
    label_missing = [label for label in df_cl.label_text.unique() if label not in df_sample_smart.label_text.unique()]
    label_missing = pd.Series([0] * len(label_missing), index=label_missing)
    labels_count_insufficient_sample = labels_count_insufficient_sample.append(label_missing)
    #print(f"For these label(s) {labels_count_insufficient_sample}, < 3 gold texts were sampled. Filling them up to 3 samples.")
    for index_label, value in labels_count_insufficient_sample.iteritems():
      df_sample_fill = df_nli_train[df_nli_train.label_text == index_label].sample(n=3-int(value), random_state=SEED_GLOBAL)
      df_sample_smart = pd.concat([df_sample_smart, df_sample_fill])

  return df_sample_smart

### random sample
def random_sample_fill(df_train=None, n_sample_per_class=None):
  df_sample_random = df_train.sample(n=n_sample_per_class_smart*len(df_train.label_text.unique()), random_state=SEED_GLOBAL)
  
  ## fill up to have at least 3 per class
  if sum(df_sample_random.label_text.value_counts() >= 3) != len(df_cl.label_text.unique()):
    labels_count_insufficient_sample = df_sample_random.label_text.value_counts().where(lambda x: x < 3).dropna()  # returns series with label_names and number of hits for predicted labels with less than 16 or n_sample_per_class hits
    # add labels and counts which don't appear a single time in first df_sample_random
    label_missing = [label for label in df_cl.label_text.unique() if label not in df_sample_random.label_text.unique()]
    label_missing = pd.Series([0] * len(label_missing), index=label_missing)
    labels_count_insufficient_sample = labels_count_insufficient_sample.append(label_missing)
    #print(f"For these label(s) {labels_count_insufficient_sample}, < 3 gold texts were sampled. Filling them up to 3 samples.")
    for index_label, value in labels_count_insufficient_sample.iteritems():
      df_sample_fill = df_train[df_train.label_text == index_label].sample(n=3-int(value), random_state=SEED_GLOBAL)
      df_sample_random = pd.concat([df_sample_random, df_sample_fill])

  return df_sample_random



# In[ ]:



n_sample_per_class_smart = 16

df_sample_smart = smart_sampling_nli(df_nli_max_2=df_nli_max_2, n_sample_per_class=n_sample_per_class_smart)

df_sample_random = random_sample_fill(df_train=df_train, n_sample_per_class=n_sample_per_class_smart)

df_sample_gold_v_pred = pd.DataFrame(data={"label_text_predicted": df_sample_smart.label_text_predicted.value_counts(), "label_text": df_sample_smart.label_text.value_counts(), "label_text_gold_overall": df_cl.label_text.value_counts(),
                                           "label_text_random": df_sample_random.label_text.value_counts()},
                                     index=df_cl.label_text.unique()).fillna(0).sort_values(by=["label_text", "label_text_predicted", "label_text_random"])
df_sample_gold_v_pred


# In[ ]:


DataTable(df_sample_random, num_rows_per_page=5)


# In[ ]:


# prediction accuracy zero-shot
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(df_sample_smart.label_text, df_sample_smart.label_text_predicted, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(df_sample_smart.label_text, df_sample_smart.label_text_predicted, average='micro')  # https://scikit-learn.org/stabl
print("Zero-shot metrics: ", {'f1_macro': f1_macro, 'f1_micro': f1_micro})

## sample smart metrics with iterative sampling
# MiniLM-L6-mnli-3c,  0shot
# n_sample_per_class_smart = 16  
# n_sample_per_class_smart = 32  {0.5153648655350834, 'f1_micro': 0.40540540540540543}



# In[ ]:


### screwded up sample code - cannot find the bugs
# code might be fine now - main flaw is now that the certainty intervals resample partly the same because no upper & lower bound combined
"""n_sample_per_class_smart = 32
import tqdm 
tqdm.notebook.tqdm.pandas(desc="Pandas apply")

## iteratively sample most certain predictions based on absolute and relative certainty
# two conditions to sample rows on. gradually make conditions easier
certainty_minimum = [0.5, 0.5, 0.6, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]  # should be at least 0.3 certain, otherwise fully random
certainty_distance_next = [0.12, 0.10, 0.07, 0.04, 0.02, 0.01, 0, 0, 0, 0]

def conditioned_df_selection(x, certainty_interval=None):
  return x[(x.entailment == x.entailment.max()) & ((x.entailment.nlargest(2).iloc[0] - x.entailment.nlargest(2).iloc[1]) > certainty_distance_next[certainty_interval]) & (x.entailment.max() > certainty_minimum[certainty_interval])]

df_nli_max_2=df_nli_max_2
n_sample_per_class=n_sample_per_class_smart
#def smart_sampling_nli(df_nli_max_2=None, n_sample_per_class=None):
certainty_interval = 0  # step integers for each iteration to select certainty conditions
## take first best sample
df_nli_max_best = df_nli_max_2.groupby(by="text", group_keys=False, as_index=False, sort=False).progress_apply(lambda x: conditioned_df_selection(x, certainty_interval=0))
df_sample_smart = df_nli_max_best.groupby(by="hypothesis", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(min(len(x), n_sample_per_class), random_state=SEED_GLOBAL))

## fill sample cases where best sampling condition did not yield enough
# While any of the label_predicted has less than n_sample_per_class (e.g. 16), sample more texts for the low_n classes and append them to main df until all label_text_predicted have n_sample_per_class
certainty_interval += 1
while sum(df_sample_smart.label_text_predicted.value_counts() >= n_sample_per_class) != len(df_cl.label_text.unique()):
  labels_count_insufficient_sample = df_sample_smart.label_text_predicted.value_counts().where(lambda x: x < n_sample_per_class).dropna()  # returns series with label_names and number of hits for predicted labels with less than 16 or n_sample_per_class hits
  # add labels and counts which don't appear a single time in first df_sample_smart
  label_missing = [label for label in df_cl.label_text.unique() if label not in df_sample_smart.label_text.unique()]
  label_missing = pd.Series([0] * len(label_missing), index=label_missing)
  labels_count_insufficient_sample = labels_count_insufficient_sample.append(label_missing)
  print(labels_count_insufficient_sample)

  for label_index, n_sample in labels_count_insufficient_sample.iteritems():
    # Select only those rows where the low_n class is entailment.max() among the two max predictions
    df_nli_max_2_label = df_nli_max_2[df_nli_max_2.premise_label_predicted_max == label_index]
    df_nli_max_2_label_best = df_nli_max_2_label.groupby(by="text", group_keys=False, as_index=False, sort=False).apply(lambda x: conditioned_df_selection(x, certainty_interval=certainty_interval))
    if len(df_nli_max_2_label_best) > 0:
      df_sample_fill = df_nli_max_2_label_best.sample(n=min(len(df_nli_max_2_label_best), n_sample_per_class-int(n_sample)), random_state=SEED_GLOBAL)
      df_sample_smart = pd.concat([df_sample_smart, df_sample_fill])
  # remove duplicates
  df_sample_smart = df_sample_smart[~df_sample_smart.text.duplicated(keep='first')]


  certainty_interval += 1
  print(certainty_interval)
  ## clean and fill
  if certainty_interval >= len(certainty_minimum):
    # remove texts which do not correspond to gold label. simulates manual correction of dataset  # keeping those where label predicted was different from gold was too adversarial and confused model
    df_sample_smart = df_sample_smart[df_sample_smart.label_text == df_sample_smart.label_text_predicted]
    # take fully random sample to fill rest
    n_to_fill = n_sample_per_class * len(df_nli_max_2.label_text.unique()) - len(df_sample_smart)
    while n_to_fill > 0:
      certainty_interval += 1
      print(certainty_interval)
      df_sample_fill = df_nli_train[~df_nli_train.text.duplicated()].sample(n=n_to_fill, random_state=SEED_GLOBAL)  # can either sample from df_nli_train or df_nli_max_2
      df_sample_smart = pd.concat([df_sample_smart, df_sample_fill])
      df_sample_smart = df_sample_smart[~df_sample_smart.text.duplicated(keep='first')]
      n_to_fill = n_sample_per_class * len(df_nli_max_2.label_text.unique()) - len(df_sample_smart)
      if certainty_interval >= 20:
        break
    break
  #print("Target sample size: ", n_sample_per_class * len(df_nli_max_2.label_text.unique()))
  #print("Actual sample size: ", len(df_sample_smart))

## adding at least 3 gold examples for each class
# realistic that researchers would add at least three classes, researchers would just write them manually and solves big headache for downstream training
if sum(df_sample_smart.label_text.value_counts() >= 3) != len(df_cl.label_text.unique()):
  labels_count_insufficient_sample = df_sample_smart.label_text.value_counts().where(lambda x: x < 3).dropna()  # returns series with label_names and number of hits for predicted labels with less than 16 or n_sample_per_class hits
  # add labels and counts which don't appear a single time in first df_sample_smart
  label_missing = [label for label in df_cl.label_text.unique() if label not in df_sample_smart.label_text.unique()]
  label_missing = pd.Series([0] * len(label_missing), index=label_missing)
  labels_count_insufficient_sample = labels_count_insufficient_sample.append(label_missing)
  #label_missing = [label for label in df_cl.label_text.unique() if label not in df_sample_smart.label_text.unique()]
  print(f"For these label(s) {labels_count_insufficient_sample}, < 3 gold texts were sampled. Filling them up to 3 samples.")
  for index_label, value in labels_count_insufficient_sample.iteritems():
    df_sample_fill = df_nli_train[df_nli_train.label_text == index_label].sample(n=3-int(value), random_state=SEED_GLOBAL)
    df_sample_smart = pd.concat([df_sample_smart, df_sample_fill])
  #if sum(df_sample_smart.label_text.value_counts() >= 3) != len(df_cl.label_text.unique()):
  #  raise Exception("There is probably a class for which no label_text (gold) is available in df_nli_train")

# replace NA
df_sample_smart["text_preceding"] = df_sample_smart.text_preceding.fillna('.')
df_sample_smart["text_original"] = df_sample_smart.text_original.fillna('.')
df_sample_smart["text"] = df_sample_smart.text.fillna('.')
  #return df_sample_smart

df_sample_gold_v_pred = pd.DataFrame(data={"label_text_predicted": df_sample_smart.label_text_predicted.value_counts(), "label_text_gold": df_sample_smart.label_text.value_counts(), "label_text_gold_overall": df_cl.label_text.value_counts()},
                                     index=df_cl.label_text.unique()).fillna(0).sort_values(by=["label_text_gold", "label_text_predicted"])
df_sample_gold_v_pred
"""


# ## Training pipeline

# In[ ]:


### import local functions
from google.colab import drive
import sys, os
drive.mount('/content/drive', force_remount=False)
#drive.flush_and_unmount()
print(os.getcwd())
os.chdir("/content/drive/My Drive/Colab Notebooks/")
print(os.getcwd())

sys.path.append("/content/drive/My Drive/Colab Notebooks/NLI-experiments")

import few_shot_func
import importlib  # in case of manual updates in .py file
importlib.reload(few_shot_func)

from few_shot_func import custom_train_test_split, custom_train_test_split_sent_overlapp, format_nli_testset, format_nli_trainset
from few_shot_func import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer
from few_shot_func import compute_metrics_standard, compute_metrics_nli_binary, clean_memory, aug_back_translation

from sklearn.model_selection import train_test_split
np.random.seed(SEED_GLOBAL)


# In[ ]:


### random sample
def random_sample_fill(df_train=None, n_sample_per_class=None, seed=None):
  df_sample_random = df_train.sample(n=n_sample_per_class*len(df_train.label_text.unique()), random_state=seed)
  if len(df_sample_random) == 0:  # for zero-shot
    return df_sample_random 

  ## fill up to have at least 3 per class
  if sum(df_sample_random.label_text.value_counts() >= 3) != len(df_cl.label_text.unique()):
    labels_count_insufficient_sample = df_sample_random.label_text.value_counts().where(lambda x: x < 3).dropna()  # returns series with label_names and number of hits for predicted labels with less than 16 or n_sample_per_class hits
    # add labels and counts which don't appear a single time in first df_sample_random
    label_missing = [label for label in df_cl.label_text.unique() if label not in df_sample_random.label_text.unique()]
    label_missing = pd.Series([0] * len(label_missing), index=label_missing)
    labels_count_insufficient_sample = labels_count_insufficient_sample.append(label_missing)
    #print(f"For these label(s) {labels_count_insufficient_sample}, < 3 gold texts were sampled. Filling them up to 3 samples.")
    for index_label, value in labels_count_insufficient_sample.iteritems():
      df_sample_fill = df_train[df_train.label_text == index_label].sample(n=3-int(value), random_state=seed)
      df_sample_random = pd.concat([df_sample_random, df_sample_fill])
  return df_sample_random


# In[ ]:


# define fotmat_text function further up together with hyptheses for more flexibility
"""def format_text(df=None, text_format=None):
  if text_format == 'template_quote_single_text':
    df["text"] = 'The quote: "' + df.text_original + '".'
  elif text_format == 'template_quote_two_text_concat':
    df["text"] = 'The quote: "' + df.text_preceding + ' ' + df.text_original + '".'
  elif text_format == 'template_quote_two_text_separate':
    df["text"] = 'The first quote: "' + df.text_preceding + '". The second quote: "' + df.text_original + '".'
  elif text_format == 'template_quote_three_text_separate':
    df["text"] = df.text_preceding + '. - The quote: "' + df.text_original + '". - ' + df.text_following
  elif text_format == 'template_two_text':
    df["text"] = df.text_preceding + " " + df.text_original
  else:
    raise Exception('Hypothesis template not found.')
  return df"""


# #### Hyperparameter tuning

# In[ ]:


# debugging in colab https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.06-Errors-and-Debugging.ipynb#scrollTo=TF-MxTMBdGEH
#import pdb; pdb.set_trace()

import optuna
#import warnings
#transformers.logging.set_verbosity_warning()  # https://huggingface.co/transformers/main_classes/logging.html
#warnings.filterwarnings(action='ignore')

METHOD = "nli"  # "standard_dl", "nli", "nsp"
N_MAX_SAMPLE_PER_CLASS_DEV = ["fixed"] # [16, 32, 64, 128]
TRAINING_DIRECTORY = "nli-few-shot/pimpo"
LABEL_TEXT_ALPHABETICAL = np.sort(df_cl.label_text.unique())
#HYPOTHESIS_TYPE = hypo_label_dic_short  # hypo_label_dic_short , hypo_label_dic_long, hypo_label_dic_short_subcat
DISABLE_TQDM = False
CROSS_VALIDATION_REPETITIONS = 2
#HYPOTHESIS_TEMPLATE = "template_quote_three_text_separate"  # trial.suggest_categorical("hypothesis_templatee", list(hypothesis_hyperparams_dic.keys()))

#MODEL_NAME = "./results/nli-few-shot/mnli-2c/MiniLM-L6-mnli-binary/" # "textattack/bert-base-uncased-MNLI"  # "./results/nli-few-shot/MiniLM-L6-mnli-binary/",   # './results/nli-few-shot/all-nli-2c/MiniLM-L6-allnli-2c-v1'
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" #"MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"    # "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",    # "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
#MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
#MODEL_NAME = "bert-base-uncased"  # "google/fnet-base" ,  "bert-base-uncased" , "bert-large-uncased" , "google/mobilebert-uncased"

# FP16? if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa" in MODEL_NAME: fp16_bool = False  # mDeBERTa does not support FP16 yet
  

def data_preparation(random_seed=None, hypothesis_template=None, hypo_label_dic=None, n_sample_per_class=None): # df_train=None
  ## unrealistic oracle sample
  #df_train_samp = df_train.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), n_sample_per_class), random_state=random_seed))
  ## smart sampling
  #df_train_samp = smart_sampling_nli(df_nli_max_2=df_nli_max_2, n_sample_per_class=n_sample_per_class)
  #df_train_samp = df_train_samp[["label", "label_text", "text_original", "text_preceding"]]  # maybe helps avoiding downstream errors?
  ## random sampling
  if n_sample_per_class == "fixed":
    df_train_samp = df_train
  else:
    df_train_samp = random_sample_fill(df_train=df_train, n_sample_per_class=n_sample_per_class, seed=random_seed)

  # chose the text format depending on hyperparams
  df_train_samp = format_text(df=df_train_samp, text_format=hypothesis_template)

  # 50% split as recommended by https://arxiv.org/pdf/2109.12742.pdf
  df_train_samp, df_dev_samp = train_test_split(df_train_samp, test_size=0.50, shuffle=True, random_state=random_seed)

  # format train and dev set for NLI etc.
  df_train_samp = format_nli_trainset(df_train_samp=df_train_samp, hypo_label_dic=hypo_label_dic, method=METHOD)  # hypo_label_dic_short , hypo_label_dic_long
  df_dev_samp = format_nli_testset(df_test=df_dev_samp, hypo_label_dic=hypo_label_dic, method=METHOD)  # hypo_label_dic_short , hypo_label_dic_long
  return df_train_samp, df_dev_samp


def inference_run(df_train=None, df_dev=None, random_seed=None, hyperparams_dic=None, n_sample_per_class=None):
  clean_memory()
  model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
  encoded_dataset = tokenize_datasets(df_train_samp=df_train, df_test=df_dev, tokenizer=tokenizer, method=METHOD, max_length=None)

  train_args = set_train_args(hyperparams_dic=hyperparams_dic, training_directory=TRAINING_DIRECTORY, disable_tqdm=DISABLE_TQDM, evaluation_strategy="no", fp16=fp16_bool) 
  trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=encoded_dataset, train_args=train_args, 
                           method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
  clean_memory()

  if n_sample_per_class != 0:
    trainer.train()
  results = trainer.evaluate()  # eval_dataset=encoded_dataset_test
  # dataset statistics
  #dataset_stats_dic = {"share_traing_samp_to_full": round((len(df_train_samp) / len(df_train)) * 100, 2), "n_training_samp": len(df_train_samp), "n_train_full": len(df_train)}

  run_info_dic = {"method": METHOD, "n_sample_per_class": n_sample_per_class, "model": MODEL_NAME, "results": results, "hyper_params": hyperparams_dic}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic
  #transformers.logging.set_verbosity_warning()  # https://huggingface.co/transformers/main_classes/logging.html
  return run_info_dic


def optuna_objective(trial, hypothesis_hyperparams_dic=None, n_sample_per_class=None):  #df_train=None,
  np.random.seed(SEED_GLOBAL)  # don't understand why this needs to be run here at each iteration. it should stay constant once set globally?!
  hyperparams = {
    #"warmup_ratio": 0.06,  # hf default: 0  # FB paper uses 0.0
    "lr_scheduler_type": "constant",  # hf default: linear, not constant.  FB paper also uses constant 
    "learning_rate": trial.suggest_float("learning_rate", 1e-7, 9e-5, log=False),
    #"learning_rate": trial.suggest_categorical("learning_rate", [5e-6, 1e-5, 5e-5, 9e-5, 5e-4], log=False),
    #"num_train_epochs": trial.suggest_categorical("num_train_epochs", [5, 10]),
    "num_train_epochs": 5,
    #"seed": trial.suggest_categorical("seed_hf_trainer", np.random.choice(range(43), size=5).tolist() ),  # for training order of examples in hf trainer. For standard_dl, this also influences head
    "seed": SEED_GLOBAL,
    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),  # mDeBERTa cannot handle 48+ batch, with all languages it also crashes with 32 batch (but performance on lower batches seems similar)
    #"per_device_train_batch_size": 32,
    "warmup_ratio": 0.06,
    "weight_decay": 0.05,
    "per_device_eval_batch_size": 128,  # increase eval speed
  }
  hypothesis_template = trial.suggest_categorical("hypothesis_template", list(hypothesis_hyperparams_dic.keys()))
  
  hyperparams_optuna = dict(**hyperparams, **{"hypothesis_template": hypothesis_template})
  trial.set_user_attr("hyperparameters_all", hyperparams_optuna)
  print("Hyperparameters for this run: ", hyperparams_optuna)

  # cross-validation loop. Objective: determine F1_macro for specific sample for specific hyperparams, without a test set
  run_info_dic_lst = []
  for step_i, random_seed_cross_val in enumerate(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS)):
    df_train_samp, df_dev_samp = data_preparation(random_seed=random_seed_cross_val, #df_train=df_train
                                                  hypothesis_template=hypothesis_template, 
                                                  hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template], 
                                                  n_sample_per_class=n_sample_per_class)
    #import pdb; pdb.set_trace()
    run_info_dic = inference_run(df_train=df_train_samp, df_dev=df_dev_samp, hyperparams_dic=hyperparams, n_sample_per_class=n_sample_per_class)
    run_info_dic_lst.append(run_info_dic)
    
    # Report intermediate objective value.
    intermediate_value = (run_info_dic["results"]["eval_f1_macro"] + run_info_dic["results"]["eval_f1_micro"]) / 2
    trial.report(intermediate_value, step_i)
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
      raise optuna.TrialPruned()

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

#optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0, interval_steps=1, n_min_trials=5)  # https://optuna.readthedocs.io/en/stable/reference/pruners.html
#optuna_sampler = optuna.samplers.TPESampler(seed=SEED_GLOBAL, consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False, n_startup_trials=8, n_ei_candidates=24, multivariate=False, group=False, warn_independent_sampling=True, constant_liar=False)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
#study = optuna.create_study(direction="maximize", study_name=None, pruner=optuna_pruner, sampler=optuna_sampler)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
#study.optimize(lambda trial: optuna_objective(trial, df_train=df_train, hypothesis_hyperparams_dic=hypothesis_hyperparams_dic), 
#               n_trials=10, show_progress_bar=True)  # Objective function with additional arguments https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments

#warnings.filterwarnings(action='default')


# In[ ]:


#warnings.filterwarnings(action='ignore')

def run_study(n_sample_per_class=None):
  optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0, interval_steps=1, n_min_trials=5)  # https://optuna.readthedocs.io/en/stable/reference/pruners.html
  optuna_sampler = optuna.samplers.TPESampler(seed=SEED_GLOBAL, consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False, 
                                              n_startup_trials=5, n_ei_candidates=24, multivariate=False, group=False, warn_independent_sampling=True, constant_liar=False)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
  study = optuna.create_study(direction="maximize", study_name=None, pruner=optuna_pruner, sampler=optuna_sampler)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html

  #study.enqueue_trial({'learning_rate': 0.00045082186929817176, 'num_train_epochs': 15, 'seed_hf_trainer': 38, 'per_device_train_batch_size': 8, 'hypothesis_template': 'template_quote'})
  study.optimize(lambda trial: optuna_objective(trial, hypothesis_hyperparams_dic=hypothesis_hyperparams_dic, n_sample_per_class=n_sample_per_class),   #df_train=df_train,
                n_trials=12, show_progress_bar=True)  # Objective function with additional arguments https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
  return study

study_lst = [run_study(n_sample_per_class=n_sample) for n_sample in tqdm.notebook.tqdm(N_MAX_SAMPLE_PER_CLASS_DEV)]

#warnings.filterwarnings(action='default')


# In[ ]:


#best_params_lst = [study.best_params for study in study_lst]
for n_samples, study in zip(N_MAX_SAMPLE_PER_CLASS_DEV, study_lst):
  print(f"Study with max {n_samples} samples per class:")
  print("Best hyperparameters: ", study.best_params)
  print("Best hyperparameters all: ", study.best_trial.user_attrs["hyperparameters_all"])
  print("Best performance: ", study.best_value)
  print("Best performance details: ", study.best_trial.user_attrs["metric_details"])
  print("Best trial full info: ", study.best_trial, "\n")
  #print(study.trials)


# In[ ]:


#Study with max fixed samples per class from gold set:
#Best hyperparameters:  {'learning_rate': 0.0002359717178299618, 'per_device_train_batch_size': 64}
#Best hyperparameters all:  {'lr_scheduler_type': 'constant', 'learning_rate': 0.0002359717178299618, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 64, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}
#Best performance:  0.4140461707040142
#Best performance details:  {'F1_macro_mean': 0.3108509620976836, 'F1_micro_mean': 0.5172413793103449, 'F1_macro_std': 0.02228232624887797, 'F1_micro_std': 0.02438299245470853}
#Best trial full info:  FrozenTrial(number=13, values=[0.4140461707040142], datetime_start=datetime.datetime(2022, 1, 4, 11, 23, 2, 945125), datetime_complete=datetime.datetime(2022, 1, 4, 11, 23, 10, 514951), params={'learning_rate': 0.0002359717178299618, 'per_device_train_batch_size': 64}, distributions={'learning_rate': UniformDistribution(high=0.0005, low=5e-06), 'per_device_train_batch_size': CategoricalDistribution(choices=(8, 32, 64))}, user_attrs={'hyperparameters_all': {'lr_scheduler_type': 'constant', 'learning_rate': 0.0002359717178299618, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 64, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}, 'metric_details': {'F1_macro_mean': 0.3108509620976836, 'F1_micro_mean': 0.5172413793103449, 'F1_macro_std': 0.02228232624887797, 'F1_micro_std': 0.02438299245470853}, 'results_trainer': [{'eval_loss': 0.6527978777885437, 'eval_f1_macro': 0.31176158457151637, 'eval_f1_micro': 0.5, 'eval_accuracy_balanced': 0.5369883040935672, 'eval_accuracy_not_b': 0.5, 'eval_precision_macro': 0.3761756876663709, 'eval_recall_macro': 0.4602756892230576, 'eval_precision_micro': 0.5, 'eval_recall_micro': 0.5, 'eval_label_gold_raw': [6, 2, 5, 6, 4, 5, 2, 6, 6, 1, 2, 6, 6, 1, 5, 6, 6, 1, 5, 5, 2, 6, 5, 5, 5, 6, 2, 6, 5, 6, 5, 2, 1, 6, 6, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 5, 2, 1, 5, 6, 2, 6, 5, 2, 0, 6, 2, 6], 'eval_label_predicted_raw': [6, 6, 1, 6, 4, 6, 1, 6, 6, 1, 1, 6, 6, 1, 3, 6, 6, 1, 2, 1, 1, 6, 5, 6, 4, 4, 3, 6, 2, 6, 2, 1, 1, 6, 6, 2, 3, 6, 6, 2, 3, 4, 6, 1, 2, 5, 1, 1, 2, 6, 1, 6, 2, 0, 2, 6, 2, 6], 'eval_runtime': 0.2488, 'eval_samples_per_second': 1631.856, 'eval_steps_per_second': 16.077, 'epoch': 5.0}, {'eval_loss': 0.6226587295532227, 'eval_f1_macro': 0.28311688311688316, 'eval_f1_micro': 0.5, 'eval_accuracy_balanced': 0.29723748473748474, 'eval_accuracy_not_b': 0.5, 'eval_precision_macro': 0.34608843537414963, 'eval_recall_macro': 0.25477498691784406, 'eval_precision_micro': 0.5, 'eval_recall_micro': 0.5, 'eval_label_gold_raw': [1, 6, 6, 5, 6, 2, 6, 5, 2, 4, 6, 6, 6, 6, 2, 3, 6, 6, 1, 2, 2, 5, 2, 6, 2, 5, 1, 6, 3, 2, 1, 5, 2, 2, 5, 2, 1, 6, 6, 4, 6, 5, 5, 5, 1, 2, 2, 5, 6, 1, 6, 6, 6, 5, 6, 5, 1, 6], 'eval_label_predicted_raw': [4, 6, 6, 2, 6, 6, 6, 6, 1, 0, 6, 6, 6, 6, 4, 5, 6, 6, 1, 2, 5, 5, 2, 6, 2, 3, 0, 4, 0, 4, 6, 4, 4, 2, 5, 5, 4, 6, 3, 6, 6, 1, 4, 5, 6, 4, 2, 5, 6, 6, 6, 6, 6, 2, 6, 5, 3, 0], 'eval_runtime': 0.2479, 'eval_samples_per_second': 1637.902, 'eval_steps_per_second': 16.137, 'epoch': 5.0}, {'eval_loss': 0.9034287929534912, 'eval_f1_macro': 0.3376744186046512, 'eval_f1_micro': 0.5517241379310345, 'eval_accuracy_balanced': 0.35373376623376623, 'eval_accuracy_not_b': 0.5517241379310345, 'eval_precision_macro': 0.3419501133786848, 'eval_recall_macro': 0.35373376623376623, 'eval_precision_micro': 0.5517241379310345, 'eval_recall_micro': 0.5517241379310345, 'eval_label_gold_raw': [5, 5, 6, 5, 6, 5, 4, 6, 2, 2, 1, 5, 2, 6, 6, 6, 6, 6, 3, 1, 6, 6, 3, 6, 5, 0, 2, 2, 5, 6, 2, 5, 5, 1, 5, 5, 6, 6, 6, 6, 5, 6, 6, 5, 1, 2, 6, 6, 2, 1, 5, 5, 4, 2, 6, 2, 6, 5], 'eval_label_predicted_raw': [1, 5, 4, 5, 6, 3, 4, 6, 2, 2, 2, 4, 5, 6, 6, 6, 6, 6, 5, 2, 6, 6, 5, 6, 5, 2, 1, 1, 2, 6, 2, 5, 5, 0, 3, 2, 5, 6, 3, 6, 6, 6, 6, 2, 1, 2, 6, 6, 2, 6, 2, 2, 3, 2, 6, 1, 6, 2], 'eval_runtime': 0.2239, 'eval_samples_per_second': 1813.23, 'eval_steps_per_second': 17.864, 'epoch': 5.0}]}, system_attrs={}, intermediate_values={0: 0.4058807922857582, 1: 0.39155844155844155, 2: 0.44469927826784283}, trial_id=13, state=TrialState.COMPLETE, value=None) 

#Study with max 20 max samples per class, single quote.
#Best hyperparameters:  {'learning_rate': 0.00014805742303997563, 'per_device_train_batch_size': 64}
#Best hyperparameters all:  {'lr_scheduler_type': 'constant', 'learning_rate': 0.00014805742303997563, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 64, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}
#Best performance:  0.1843098136288516
#Best performance details:  {'F1_macro_mean': 0.17321732840712847, 'F1_micro_mean': 0.19540229885057472, 'F1_macro_std': 0.026024674689231975, 'F1_micro_std': 0.03542766668372975}
#Best trial full info:  FrozenTrial(number=14, values=[0.1843098136288516], datetime_start=datetime.datetime(2022, 1, 4, 12, 6, 41, 274387), datetime_complete=datetime.datetime(2022, 1, 4, 12, 6, 49, 71059), params={'learning_rate': 0.00014805742303997563, 'per_device_train_batch_size': 64}, distributions={'learning_rate': UniformDistribution(high=0.0005, low=5e-06), 'per_device_train_batch_size': CategoricalDistribution(choices=(8, 32, 64))}, user_attrs={'hyperparameters_all': {'lr_scheduler_type': 'constant', 'learning_rate': 0.00014805742303997563, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 64, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}, 'metric_details': {'F1_macro_mean': 0.17321732840712847, 'F1_micro_mean': 0.19540229885057472, 'F1_macro_std': 0.026024674689231975, 'F1_micro_std': 0.03542766668372975}, 'results_trainer': [{'eval_loss': 0.7547670602798462, 'eval_f1_macro': 0.20826299750669497, 'eval_f1_micro': 0.2413793103448276, 'eval_accuracy_balanced': 0.1896103896103896, 'eval_accuracy_not_b': 0.2413793103448276, 'eval_precision_macro': 0.3242630385487528, 'eval_recall_macro': 0.1896103896103896, 'eval_precision_micro': 0.2413793103448276, 'eval_recall_micro': 0.2413793103448276, 'eval_label_gold_raw': [6, 2, 0, 5, 1, 0, 0, 1, 5, 2, 0, 2, 2, 2, 6, 6, 5, 5, 0, 2, 0, 1, 6, 0, 6, 1, 1, 5, 1, 3, 1, 0, 6, 4, 4, 5, 4, 5, 5, 2, 6, 1, 0, 3, 6, 2, 0, 1, 2, 5, 2, 5, 2, 0, 6, 6, 1, 5], 'eval_label_predicted_raw': [6, 2, 3, 5, 6, 4, 4, 6, 5, 6, 4, 6, 5, 2, 2, 1, 5, 2, 4, 6, 3, 1, 6, 0, 6, 1, 2, 6, 6, 5, 2, 6, 6, 6, 6, 4, 2, 4, 2, 2, 2, 6, 6, 6, 2, 3, 3, 4, 4, 5, 4, 4, 6, 2, 2, 2, 3, 0], 'eval_runtime': 0.2259, 'eval_samples_per_second': 1797.07, 'eval_steps_per_second': 17.705, 'epoch': 5.0}, {'eval_loss': 0.6763224005699158, 'eval_f1_macro': 0.14595917942773926, 'eval_f1_micro': 0.15517241379310345, 'eval_accuracy_balanced': 0.13522588522588522, 'eval_accuracy_not_b': 0.15517241379310345, 'eval_precision_macro': 0.17913832199546484, 'eval_recall_macro': 0.13522588522588522, 'eval_precision_micro': 0.15517241379310345, 'eval_recall_micro': 0.15517241379310345, 'eval_label_gold_raw': [6, 4, 5, 2, 5, 2, 5, 0, 0, 1, 5, 1, 1, 3, 2, 4, 5, 1, 2, 6, 0, 1, 0, 5, 1, 2, 2, 5, 4, 4, 5, 6, 0, 0, 2, 5, 0, 2, 5, 4, 1, 2, 1, 2, 2, 0, 5, 1, 4, 2, 4, 6, 6, 2, 4, 6, 5, 5], 'eval_label_predicted_raw': [4, 3, 2, 3, 1, 6, 5, 0, 5, 4, 2, 3, 1, 5, 4, 5, 4, 3, 3, 3, 6, 4, 3, 6, 6, 0, 5, 2, 3, 2, 6, 2, 5, 4, 5, 5, 1, 5, 3, 5, 3, 4, 1, 5, 6, 3, 2, 5, 4, 5, 3, 6, 4, 0, 5, 0, 5, 5], 'eval_runtime': 0.2393, 'eval_samples_per_second': 1696.409, 'eval_steps_per_second': 16.713, 'epoch': 5.0}, {'eval_loss': 0.9227601885795593, 'eval_f1_macro': 0.16542980828695114, 'eval_f1_micro': 0.1896551724137931, 'eval_accuracy_balanced': 0.17128942486085344, 'eval_accuracy_not_b': 0.1896551724137931, 'eval_precision_macro': 0.20303506017791734, 'eval_recall_macro': 0.17128942486085344, 'eval_precision_micro': 0.1896551724137931, 'eval_recall_micro': 0.1896551724137931, 'eval_label_gold_raw': [1, 1, 5, 1, 6, 2, 6, 5, 0, 0, 2, 1, 6, 4, 1, 4, 5, 5, 4, 1, 4, 5, 4, 4, 6, 6, 0, 0, 2, 5, 2, 1, 0, 0, 6, 6, 1, 6, 5, 5, 0, 2, 3, 5, 2, 5, 4, 5, 1, 6, 2, 0, 1, 0, 6, 0, 2, 0], 'eval_label_predicted_raw': [4, 6, 5, 5, 3, 4, 3, 0, 3, 5, 5, 3, 5, 4, 3, 4, 2, 5, 3, 3, 3, 6, 4, 5, 6, 5, 3, 2, 3, 5, 5, 3, 6, 3, 4, 5, 3, 4, 3, 5, 4, 3, 4, 0, 4, 5, 3, 3, 4, 0, 3, 3, 3, 4, 5, 0, 2, 4], 'eval_runtime': 0.2256, 'eval_samples_per_second': 1799.68, 'eval_steps_per_second': 17.731, 'epoch': 5.0}]}, system_attrs={}, intermediate_values={0: 0.2248211539257613, 1: 0.15056579661042135, 2: 0.17754249035037212}, trial_id=14, state=TrialState.COMPLETE, value=None) 

#Study with max 20 max samples per class, three texts.
Study with max fixed samples per class:
Best hyperparameters:  {'learning_rate': 8.929102038940433e-05, 'per_device_train_batch_size': 64}
Best hyperparameters all:  {'lr_scheduler_type': 'constant', 'learning_rate': 8.929102038940433e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 64, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}
Best performance:  0.25701951581595794
Best performance details:  {'F1_macro_mean': 0.232429836229617, 'F1_micro_mean': 0.28160919540229884, 'F1_macro_std': 0.017807276535169263, 'F1_micro_std': 0.03542766668372974}
Best trial full info:  FrozenTrial(number=10, values=[0.25701951581595794], datetime_start=datetime.datetime(2022, 1, 4, 12, 24, 33, 943285), datetime_complete=datetime.datetime(2022, 1, 4, 12, 24, 44, 257252), params={'learning_rate': 8.929102038940433e-05, 'per_device_train_batch_size': 64}, distributions={'learning_rate': UniformDistribution(high=0.0005, low=5e-06), 'per_device_train_batch_size': CategoricalDistribution(choices=(8, 32, 64))}, user_attrs={'hyperparameters_all': {'lr_scheduler_type': 'constant', 'learning_rate': 8.929102038940433e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 64, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}, 'metric_details': {'F1_macro_mean': 0.232429836229617, 'F1_micro_mean': 0.28160919540229884, 'F1_macro_std': 0.017807276535169263, 'F1_micro_std': 0.03542766668372974}, 'results_trainer': [{'eval_loss': 0.8935056924819946, 'eval_f1_macro': 0.24313725490196075, 'eval_f1_micro': 0.3275862068965517, 'eval_accuracy_balanced': 0.32510822510822507, 'eval_accuracy_not_b': 0.3275862068965517, 'eval_precision_macro': 0.2251733229176838, 'eval_recall_macro': 0.32510822510822507, 'eval_precision_micro': 0.3275862068965517, 'eval_recall_micro': 0.3275862068965517, 'eval_label_gold_raw': [6, 2, 0, 5, 1, 0, 0, 1, 5, 2, 0, 2, 2, 2, 6, 6, 5, 5, 0, 2, 0, 1, 6, 0, 6, 1, 1, 5, 1, 3, 1, 0, 6, 4, 4, 5, 4, 5, 5, 2, 6, 1, 0, 3, 6, 2, 0, 1, 2, 5, 2, 5, 2, 0, 6, 6, 1, 5], 'eval_label_predicted_raw': [6, 2, 5, 3, 2, 4, 6, 4, 5, 3, 6, 4, 2, 2, 6, 6, 5, 5, 6, 4, 4, 3, 4, 2, 6, 4, 2, 0, 3, 5, 2, 4, 6, 4, 4, 3, 6, 5, 5, 5, 5, 4, 5, 4, 6, 5, 5, 3, 5, 5, 4, 4, 5, 5, 6, 5, 5, 5], 'eval_runtime': 0.4088, 'eval_samples_per_second': 993.193, 'eval_steps_per_second': 9.785, 'epoch': 5.0}, {'eval_loss': 0.7136803865432739, 'eval_f1_macro': 0.24681598594642074, 'eval_f1_micro': 0.27586206896551724, 'eval_accuracy_balanced': 0.4021672771672772, 'eval_accuracy_not_b': 0.27586206896551724, 'eval_precision_macro': 0.2630325814536341, 'eval_recall_macro': 0.4021672771672772, 'eval_precision_micro': 0.27586206896551724, 'eval_recall_micro': 0.27586206896551724, 'eval_label_gold_raw': [6, 4, 5, 2, 5, 2, 5, 0, 0, 1, 5, 1, 1, 3, 2, 4, 5, 1, 2, 6, 0, 1, 0, 5, 1, 2, 2, 5, 4, 4, 5, 6, 0, 0, 2, 5, 0, 2, 5, 4, 1, 2, 1, 2, 2, 0, 5, 1, 4, 2, 4, 6, 6, 2, 4, 6, 5, 5], 'eval_label_predicted_raw': [6, 4, 5, 4, 4, 3, 4, 3, 5, 4, 0, 6, 1, 3, 4, 3, 4, 3, 6, 6, 6, 4, 3, 6, 5, 4, 5, 3, 6, 1, 5, 4, 4, 4, 0, 3, 4, 3, 5, 6, 4, 6, 1, 5, 4, 3, 5, 4, 4, 0, 4, 6, 6, 4, 5, 6, 5, 0], 'eval_runtime': 0.3803, 'eval_samples_per_second': 1067.709, 'eval_steps_per_second': 10.519, 'epoch': 5.0}, {'eval_loss': 0.6689786911010742, 'eval_f1_macro': 0.20733626784046952, 'eval_f1_micro': 0.2413793103448276, 'eval_accuracy_balanced': 0.21674397031539888, 'eval_accuracy_not_b': 0.2413793103448276, 'eval_precision_macro': 0.23809523809523808, 'eval_recall_macro': 0.21674397031539888, 'eval_precision_micro': 0.2413793103448276, 'eval_recall_micro': 0.2413793103448276, 'eval_label_gold_raw': [1, 1, 5, 1, 6, 2, 6, 5, 0, 0, 2, 1, 6, 4, 1, 4, 5, 5, 4, 1, 4, 5, 4, 4, 6, 6, 0, 0, 2, 5, 2, 1, 0, 0, 6, 6, 1, 6, 5, 5, 0, 2, 3, 5, 2, 5, 4, 5, 1, 6, 2, 0, 1, 0, 6, 0, 2, 0], 'eval_label_predicted_raw': [2, 4, 4, 2, 6, 4, 3, 3, 3, 3, 5, 4, 6, 4, 4, 4, 3, 3, 0, 4, 4, 4, 3, 5, 6, 6, 6, 2, 4, 5, 0, 6, 5, 4, 3, 6, 4, 3, 4, 4, 4, 4, 2, 4, 2, 5, 6, 5, 2, 6, 0, 3, 4, 4, 3, 0, 6, 3], 'eval_runtime': 0.3866, 'eval_samples_per_second': 1050.316, 'eval_steps_per_second': 10.348, 'epoch': 5.0}]}, system_attrs={}, intermediate_values={0: 0.2853617308992562, 1: 0.261339027455969, 2: 0.22435778909264856}, trial_id=10, state=TrialState.COMPLETE, value=None) 

#Study with max 20 max samples per class, three texts, deberta-v3-mnli-fever-anli
Study with max fixed samples per class:
Best hyperparameters:  {'learning_rate': 1.835950920317013e-05, 'per_device_train_batch_size': 32}
Best hyperparameters all:  {'lr_scheduler_type': 'constant', 'learning_rate': 1.835950920317013e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}
Best performance:  0.44289490979662705
Best performance details:  {'F1_macro_mean': 0.38578981959325404, 'F1_micro_mean': 0.5, 'F1_macro_std': 0.026935729008836777, 'F1_micro_std': 0.03724563619774631}
Best trial full info:  FrozenTrial(number=9, values=[0.44289490979662705], datetime_start=datetime.datetime(2022, 1, 4, 12, 40, 10, 304474), datetime_complete=datetime.datetime(2022, 1, 4, 12, 41, 4, 119669), params={'learning_rate': 1.835950920317013e-05, 'per_device_train_batch_size': 32}, distributions={'learning_rate': UniformDistribution(high=0.0005, low=5e-06), 'per_device_train_batch_size': CategoricalDistribution(choices=(8, 32, 64))}, user_attrs={'hyperparameters_all': {'lr_scheduler_type': 'constant', 'learning_rate': 1.835950920317013e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128}, 'metric_details': {'F1_macro_mean': 0.38578981959325404, 'F1_micro_mean': 0.5, 'F1_macro_std': 0.026935729008836777, 'F1_micro_std': 0.03724563619774631}, 'results_trainer': [{'eval_loss': 0.6485217809677124, 'eval_f1_macro': 0.3770551698940446, 'eval_f1_micro': 0.5172413793103449, 'eval_accuracy_balanced': 0.4090909090909091, 'eval_accuracy_not_b': 0.5172413793103449, 'eval_precision_macro': 0.35727018079959255, 'eval_recall_macro': 0.4090909090909091, 'eval_precision_micro': 0.5172413793103449, 'eval_recall_micro': 0.5172413793103449, 'eval_label_gold_raw': [6, 2, 0, 5, 1, 0, 0, 1, 5, 2, 0, 2, 2, 2, 6, 6, 5, 5, 0, 2, 0, 1, 6, 0, 6, 1, 1, 5, 1, 3, 1, 0, 6, 4, 4, 5, 4, 5, 5, 2, 6, 1, 0, 3, 6, 2, 0, 1, 2, 5, 2, 5, 2, 0, 6, 6, 1, 5], 'eval_label_predicted_raw': [6, 2, 5, 5, 1, 0, 2, 1, 5, 5, 2, 2, 0, 2, 6, 6, 5, 5, 1, 2, 2, 0, 6, 1, 6, 6, 1, 5, 0, 5, 1, 1, 6, 1, 0, 3, 1, 2, 5, 5, 6, 1, 5, 0, 5, 2, 2, 5, 5, 2, 2, 5, 5, 6, 6, 6, 1, 5], 'eval_runtime': 1.5976, 'eval_samples_per_second': 254.135, 'eval_steps_per_second': 2.504, 'epoch': 5.0}, {'eval_loss': 0.6763760447502136, 'eval_f1_macro': 0.3580467151895723, 'eval_f1_micro': 0.4482758620689655, 'eval_accuracy_balanced': 0.4174297924297924, 'eval_accuracy_not_b': 0.4482758620689655, 'eval_precision_macro': 0.3366300366300366, 'eval_recall_macro': 0.4174297924297924, 'eval_precision_micro': 0.4482758620689655, 'eval_recall_micro': 0.4482758620689655, 'eval_label_gold_raw': [6, 4, 5, 2, 5, 2, 5, 0, 0, 1, 5, 1, 1, 3, 2, 4, 5, 1, 2, 6, 0, 1, 0, 5, 1, 2, 2, 5, 4, 4, 5, 6, 0, 0, 2, 5, 0, 2, 5, 4, 1, 2, 1, 2, 2, 0, 5, 1, 4, 2, 4, 6, 6, 2, 4, 6, 5, 5], 'eval_label_predicted_raw': [6, 2, 5, 5, 6, 2, 6, 0, 2, 2, 5, 1, 1, 5, 0, 2, 5, 1, 5, 6, 6, 1, 0, 6, 5, 2, 5, 1, 2, 1, 2, 6, 0, 2, 6, 5, 1, 1, 2, 6, 1, 2, 1, 5, 5, 1, 5, 1, 1, 2, 0, 6, 6, 2, 5, 6, 2, 2], 'eval_runtime': 1.537, 'eval_samples_per_second': 264.144, 'eval_steps_per_second': 2.602, 'epoch': 5.0}, {'eval_loss': 0.8832194805145264, 'eval_f1_macro': 0.42226757369614515, 'eval_f1_micro': 0.5344827586206896, 'eval_accuracy_balanced': 0.44545454545454544, 'eval_accuracy_not_b': 0.5344827586206896, 'eval_precision_macro': 0.42766439909297044, 'eval_recall_macro': 0.44545454545454544, 'eval_precision_micro': 0.5344827586206896, 'eval_recall_micro': 0.5344827586206896, 'eval_label_gold_raw': [1, 1, 5, 1, 6, 2, 6, 5, 0, 0, 2, 1, 6, 4, 1, 4, 5, 5, 4, 1, 4, 5, 4, 4, 6, 6, 0, 0, 2, 5, 2, 1, 0, 0, 6, 6, 1, 6, 5, 5, 0, 2, 3, 5, 2, 5, 4, 5, 1, 6, 2, 0, 1, 0, 6, 0, 2, 0], 'eval_label_predicted_raw': [1, 1, 1, 0, 6, 3, 6, 6, 0, 6, 5, 1, 6, 1, 1, 1, 5, 6, 0, 1, 0, 3, 5, 5, 6, 6, 0, 3, 2, 5, 2, 1, 2, 1, 6, 6, 6, 6, 4, 1, 1, 2, 1, 5, 2, 5, 6, 5, 1, 6, 5, 4, 1, 2, 6, 0, 3, 0], 'eval_runtime': 1.5363, 'eval_samples_per_second': 264.275, 'eval_steps_per_second': 2.604, 'epoch': 5.0}]}, system_attrs={}, intermediate_values={0: 0.44714827460219475, 1: 0.4031612886292689, 2: 0.47837516615841735}, trial_id=9, state=TrialState.COMPLETE, value=None) 

#Study with max 20 max samples per class (100 no_topic), deberta-v3-mnli-fever-anli, hp-search over hypos and text formatting
Study with max fixed samples per class:
Best hyperparameters:  {'learning_rate': 4.883776913132329e-06, 'per_device_train_batch_size': 32, 'hypothesis_template': 'template_quote_three_text_separate_v2'}
Best hyperparameters all:  {'lr_scheduler_type': 'constant', 'learning_rate': 4.883776913132329e-06, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128, 'hypothesis_template': 'template_quote_three_text_separate_v2'}
Best performance:  0.501353076510944
Best performance details:  {'F1_macro_mean': 0.3564476496205275, 'F1_micro_mean': 0.6462585034013605, 'F1_macro_std': 0.010396278304630606, 'F1_micro_std': 0.009620500424306727}
Best trial full info:  FrozenTrial(number=11, values=[0.501353076510944], datetime_start=datetime.datetime(2022, 1, 4, 16, 50, 18, 853803), datetime_complete=datetime.datetime(2022, 1, 4, 16, 52, 2, 903530), params={'learning_rate': 4.883776913132329e-06, 'per_device_train_batch_size': 32, 'hypothesis_template': 'template_quote_three_text_separate_v2'}, distributions={'learning_rate': UniformDistribution(high=0.0005, low=1e-06), 'per_device_train_batch_size': CategoricalDistribution(choices=(8, 32, 64)), 'hypothesis_template': CategoricalDistribution(choices=('hypotheses_raw_v1', 'template_quote_three_text_separate_v1', 'template_quote_three_text_separate_v2', 'template_quote_two_text_separate_v1'))}, user_attrs={'hyperparameters_all': {'lr_scheduler_type': 'constant', 'learning_rate': 4.883776913132329e-06, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0, 'weight_decay': 0, 'per_device_eval_batch_size': 128, 'hypothesis_template': 'template_quote_three_text_separate_v2'}, 'metric_details': {'F1_macro_mean': 0.3564476496205275, 'F1_micro_mean': 0.6462585034013605, 'F1_macro_std': 0.010396278304630606, 'F1_micro_std': 0.009620500424306727}, 'results_trainer': [{'eval_loss': 0.4567376971244812, 'eval_f1_macro': 0.3434570191713049, 'eval_f1_micro': 0.6530612244897959, 'eval_accuracy_balanced': 0.3734176575146925, 'eval_accuracy_not_b': 0.6530612244897959, 'eval_precision_macro': 0.38734335839598993, 'eval_recall_macro': 0.3734176575146925, 'eval_precision_micro': 0.6530612244897959, 'eval_recall_micro': 0.6530612244897959, 'eval_label_gold_raw': [6, 6, 6, 1, 6, 6, 6, 5, 6, 1, 2, 2, 6, 6, 6, 6, 6, 5, 6, 0, 1, 4, 3, 3, 6, 6, 6, 6, 0, 6, 5, 6, 4, 0, 5, 2, 4, 6, 6, 0, 5, 6, 2, 6, 1, 6, 6, 6, 6, 0, 3, 6, 6, 2, 6, 6, 5, 6, 6, 6, 6, 6, 6, 2, 6, 5, 6, 1, 1, 2, 0, 6, 3, 6, 2, 4, 6, 6, 5, 5, 6, 6, 2, 6, 6, 0, 6, 5, 0, 6, 6, 5, 6, 0, 6, 4, 1, 6], 'eval_label_predicted_raw': [6, 2, 6, 1, 6, 6, 6, 5, 6, 1, 1, 5, 6, 6, 6, 6, 6, 6, 6, 0, 1, 1, 1, 6, 6, 6, 2, 6, 2, 6, 2, 6, 6, 1, 6, 2, 1, 6, 6, 5, 2, 6, 5, 6, 1, 6, 6, 6, 6, 2, 1, 6, 6, 0, 6, 6, 2, 6, 6, 6, 6, 6, 6, 2, 6, 2, 6, 2, 1, 2, 2, 6, 4, 6, 1, 1, 6, 6, 2, 5, 6, 6, 2, 6, 6, 2, 6, 1, 1, 6, 6, 6, 6, 6, 6, 4, 0, 6], 'eval_runtime': 3.7528, 'eval_samples_per_second': 182.796, 'eval_steps_per_second': 1.599, 'epoch': 5.0}, {'eval_loss': 0.5056496858596802, 'eval_f1_macro': 0.3569800339986675, 'eval_f1_micro': 0.6530612244897959, 'eval_accuracy_balanced': 0.38186119436119437, 'eval_accuracy_not_b': 0.6530612244897959, 'eval_precision_macro': 0.3480642256902761, 'eval_recall_macro': 0.38186119436119437, 'eval_precision_micro': 0.6530612244897959, 'eval_recall_micro': 0.6530612244897959, 'eval_label_gold_raw': [6, 6, 2, 6, 6, 5, 6, 5, 6, 2, 6, 6, 1, 6, 6, 6, 5, 5, 6, 6, 6, 2, 2, 4, 6, 1, 6, 6, 4, 1, 1, 0, 6, 6, 0, 0, 6, 6, 6, 2, 2, 6, 4, 5, 2, 4, 6, 6, 6, 0, 1, 6, 2, 6, 4, 6, 5, 6, 5, 0, 1, 1, 6, 6, 6, 6, 6, 6, 5, 6, 6, 4, 3, 5, 6, 4, 4, 6, 1, 6, 1, 5, 1, 6, 6, 3, 6, 2, 3, 5, 0, 6, 5, 1, 6, 5, 6, 6], 'eval_label_predicted_raw': [6, 6, 5, 6, 6, 6, 6, 2, 6, 5, 6, 6, 1, 6, 6, 6, 5, 5, 6, 6, 6, 0, 2, 1, 6, 0, 6, 6, 0, 1, 6, 5, 2, 6, 1, 1, 6, 6, 6, 1, 2, 6, 2, 2, 2, 1, 6, 6, 6, 1, 1, 6, 1, 6, 1, 6, 2, 6, 5, 0, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 5, 6, 1, 6, 6, 0, 6, 1, 5, 0, 6, 6, 0, 6, 2, 1, 1, 0, 6, 5, 1, 6, 2, 6, 6], 'eval_runtime': 3.6967, 'eval_samples_per_second': 185.572, 'eval_steps_per_second': 1.623, 'epoch': 5.0}, {'eval_loss': 0.43839916586875916, 'eval_f1_macro': 0.36890589569161, 'eval_f1_micro': 0.6326530612244898, 'eval_accuracy_balanced': 0.38606060606060605, 'eval_accuracy_not_b': 0.6326530612244898, 'eval_precision_macro': 0.3929292929292929, 'eval_recall_macro': 0.38606060606060605, 'eval_precision_micro': 0.6326530612244898, 'eval_recall_micro': 0.6326530612244898, 'eval_label_gold_raw': [6, 5, 2, 6, 6, 6, 5, 2, 0, 5, 6, 3, 6, 6, 4, 6, 5, 2, 6, 6, 5, 6, 0, 5, 6, 5, 0, 0, 6, 6, 6, 0, 6, 6, 6, 5, 6, 6, 6, 6, 1, 4, 6, 2, 2, 6, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 6, 5, 4, 6, 4, 6, 6, 6, 6, 1, 0, 6, 5, 5, 6, 5, 4, 6, 2, 4, 6, 1, 0, 6, 6, 6, 0, 6, 6, 6, 0, 6, 3, 6, 6, 1, 0, 1, 6, 6, 6, 0], 'eval_label_predicted_raw': [6, 6, 0, 6, 6, 6, 0, 5, 0, 6, 6, 3, 6, 6, 1, 6, 2, 5, 6, 6, 2, 6, 0, 2, 6, 0, 0, 0, 0, 6, 6, 0, 6, 6, 6, 3, 6, 6, 6, 6, 0, 0, 6, 0, 2, 6, 0, 6, 6, 1, 0, 0, 0, 6, 0, 0, 6, 5, 6, 6, 0, 6, 6, 6, 6, 1, 5, 6, 6, 5, 6, 6, 2, 6, 1, 2, 6, 2, 5, 6, 6, 6, 6, 6, 6, 6, 0, 6, 2, 6, 6, 1, 2, 6, 6, 6, 6, 0], 'eval_runtime': 3.622, 'eval_samples_per_second': 189.399, 'eval_steps_per_second': 1.657, 'epoch': 5.0}]}, system_attrs={}, intermediate_values={0: 0.4982591218305504, 1: 0.5050206292442316, 2: 0.5007794784580499}, trial_id=11, state=TrialState.COMPLETE, value=None) 

# multilingual Study with max 20 max samples per class (50 no_topic), mdeberta-v3-mnli-xnli, hp-search over hypos and text formatting, only eng training
Study with max fixed samples per class:
Best hyperparameters:  {'learning_rate': 2.7451377642062443e-05, 'per_device_train_batch_size': 8, 'hypothesis_template': 'template_quote_three_text_separate_v1'}
Best hyperparameters all:  {'lr_scheduler_type': 'constant', 'learning_rate': 2.7451377642062443e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 8, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 160, 'hypothesis_template': 'template_quote_three_text_separate_v1'}
Best performance:  0.39477127761610625
Best performance details:  {'F1_macro_mean': 0.31009050043769193, 'F1_micro_mean': 0.4794520547945206, 'F1_macro_std': 0.07219534237446665, 'F1_micro_std': 0.02739726027397263}
Best trial full info:  FrozenTrial(number=2, values=[0.39477127761610625], datetime_start=datetime.datetime(2022, 1, 5, 16, 16, 18, 399793), datetime_complete=datetime.datetime(2022, 1, 5, 16, 17, 42, 519331), params={'learning_rate': 2.7451377642062443e-05, 'per_device_train_batch_size': 8, 'hypothesis_template': 'template_quote_three_text_separate_v1'}, distributions={'learning_rate': UniformDistribution(high=9e-05, low=1e-07), 'per_device_train_batch_size': CategoricalDistribution(choices=(8, 16, 32)), 'hypothesis_template': CategoricalDistribution(choices=('template_quote_three_text_separate_v1', 'template_quote_three_text_separate_v2', 'template_quote_two_text_separate_v1', 'template_quote_one_text'))}, user_attrs={'hyperparameters_all': {'lr_scheduler_type': 'constant', 'learning_rate': 2.7451377642062443e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 8, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 160, 'hypothesis_template': 'template_quote_three_text_separate_v1'}, 'metric_details': {'F1_macro_mean': 0.31009050043769193, 'F1_micro_mean': 0.4794520547945206, 'F1_macro_std': 0.07219534237446665, 'F1_micro_std': 0.02739726027397263}, 'results_trainer': [{'eval_loss': 0.5025461912155151, 'eval_f1_macro': 0.2378951580632253, 'eval_f1_micro': 0.4520547945205479, 'eval_accuracy_balanced': 0.2952380952380952, 'eval_accuracy_not_b': 0.4520547945205479, 'eval_precision_macro': 0.21544011544011546, 'eval_recall_macro': 0.2952380952380952, 'eval_precision_micro': 0.4520547945205479, 'eval_recall_micro': 0.4520547945205479, 'eval_label_gold_raw': [2, 6, 6, 1, 0, 6, 6, 1, 6, 5, 0, 1, 5, 0, 0, 6, 1, 0, 0, 0, 4, 6, 6, 0, 2, 5, 6, 2, 5, 2, 6, 6, 6, 1, 5, 2, 1, 1, 5, 6, 6, 1, 5, 6, 0, 3, 1, 5, 4, 6, 3, 6, 6, 3, 4, 6, 6, 3, 1, 0, 6, 0, 1, 6, 5, 6, 0, 5, 6, 6, 1, 6, 2], 'eval_label_predicted_raw': [3, 6, 6, 6, 4, 6, 6, 5, 6, 5, 6, 0, 6, 5, 4, 6, 6, 3, 5, 4, 6, 6, 6, 5, 3, 5, 6, 3, 0, 4, 6, 6, 6, 6, 6, 6, 0, 5, 6, 6, 6, 4, 5, 6, 4, 5, 4, 5, 6, 6, 3, 6, 6, 2, 4, 6, 6, 0, 2, 6, 6, 0, 4, 6, 3, 6, 5, 6, 6, 6, 2, 6, 6], 'eval_runtime': 3.6622, 'eval_samples_per_second': 139.535, 'eval_steps_per_second': 1.092, 'epoch': 5.0}, {'eval_loss': 0.8109843730926514, 'eval_f1_macro': 0.3822858428121586, 'eval_f1_micro': 0.5068493150684932, 'eval_accuracy_balanced': 0.39724310776942356, 'eval_accuracy_not_b': 0.5068493150684932, 'eval_precision_macro': 0.431725292251608, 'eval_recall_macro': 0.39724310776942356, 'eval_precision_micro': 0.5068493150684932, 'eval_recall_micro': 0.5068493150684932, 'eval_label_gold_raw': [6, 0, 6, 5, 4, 6, 1, 0, 6, 1, 4, 2, 4, 0, 0, 5, 4, 1, 6, 1, 1, 3, 4, 6, 2, 6, 5, 5, 0, 1, 1, 6, 6, 6, 6, 6, 6, 0, 3, 0, 1, 0, 6, 5, 2, 2, 3, 5, 2, 4, 6, 1, 5, 1, 6, 2, 0, 5, 2, 6, 5, 1, 4, 6, 0, 1, 1, 6, 5, 3, 1, 2, 1], 'eval_label_predicted_raw': [6, 2, 6, 3, 0, 6, 1, 0, 6, 3, 2, 0, 2, 0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 6, 1, 6, 0, 0, 0, 1, 1, 0, 6, 6, 6, 6, 6, 2, 0, 0, 1, 0, 6, 5, 1, 1, 3, 2, 5, 0, 6, 6, 0, 1, 6, 2, 2, 5, 0, 6, 0, 1, 2, 6, 2, 0, 0, 6, 3, 0, 0, 2, 1], 'eval_runtime': 4.0328, 'eval_samples_per_second': 126.712, 'eval_steps_per_second': 0.992, 'epoch': 5.0}]}, system_attrs={}, intermediate_values={0: 0.3449749762918866, 1: 0.4445675789403259}, trial_id=2, state=TrialState.COMPLETE, value=None) 

# multilingual Study with max 20 max samples per class (50 no_topic), mdeberta-v3-mnli-xnli, hp-search over hypos and text formatting, multilingual training & eng hypos
Study with max fixed samples per class:
Best hyperparameters:  {'learning_rate': 2.579830232921441e-05, 'per_device_train_batch_size': 16, 'hypothesis_template': 'template_quote_three_text_separate_v1'}
Best hyperparameters all:  {'lr_scheduler_type': 'constant', 'learning_rate': 2.579830232921441e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 16, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128, 'hypothesis_template': 'template_quote_three_text_separate_v1'}
Best performance:  0.47325685023745495
Best performance details:  {'F1_macro_mean': 0.4018708433320527, 'F1_micro_mean': 0.5446428571428572, 'F1_macro_std': 0.008754866139568634, 'F1_micro_std': 0.005952380952380988}
Best trial full info:  FrozenTrial(number=9, values=[0.47325685023745495], datetime_start=datetime.datetime(2022, 1, 5, 19, 41, 24, 615170), datetime_complete=datetime.datetime(2022, 1, 5, 19, 50, 23, 73428), params={'learning_rate': 2.579830232921441e-05, 'per_device_train_batch_size': 16, 'hypothesis_template': 'template_quote_three_text_separate_v1'}, distributions={'learning_rate': UniformDistribution(high=9e-05, low=1e-07), 'per_device_train_batch_size': CategoricalDistribution(choices=(8, 16)), 'hypothesis_template': CategoricalDistribution(choices=('template_quote_three_text_separate_v1', 'template_quote_three_text_separate_v2', 'template_quote_two_text_separate_v1', 'template_quote_one_text'))}, user_attrs={'hyperparameters_all': {'lr_scheduler_type': 'constant', 'learning_rate': 2.579830232921441e-05, 'num_train_epochs': 5, 'seed': 42, 'per_device_train_batch_size': 16, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128, 'hypothesis_template': 'template_quote_three_text_separate_v1'}, 'metric_details': {'F1_macro_mean': 0.4018708433320527, 'F1_micro_mean': 0.5446428571428572, 'F1_macro_std': 0.008754866139568634, 'F1_micro_std': 0.005952380952380988}, 'results_trainer': [{'eval_loss': 1.1158760786056519, 'eval_f1_macro': 0.4106257094716213, 'eval_f1_micro': 0.5386904761904762, 'eval_accuracy_balanced': 0.4391013359787371, 'eval_accuracy_not_b': 0.5386904761904762, 'eval_precision_macro': 0.42727430089121754, 'eval_recall_macro': 0.4391013359787371, 'eval_precision_micro': 0.5386904761904762, 'eval_recall_micro': 0.5386904761904762, 'eval_label_gold_raw': [2, 5, 5, 6, 1, 1, 6, 4, 6, 6, 2, 3, 0, 2, 0, 5, 1, 6, 6, 3, 5, 6, 2, 6, 3, 5, 5, 6, 6, 2, 6, 6, 6, 1, 2, 6, 6, 1, 0, 3, 2, 2, 6, 5, 3, 2, 6, 0, 1, 1, 6, 6, 6, 0, 6, 3, 6, 2, 3, 5, 3, 6, 6, 6, 0, 5, 1, 5, 4, 6, 6, 1, 4, 0, 3, 6, 2, 6, 2, 2, 6, 0, 1, 2, 4, 2, 0, 0, 6, 1, 6, 5, 6, 4, 1, 2, 2, 0, 6, 6, 6, 5, 6, 3, 6, 6, 0, 0, 4, 6, 4, 2, 5, 2, 6, 5, 5, 4, 0, 5, 5, 0, 6, 6, 6, 5, 4, 4, 6, 4, 2, 6, 6, 6, 3, 1, 6, 6, 6, 4, 4, 5, 6, 6, 5, 6, 6, 2, 2, 3, 3, 5, 2, 5, 1, 0, 3, 4, 1, 6, 6, 4, 1, 5, 1, 0, 2, 5, 2, 5, 6, 6, 3, 1, 6, 6, 5, 6, 5, 4, 1, 6, 3, 5, 6, 2, 5, 6, 0, 6, 6, 3, 0, 5, 6, 2, 1, 6, 5, 1, 0, 6, 2, 4, 1, 3, 6, 1, 0, 3, 3, 1, 1, 6, 5, 6, 4, 0, 0, 6, 3, 6, 2, 4, 6, 6, 6, 6, 1, 0, 6, 4, 3, 2, 6, 2, 0, 1, 4, 3, 2, 6, 6, 4, 3, 5, 2, 1, 0, 5, 6, 6, 3, 6, 4, 4, 0, 2, 4, 0, 3, 3, 3, 1, 6, 4, 6, 2, 6, 6, 6, 5, 6, 4, 1, 0, 5, 5, 1, 2, 1, 5, 5, 2, 1, 5, 6, 2, 4, 4, 0, 0, 0, 3, 3, 6, 5, 6, 0, 6, 1, 2, 6, 1, 2, 4, 6, 5, 2, 0, 5, 1, 0, 6, 4, 4, 0, 6, 2, 6, 6, 5, 3, 2, 0, 6, 4, 6, 6, 3, 3, 6, 6, 2, 1, 4, 3, 6, 6, 6, 6, 5, 6, 3, 3, 1, 1, 2, 6, 5, 6, 3, 6, 2, 2, 6, 3, 6, 2, 3, 4, 0, 3, 6, 0, 6, 6, 6, 3, 6, 2, 4, 0, 6, 4, 6, 5, 3, 6, 1, 2, 5, 3, 6, 6, 6, 5, 2, 3, 4, 2, 3, 5, 1, 5, 0, 4, 6, 6, 5, 6, 3, 4, 6, 5, 4, 6, 3, 4, 4, 1, 3, 1, 6, 6, 2, 4, 5, 6, 3, 0, 0, 6, 6, 6, 1, 6, 1, 2, 6, 6, 5, 0, 4, 6, 2, 6, 0, 5, 4, 5, 0, 2, 4, 3, 6, 6, 5, 4, 3, 6, 6, 5, 5, 0, 1, 2, 5, 1, 2, 1, 6, 4, 0, 0, 2, 2, 0, 6, 0, 2, 4, 2, 4, 6, 5, 6, 1, 0, 6, 6, 1, 6, 0, 1, 5, 2, 1, 4, 2, 6, 6, 2, 2, 3, 2, 2, 6, 1, 0, 6, 4, 6, 0, 6, 5, 0, 1, 2, 6, 5, 3, 6, 2, 6, 6, 3, 6, 5, 5, 6, 6, 1, 6, 5, 6, 0, 2, 6, 0, 2, 1, 6, 1, 3, 6, 4, 6, 5, 0, 0, 1, 3, 5, 2, 5, 4, 0, 1, 5, 6, 0, 3, 4, 2, 1, 4, 1, 6, 0, 4, 2, 6, 6, 5, 6, 2, 1, 5, 4, 6, 4, 6, 4, 5, 6, 6, 0, 6, 6, 4, 2, 4, 6, 0, 1, 6, 2, 6, 3, 6, 2, 0, 5, 2, 1, 6, 3, 6, 1, 2, 6, 6, 0, 2, 6, 6, 4, 0, 2, 3, 5, 4, 4, 6, 3, 6, 6, 1, 6, 6, 0, 2, 6, 2, 6, 0, 4, 5, 6, 0, 6, 6, 6, 2, 4, 1, 1, 1, 2, 3, 5, 6, 6, 5, 1, 6, 6, 6, 6, 3, 1, 6, 6, 4, 5, 6, 6, 6, 5, 0, 5, 6, 5, 1, 3, 6, 5, 6, 0, 6, 6], 'eval_label_predicted_raw': [5, 5, 1, 6, 1, 4, 6, 4, 5, 6, 1, 2, 4, 1, 1, 5, 1, 6, 6, 5, 6, 5, 1, 6, 3, 1, 5, 6, 6, 1, 6, 6, 4, 1, 1, 6, 5, 1, 4, 5, 1, 4, 5, 6, 6, 5, 6, 2, 5, 4, 6, 6, 6, 1, 6, 0, 6, 1, 1, 5, 4, 5, 6, 6, 3, 2, 1, 5, 4, 6, 6, 1, 0, 2, 4, 6, 5, 6, 2, 2, 4, 1, 1, 1, 4, 2, 5, 1, 6, 2, 6, 5, 6, 4, 1, 1, 2, 2, 6, 6, 6, 5, 2, 2, 6, 6, 2, 6, 5, 5, 5, 2, 5, 2, 6, 4, 5, 5, 5, 4, 2, 1, 6, 2, 6, 5, 3, 3, 6, 5, 2, 6, 6, 6, 3, 1, 6, 4, 6, 4, 4, 5, 6, 6, 4, 6, 6, 2, 2, 6, 5, 5, 2, 1, 1, 1, 4, 4, 1, 6, 6, 4, 5, 5, 1, 2, 1, 4, 2, 3, 6, 1, 4, 1, 6, 6, 4, 2, 5, 3, 1, 6, 1, 5, 6, 2, 5, 6, 2, 6, 6, 4, 1, 4, 4, 5, 4, 6, 4, 1, 0, 6, 0, 5, 2, 4, 6, 1, 3, 4, 1, 6, 1, 4, 5, 6, 4, 2, 1, 6, 2, 2, 0, 1, 6, 6, 6, 6, 4, 5, 6, 1, 4, 2, 6, 1, 4, 1, 4, 3, 2, 6, 6, 4, 6, 5, 4, 0, 6, 6, 6, 6, 0, 6, 4, 4, 4, 1, 4, 1, 4, 6, 1, 6, 6, 4, 6, 2, 6, 6, 6, 6, 6, 1, 5, 1, 4, 5, 1, 5, 1, 3, 2, 1, 1, 4, 6, 5, 4, 4, 1, 2, 6, 4, 6, 6, 5, 6, 0, 1, 1, 2, 6, 6, 5, 5, 3, 5, 1, 2, 5, 6, 3, 6, 6, 4, 3, 6, 2, 6, 6, 5, 4, 2, 1, 6, 5, 6, 6, 5, 1, 6, 6, 2, 1, 4, 2, 6, 6, 6, 6, 5, 6, 4, 0, 1, 1, 1, 6, 4, 6, 5, 5, 4, 1, 6, 3, 6, 4, 6, 1, 6, 4, 6, 2, 6, 6, 6, 5, 6, 5, 5, 1, 6, 1, 4, 6, 1, 6, 2, 5, 3, 5, 5, 6, 5, 4, 2, 5, 4, 1, 5, 5, 1, 5, 0, 1, 6, 6, 5, 6, 5, 1, 6, 0, 1, 6, 3, 4, 6, 4, 4, 1, 6, 6, 5, 4, 5, 6, 3, 1, 4, 6, 6, 6, 4, 6, 1, 2, 6, 6, 1, 1, 1, 6, 5, 6, 3, 5, 5, 4, 2, 1, 4, 3, 6, 4, 5, 1, 0, 6, 6, 5, 5, 1, 1, 1, 5, 1, 2, 1, 6, 4, 1, 2, 2, 2, 1, 6, 1, 2, 1, 2, 4, 6, 6, 6, 2, 6, 6, 6, 0, 6, 0, 6, 5, 2, 1, 5, 2, 6, 6, 1, 6, 1, 1, 4, 6, 1, 4, 6, 4, 6, 1, 2, 5, 2, 1, 1, 6, 5, 2, 6, 2, 6, 6, 1, 6, 2, 5, 6, 6, 1, 6, 0, 6, 2, 4, 6, 6, 2, 3, 6, 1, 0, 6, 4, 6, 5, 2, 1, 1, 4, 4, 5, 1, 4, 1, 2, 0, 6, 6, 2, 0, 5, 1, 4, 1, 6, 2, 5, 1, 6, 5, 5, 6, 2, 1, 5, 5, 6, 2, 6, 1, 4, 6, 6, 1, 6, 6, 4, 1, 4, 6, 1, 0, 6, 4, 4, 5, 1, 2, 2, 5, 2, 1, 6, 5, 6, 4, 2, 6, 6, 1, 2, 6, 6, 1, 1, 1, 1, 5, 1, 4, 6, 4, 6, 6, 2, 6, 6, 1, 2, 6, 6, 6, 1, 4, 5, 6, 2, 6, 6, 6, 2, 1, 2, 1, 1, 0, 6, 5, 6, 6, 5, 4, 6, 4, 6, 6, 4, 2, 6, 6, 4, 4, 5, 6, 6, 4, 2, 2, 6, 4, 2, 5, 2, 6, 6, 1, 6, 6], 'eval_runtime': 44.8637, 'eval_samples_per_second': 104.851, 'eval_steps_per_second': 0.825, 'epoch': 5.0}, {'eval_loss': 0.7983213663101196, 'eval_f1_macro': 0.39311597719248403, 'eval_f1_micro': 0.5505952380952381, 'eval_accuracy_balanced': 0.4341034777249689, 'eval_accuracy_not_b': 0.5505952380952381, 'eval_precision_macro': 0.37655051959372415, 'eval_recall_macro': 0.4341034777249689, 'eval_precision_micro': 0.5505952380952381, 'eval_recall_micro': 0.5505952380952381, 'eval_label_gold_raw': [3, 6, 5, 6, 3, 1, 4, 6, 6, 4, 5, 2, 6, 4, 0, 6, 6, 4, 5, 5, 2, 6, 6, 2, 6, 6, 0, 1, 5, 4, 6, 3, 5, 3, 6, 4, 2, 4, 2, 5, 6, 1, 5, 2, 6, 6, 3, 6, 5, 1, 3, 5, 2, 6, 6, 2, 6, 6, 4, 2, 0, 1, 5, 1, 6, 5, 6, 6, 5, 3, 1, 0, 6, 6, 6, 6, 6, 2, 6, 1, 1, 6, 2, 6, 5, 4, 6, 6, 2, 6, 0, 2, 1, 5, 6, 3, 2, 6, 0, 5, 6, 4, 2, 5, 2, 0, 6, 0, 1, 3, 6, 1, 6, 2, 5, 3, 3, 6, 0, 1, 3, 1, 0, 5, 4, 1, 0, 6, 1, 6, 2, 0, 6, 6, 3, 1, 6, 6, 1, 1, 6, 3, 5, 6, 0, 0, 3, 6, 6, 5, 2, 4, 6, 6, 2, 4, 6, 6, 6, 5, 4, 2, 2, 4, 5, 5, 6, 2, 2, 0, 6, 5, 4, 2, 3, 6, 4, 1, 4, 0, 1, 6, 4, 2, 1, 2, 5, 6, 1, 6, 0, 4, 6, 0, 2, 3, 4, 2, 6, 0, 0, 6, 2, 1, 1, 5, 5, 1, 0, 0, 6, 1, 6, 6, 4, 3, 1, 4, 6, 3, 6, 1, 1, 6, 2, 5, 0, 0, 4, 1, 4, 1, 6, 5, 3, 1, 2, 6, 2, 3, 5, 0, 3, 6, 2, 2, 2, 4, 6, 5, 1, 5, 0, 0, 6, 4, 0, 6, 5, 0, 0, 5, 3, 2, 6, 4, 1, 6, 2, 4, 5, 6, 2, 4, 2, 6, 5, 6, 2, 2, 6, 6, 6, 6, 6, 2, 4, 0, 1, 6, 0, 6, 4, 6, 3, 6, 0, 1, 1, 1, 6, 1, 6, 6, 6, 1, 5, 5, 4, 0, 5, 1, 1, 5, 4, 3, 2, 4, 6, 6, 5, 0, 2, 0, 5, 1, 4, 3, 5, 5, 2, 6, 0, 6, 5, 3, 6, 2, 1, 2, 6, 5, 0, 4, 5, 4, 2, 3, 1, 1, 3, 3, 5, 0, 6, 1, 6, 6, 0, 3, 2, 3, 6, 5, 2, 6, 0, 6, 6, 4, 3, 6, 3, 6, 0, 6, 0, 4, 6, 4, 4, 0, 6, 6, 4, 6, 1, 6, 6, 6, 3, 6, 2, 1, 0, 4, 0, 6, 4, 1, 6, 1, 6, 1, 2, 5, 4, 2, 6, 1, 2, 2, 6, 2, 0, 1, 6, 6, 6, 0, 4, 6, 5, 0, 6, 5, 4, 5, 2, 6, 6, 6, 1, 1, 6, 0, 6, 6, 4, 6, 4, 6, 6, 2, 5, 2, 6, 1, 1, 5, 6, 3, 3, 6, 6, 6, 2, 6, 2, 5, 1, 6, 0, 5, 6, 6, 5, 3, 4, 5, 6, 2, 3, 3, 5, 5, 2, 2, 6, 4, 6, 0, 6, 3, 3, 6, 6, 2, 6, 0, 0, 6, 0, 1, 2, 1, 6, 0, 6, 2, 3, 1, 1, 6, 6, 1, 4, 6, 4, 6, 1, 5, 6, 3, 4, 6, 2, 6, 1, 6, 6, 2, 6, 5, 6, 2, 2, 6, 5, 6, 6, 2, 6, 6, 4, 6, 3, 6, 3, 6, 3, 6, 5, 0, 6, 0, 6, 2, 1, 6, 3, 1, 6, 6, 4, 6, 2, 6, 6, 5, 5, 4, 0, 6, 6, 0, 4, 6, 5, 6, 2, 6, 6, 1, 6, 3, 6, 0, 2, 1, 3, 2, 4, 2, 6, 6, 6, 2, 2, 2, 0, 2, 6, 6, 6, 0, 6, 6, 0, 3, 5, 1, 6, 5, 2, 3, 2, 4, 6, 5, 6, 4, 0, 6, 6, 0, 1, 2, 4, 6, 6, 3, 6, 0, 5, 6, 1, 2, 6, 5, 4, 2, 2, 1, 6, 6, 3, 6, 6, 5, 1, 0, 0, 6, 6, 2, 6, 4, 6, 5, 6, 6, 1, 1, 6, 2, 4, 1, 6, 0, 0, 3, 4, 5, 0, 2, 5, 2, 4, 3, 2, 1], 'eval_label_predicted_raw': [2, 6, 6, 6, 5, 4, 4, 6, 6, 6, 5, 2, 5, 1, 2, 2, 6, 4, 5, 6, 2, 6, 5, 2, 6, 6, 1, 2, 3, 4, 6, 6, 0, 4, 6, 4, 1, 5, 2, 5, 6, 1, 5, 2, 6, 6, 5, 6, 5, 2, 6, 5, 5, 4, 6, 5, 6, 6, 6, 2, 5, 1, 5, 4, 6, 4, 6, 6, 4, 4, 1, 5, 6, 6, 6, 6, 6, 2, 6, 1, 1, 5, 0, 6, 5, 5, 6, 6, 5, 6, 1, 4, 1, 4, 6, 5, 5, 6, 1, 5, 4, 5, 5, 3, 1, 5, 6, 2, 4, 6, 6, 1, 5, 2, 4, 5, 6, 6, 2, 2, 3, 1, 4, 5, 5, 6, 1, 6, 2, 6, 2, 2, 6, 6, 6, 1, 6, 6, 1, 1, 6, 4, 2, 6, 5, 6, 6, 6, 6, 5, 4, 1, 6, 5, 2, 4, 6, 6, 6, 5, 1, 2, 1, 1, 5, 5, 6, 2, 2, 5, 5, 6, 5, 2, 5, 6, 4, 4, 4, 2, 3, 6, 4, 1, 2, 2, 5, 6, 1, 6, 2, 4, 6, 2, 2, 5, 4, 2, 6, 1, 5, 6, 2, 3, 1, 1, 2, 1, 4, 2, 5, 6, 6, 6, 5, 0, 1, 4, 6, 5, 6, 6, 1, 6, 1, 2, 1, 1, 5, 2, 5, 1, 6, 2, 4, 4, 2, 6, 4, 6, 0, 1, 5, 6, 6, 5, 2, 6, 6, 1, 2, 5, 1, 2, 6, 4, 6, 6, 6, 2, 3, 4, 4, 2, 6, 4, 1, 6, 2, 6, 5, 6, 5, 4, 5, 6, 6, 6, 2, 2, 6, 6, 6, 6, 6, 5, 4, 6, 2, 6, 2, 6, 4, 6, 1, 6, 6, 1, 1, 2, 5, 1, 6, 5, 6, 6, 5, 5, 4, 2, 5, 1, 2, 5, 1, 4, 1, 4, 2, 6, 5, 4, 5, 1, 2, 1, 4, 2, 6, 2, 2, 6, 2, 6, 2, 4, 6, 5, 2, 2, 6, 5, 1, 1, 4, 4, 5, 4, 4, 4, 5, 6, 2, 2, 6, 2, 6, 6, 1, 6, 4, 5, 6, 5, 5, 6, 1, 6, 6, 4, 2, 6, 5, 6, 1, 6, 2, 4, 6, 4, 4, 2, 6, 6, 4, 6, 1, 6, 6, 6, 5, 6, 2, 4, 2, 2, 2, 6, 4, 2, 6, 2, 6, 5, 5, 6, 2, 1, 6, 2, 2, 2, 6, 5, 2, 1, 6, 6, 6, 1, 4, 6, 5, 1, 6, 5, 6, 6, 2, 6, 5, 6, 4, 1, 6, 1, 6, 6, 4, 6, 4, 6, 6, 2, 5, 2, 6, 1, 4, 5, 6, 1, 5, 6, 6, 6, 4, 6, 2, 4, 1, 6, 1, 5, 6, 6, 5, 2, 4, 4, 6, 2, 4, 5, 5, 3, 1, 2, 6, 4, 6, 3, 5, 5, 6, 6, 6, 5, 6, 3, 1, 6, 4, 2, 2, 2, 6, 1, 6, 2, 1, 1, 1, 6, 6, 4, 5, 6, 4, 6, 1, 6, 6, 4, 4, 6, 2, 6, 6, 4, 6, 6, 6, 5, 0, 2, 6, 6, 5, 6, 6, 2, 6, 6, 1, 6, 4, 4, 5, 6, 4, 6, 5, 1, 6, 6, 6, 4, 1, 6, 5, 4, 6, 4, 4, 6, 2, 6, 6, 4, 5, 4, 1, 5, 5, 2, 1, 6, 5, 6, 5, 6, 6, 1, 6, 5, 6, 1, 2, 1, 3, 1, 1, 2, 6, 6, 6, 6, 2, 2, 1, 1, 6, 6, 6, 6, 6, 6, 2, 5, 6, 1, 6, 5, 5, 4, 2, 1, 6, 5, 5, 4, 2, 6, 6, 2, 5, 1, 4, 3, 6, 6, 6, 1, 4, 6, 2, 6, 6, 4, 1, 2, 6, 4, 6, 6, 2, 5, 6, 5, 4, 2, 1, 6, 6, 2, 6, 4, 6, 4, 6, 6, 6, 1, 6, 1, 5, 1, 6, 4, 2, 4, 4, 5, 4, 5, 5, 2, 4, 6, 6, 1], 'eval_runtime': 44.5327, 'eval_samples_per_second': 105.63, 'eval_steps_per_second': 0.831, 'epoch': 5.0}]}, system_attrs={}, intermediate_values={0: 0.47465809283104876, 1: 0.4718556076438611}, trial_id=9, state=TrialState.COMPLETE, value=None) 




# In[ ]:


## save study  # https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-save-and-resume-studies
#import joblib
#os.getcwd()
#joblib.dump(study, f"./results/{TRAINING_DIRECTORY}/optuna_study_lst_minilm_1.pkl")
## resume study
#study = joblib.load(f"./results/{TRAINING_DIRECTORY}/optuna_study_lst_1.pkl")


# In[ ]:


## optuna visualisation  https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
from optuna.visualization import plot_optimization_history, plot_contour, plot_slice, plot_param_importances

#plot_optimization_history(study)
for study in study_lst: 
  plot_optimization_history(study).show()


# In[ ]:


#plot_slice(study)
for study in study_lst: 
  plot_slice(study).show()


# In[ ]:


#plot_contour(study)


# In[ ]:


for study in study_lst:
  try:
    print(optuna.importance.get_param_importances(study))  # based on this https://github.com/automl/fanova
  except Exception as e:
    print(e)
  #plot_param_importances(study).show()

#optuna.visualization.plot_param_importances(
#    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
#)


# In[ ]:


### Confusion matrix
# get labels out of study object and concatenate labels from different runs

for study in study_lst: 
  label_gold_trial = []
  label_predicted_trial = []
  for dic in study.best_trial.user_attrs["results_trainer"]:
    label_gold_trial = label_gold_trial + dic["eval_label_gold_raw"]
    label_predicted_trial = label_predicted_trial + dic["eval_label_predicted_raw"]
  assert len(label_gold_trial) == len(label_predicted_trial)

  # map numerical labels to label texts
  label_num_text_map = {label_num: label_text for label_num, label_text in enumerate(np.sort(df_cl.label_text.unique()))}
  label_gold_trial_text = [label_num_text_map[label] for label in label_gold_trial]
  label_predicted_trial_text = [label_num_text_map[label] for label in label_predicted_trial]

  # sklearn plot https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import matplotlib.pyplot as plt
  conf_mat = confusion_matrix(label_gold_trial_text, label_predicted_trial_text, normalize=None)  #labels=np.sort(df_cl.label_text.unique())  
  disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=np.sort(df_cl.label_text.unique()))
  plt.rcParams["figure.figsize"] = (9,9)
  disp.plot(xticks_rotation="vertical")
  plt.show()


# #### Tests over K samples

# In[ ]:


#from tqdm.notebook import tqdm
#import warnings
#transformers.logging.set_verbosity_warning()  # https://huggingface.co/transformers/main_classes/logging.html
#warnings.filterwarnings(action='ignore')

## defined during hyperparameter tuning
#METHOD = "standard_dl"  # "standard_dl", "nli", "nsp"
#N_MAX_SAMPLE_PER_CLASS_DEV = [16, 32, 64, 128]
#TRAINING_DIRECTORY = "nli-few-shot/cap/sotu"
#LABEL_TEXT_ALPHABETICAL = np.sort(df_cl.label_text.unique())
#HYPOTHESIS_TEMPLATE = "template_quote_single_text"
#HYPOTHESIS_DIC = hypothesis_hyperparams_dic[HYPOTHESIS_TEMPLATE]  # hypo_label_dic_short , hypo_label_dic_long, hypo_label_dic_short_subcat
#DISABLE_TQDM = False

#MODEL_NAME = "./results/nli-few-shot/mnli-2c/MiniLM-L6-mnli-binary/" # "textattack/bert-base-uncased-MNLI"  # "./results/nli-few-shot/MiniLM-L6-mnli-binary/",   # './results/nli-few-shot/all-nli-2c/MiniLM-L6-allnli-2c-v1'
#MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"    # "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",    # "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
#MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
#MODEL_NAME = "bert-base-uncased"  # "google/fnet-base" ,  "bert-base-uncased" , "bert-large-uncased" , "google/mobilebert-uncased"

N_MAX_SAMPLE_PER_CLASS_TEST = [0] + N_MAX_SAMPLE_PER_CLASS_DEV  # [0] for zero-shot
N_RANDOM_REPETITIONS = 1
print(N_MAX_SAMPLE_PER_CLASS_TEST)

HYPER_PARAMS_LST = [study.best_trial.user_attrs["hyperparameters_all"] for study in study_lst]

HYPOTHESIS_TEMPLATE_LST = [hyperparams_dic["hypothesis_template"] for hyperparams_dic in HYPER_PARAMS_LST]
HYPOTHESIS_TEMPLATE_LST = [HYPOTHESIS_TEMPLATE_LST[0]] + HYPOTHESIS_TEMPLATE_LST  # zero-shot gets same hypo template as first study run
print(HYPOTHESIS_TEMPLATE_LST)

HYPER_PARAMS_LST = [{key:dic[key] for key in dic if key!="hypothesis_template"} for dic in HYPER_PARAMS_LST]  # return dic with all elements, except hypothesis template
HYPER_PARAMS_LST = [{key:(value*1 if key=="num_train_epochs" else value) for key, value in dic.items()} for dic in HYPER_PARAMS_LST]  # multiply the epochs compared to hp search. can also select earlier model
HYPER_PARAMS_LST_TEST = [HYPER_PARAMS_LST[0]] + HYPER_PARAMS_LST  # add random hyperparams for 0-shot run (will not be used anyways)
print(HYPER_PARAMS_LST_TEST)


# In[ ]:



### run random cross-validation for hyperparameter search without a dev set
np.random.seed(SEED_GLOBAL)

# K example intervals loop
#metrics_mean_lst = []
#metrics_std_lst = []
experiment_details_dic = {}
for n_max_sample, hyperparams, hypothesis_template in tqdm.notebook.tqdm(zip(N_MAX_SAMPLE_PER_CLASS_TEST, HYPER_PARAMS_LST_TEST, HYPOTHESIS_TEMPLATE_LST), desc="Iterations for different number of texts", leave=True):
  random_run_parameters_dic_dic = {}
  # randomness stability loop. Objective: calculate F1 across N samples to test for influence of different (random) samples
  for random_seed_sample in tqdm.notebook.tqdm(np.random.choice(range(1000), size=N_RANDOM_REPETITIONS), desc="Iterations for std", leave=True):
    # unrealistic oracle sample
    #df_train_samp = df_train.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), n_max_sample), random_state=random_seed_sample))
    # fully random sample
    #df_train_samp = random_sample_fill(df_train=df_train, n_sample_per_class=n_max_sample, seed=random_seed_sample)
    if n_max_sample == "fixed":
      df_train_samp = df_train
    else:
      df_train_samp = random_sample_fill(df_train=df_train, n_sample_per_class=n_max_sample, seed=random_seed_sample)


    # chose the text format depending on hyperparams
    df_train_samp = format_text(df=df_train_samp, text_format=hypothesis_template)
    df_test = format_text(df=df_test, text_format=hypothesis_template)

    # format train and dev set for NLI etc.
    df_train_samp_formatted = format_nli_trainset(df_train_samp=df_train_samp, hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template], method=METHOD)  # hypo_label_dic_short , hypo_label_dic_long
    df_test_formatted = format_nli_testset(df_test=df_test, hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template], method=METHOD)  # hypo_label_dic_short , hypo_label_dic_long

    clean_memory()
    model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    encoded_dataset = tokenize_datasets(df_train_samp=df_train_samp_formatted, df_test=df_test_formatted, tokenizer=tokenizer, method=METHOD, max_length=None)

    train_args = set_train_args(hyperparams_dic=hyperparams, training_directory=TRAINING_DIRECTORY, disable_tqdm=DISABLE_TQDM, evaluation_strategy="no", fp16=fp16_bool)  # seed=random_seed_sample ! can the order to data loading via seed (but then different from HP search)
    trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=encoded_dataset, train_args=train_args, 
                            method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    clean_memory()

    if n_max_sample != 0:
      trainer.train()
    results = trainer.evaluate()  # eval_dataset=encoded_dataset_test

    random_run_parameters_dic = {"method": METHOD, "n_max_sample_per_class": n_max_sample, "n_train_total": len(df_train_samp), "model": MODEL_NAME, "metrics": results, "hyperparams": hyperparams}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic
    transformers.logging.set_verbosity_warning()  # https://huggingface.co/transformers/main_classes/logging.html

    random_run_parameters_dic_dic.update({f"seed_{random_seed_sample}": random_run_parameters_dic})

    if n_max_sample == 0:  # only one inference necessary on same test set in case of zero-shot
      break

  experiment_details_dic.update({f"run_sample_{n_max_sample}": random_run_parameters_dic_dic})
  #metrics_sample_k = [dic["metrics"]["eval_f1_macro"] for dic in run_parameters_dic_lst]
  #metrics_mean_lst.append(np.mean(metrics_sample_k))
  #metrics_std_lst.append(np.std(metrics_sample_k))

#warnings.filterwarnings(action='default')


# In[ ]:


## minilm-mnli-binary, single quote
# 0-shot
Aggregate metrics:  {'f1_macro': 0.0768164895475046, 'f1_micro': 0.037334403853873946, 'accuracy_balanced': 0.20064767875020978, 'accuracy_not_b': 0.037334403853873946, 'precision_macro': 0.07879247702819803, 'recall_macro': 0.20064767875020978, 'precision_micro': 0.037334403853873946, 'recall_micro': 0.037334403853873946}
Detailed metrics:  {'immigration_neutral': {'precision': 0.022593320235756387, 'recall': 0.35384615384615387, 'f1-score': 0.04247460757156048, 'support': 65}, 'immigration_sceptical': {'precision': 0.16, 'recall': 0.037383177570093455, 'f1-score': 0.0606060606060606, 'support': 107}, 'immigration_supportive': {'precision': 0.12, 'recall': 0.16535433070866143, 'f1-score': 0.1390728476821192, 'support': 127}, 'integration_neutral': {'precision': 0.004873294346978557, 'recall': 0.5, 'f1-score': 0.009652509652509652, 'support': 10}, 'integration_sceptical': {'precision': 0.06896551724137931, 'recall': 0.11764705882352941, 'f1-score': 0.08695652173913043, 'support': 17}, 'integration_supportive': {'precision': 0.17511520737327188, 'recall': 0.23030303030303031, 'f1-score': 0.19895287958115185, 'support': 165}, 'no_topic': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 2000}, 'accuracy': 0.037334403853873946, 'macro avg': {'precision': 0.07879247702819803, 'recall': 0.20064767875020978, 'f1-score': 0.0768164895475046, 'support': 2491}, 'weighted avg': {'precision': 0.025669900348650045, 'recall': 0.037334403853873946, 'f1-score': 0.024612589614455813, 'support': 2491}} 
# gold-answers train
Aggregate metrics:  {'f1_macro': 0.16392035268878, 'f1_micro': 0.802087515054195, 'accuracy_balanced': 0.1656380562127966, 'accuracy_not_b': 0.8020875150541951, 'precision_macro': 0.3304603082798572, 'recall_macro': 0.1656380562127966, 'precision_micro': 0.8020875150541951, 'recall_micro': 0.8020875150541951}
Detailed metrics:  {'immigration_neutral': {'precision': 0.07692307692307693, 'recall': 0.03076923076923077, 'f1-score': 0.04395604395604396, 'support': 65}, 'immigration_sceptical': {'precision': 0.5, 'recall': 0.037383177570093455, 'f1-score': 0.06956521739130435, 'support': 107}, 'immigration_supportive': {'precision': 0.2857142857142857, 'recall': 0.015748031496062992, 'f1-score': 0.02985074626865672, 'support': 127}, 'integration_neutral': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 10}, 'integration_sceptical': {'precision': 0.05, 'recall': 0.058823529411764705, 'f1-score': 0.05405405405405405, 'support': 17}, 'integration_supportive': {'precision': 0.5714285714285714, 'recall': 0.024242424242424242, 'f1-score': 0.046511627906976744, 'support': 165}, 'no_topic': {'precision': 0.829156223893066, 'recall': 0.9925, 'f1-score': 0.9035047792444242, 'support': 2000}, 'accuracy': 0.8020875150541951, 'macro avg': {'precision': 0.3304603082798572, 'recall': 0.1656380562127966, 'f1-score': 0.16392035268878, 'support': 2491}, 'weighted avg': {'precision': 0.7419646231864956, 'recall': 0.8020875150541951, 'f1-score': 0.7345221043382377, 'support': 2491}} 
# it weirdly predicts too much as "no_topic", although no strong training set imbalance
# 20 max samples per class train
Aggregate metrics:  {'f1_macro': 0.25996928845550854, 'f1_micro': 0.6829959514170041, 'accuracy_balanced': 0.36037510476978357, 'accuracy_not_b': 0.6829959514170041, 'precision_macro': 0.2736226932916019, 'recall_macro': 0.36037510476978357, 'precision_micro': 0.6829959514170041, 'recall_micro': 0.6829959514170041, 
Detailed metrics:  {'immigration_neutral': {'precision': 0.16279069767441862, 'recall': 0.15217391304347827, 'f1-score': 0.15730337078651688, 'support': 46}, 'immigration_sceptical': {'precision': 0.26666666666666666, 'recall': 0.1188118811881188, 'f1-score': 0.16438356164383564, 'support': 101}, 'immigration_supportive': {'precision': 0.28448275862068967, 'recall': 0.25384615384615383, 'f1-score': 0.26829268292682923, 'support': 130}, 'integration_neutral': {'precision': 0.021739130434782608, 'recall': 0.16666666666666666, 'f1-score': 0.03846153846153846, 'support': 6}, 'integration_sceptical': {'precision': 0.0335195530726257, 'recall': 0.6, 'f1-score': 0.0634920634920635, 'support': 10}, 'integration_supportive': {'precision': 0.19708029197080293, 'recall': 0.4576271186440678, 'f1-score': 0.2755102040816327, 'support': 177}, 'no_topic': {'precision': 0.949079754601227, 'recall': 0.7735, 'f1-score': 0.8523415977961432, 'support': 2000}, 'accuracy': 0.6829959514170041, 'macro avg': {'precision': 0.2736226932916019, 'recall': 0.36037510476978357, 'f1-score': 0.25996928845550854, 'support': 2470}, 'weighted avg': {'precision': 0.8117055932152499, 'recall': 0.6829959514170041, 'f1-score': 0.7340206215154332, 'support': 2470}} 

## minilm-mnli-binary, three texts, one quote (leakage issue due to sequentiality?)
# 0-shot
Aggregate metrics:  {'f1_macro': 0.08643968497811304, 'f1_micro': 0.05344129554655871, 'accuracy_balanced': 0.1783093801699199, 'accuracy_not_b': 0.05344129554655871, 'precision_macro': 0.08581774400864713, 'recall_macro': 0.1783093801699199, 'precision_micro': 0.05344129554655871, 'recall_micro': 0.05344129554655871,
Detailed metrics:  {'immigration_neutral': {'precision': 0.02112676056338028, 'recall': 0.2608695652173913, 'f1-score': 0.03908794788273615, 'support': 46}, 'immigration_sceptical': {'precision': 0.325, 'recall': 0.12871287128712872, 'f1-score': 0.1843971631205674, 'support': 101}, 'immigration_supportive': {'precision': 0.12030075187969924, 'recall': 0.24615384615384617, 'f1-score': 0.16161616161616163, 'support': 130}, 'integration_neutral': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6}, 'integration_sceptical': {'precision': 0.03773584905660377, 'recall': 0.2, 'f1-score': 0.06349206349206349, 'support': 10}, 'integration_supportive': {'precision': 0.09656084656084656, 'recall': 0.4124293785310734, 'f1-score': 0.1564844587352626, 'support': 177}, 'no_topic': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 2000}, 'accuracy': 0.05344129554655871, 'macro avg': {'precision': 0.08581774400864713, 'recall': 0.1783093801699199, 'f1-score': 0.08643968497811304, 'support': 2470}, 'weighted avg': {'precision': 0.027086865207332906, 'recall': 0.05344129554655871, 'f1-score': 0.028244910898358823, 'support': 2470}} 
# 20 max samples per class
Aggregate metrics:  {'f1_macro': 0.2848331220068796, 'f1_micro': 0.7983805668016194, 'accuracy_balanced': 0.3144265417524609, 'accuracy_not_b': 0.7983805668016194, 'precision_macro': 0.2933859532640208, 'recall_macro': 0.3144265417524609, 'precision_micro': 0.7983805668016194, 'recall_micro': 0.7983805668016194, 
Detailed metrics:  {'immigration_neutral': {'precision': 0.08333333333333333, 'recall': 0.08695652173913043, 'f1-score': 0.0851063829787234, 'support': 46}, 'immigration_sceptical': {'precision': 0.3333333333333333, 'recall': 0.15841584158415842, 'f1-score': 0.21476510067114096, 'support': 101}, 'immigration_supportive': {'precision': 0.3106060606060606, 'recall': 0.3153846153846154, 'f1-score': 0.31297709923664124, 'support': 130}, 'integration_neutral': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6}, 'integration_sceptical': {'precision': 0.04285714285714286, 'recall': 0.3, 'f1-score': 0.075, 'support': 10}, 'integration_supportive': {'precision': 0.33185840707964603, 'recall': 0.423728813559322, 'f1-score': 0.37220843672456577, 'support': 177}, 'no_topic': {'precision': 0.9517133956386293, 'recall': 0.9165, 'f1-score': 0.9337748344370861, 'support': 2000}, 'accuracy': 0.7983805668016194, 'macro avg': {'precision': 0.2933859532640208, 'recall': 0.3144265417524609, 'f1-score': 0.2848331220068796, 'support': 2470}, 'weighted avg': {'precision': 0.8261024650355123, 'recall': 0.7983805668016194, 'f1-score': 0.8099084023724656, 'support': 2470}} 

## deberta-v3-mnli-fever-anli, three texts, one quote (leakage issue due to sequentiality?)(no)
# 0-shot
Aggregate metrics:  {'f1_macro': 0.35845508052405234, 'f1_micro': 0.7793522267206477, 'accuracy_balanced': 0.3845408379861527, 'accuracy_not_b': 0.7793522267206477, 'precision_macro': 0.42656749925909354, 'recall_macro': 0.3845408379861527, 'precision_micro': 0.7793522267206477, 'recall_micro': 0.7793522267206477, 
Detailed metrics:  {'immigration_neutral': {'precision': 0.16393442622950818, 'recall': 0.21739130434782608, 'f1-score': 0.18691588785046728, 'support': 46}, 'immigration_sceptical': {'precision': 0.8260869565217391, 'recall': 0.18811881188118812, 'f1-score': 0.30645161290322576, 'support': 101}, 'immigration_supportive': {'precision': 0.4519230769230769, 'recall': 0.36153846153846153, 'f1-score': 0.4017094017094017, 'support': 130}, 'integration_neutral': {'precision': 0.004484304932735426, 'recall': 0.16666666666666666, 'f1-score': 0.008733624454148471, 'support': 6}, 'integration_sceptical': {'precision': 0.14285714285714285, 'recall': 0.3, 'f1-score': 0.19354838709677416, 'support': 10}, 'integration_supportive': {'precision': 0.4262295081967213, 'recall': 0.5875706214689266, 'f1-score': 0.494061757719715, 'support': 177}, 'no_topic': {'precision': 0.9704570791527313, 'recall': 0.8705, 'f1-score': 0.9177648919346336, 'support': 2000}, 'accuracy': 0.7793522267206477, 'macro avg': {'precision': 0.42656749925909354, 'recall': 0.3845408379861527, 'f1-score': 0.35845508052405234, 'support': 2470}, 'weighted avg': {'precision': 0.8775457589998797, 'recall': 0.7793522267206477, 'f1-score': 0.8164933467894409, 'support': 2470}} 
# 20 max samples per class
Aggregate metrics:  {'f1_macro': 0.430937163110442, 'f1_micro': 0.8473684210526315, 'accuracy_balanced': 0.4708132495527849, 'accuracy_not_b': 0.8473684210526315, 'precision_macro': 0.4235699724746272, 'recall_macro': 0.4708132495527849, 'precision_micro': 0.8473684210526315, 'recall_micro': 0.8473684210526315, 
Detailed metrics:  {'immigration_neutral': {'precision': 0.1794871794871795, 'recall': 0.15217391304347827, 'f1-score': 0.16470588235294117, 'support': 46}, 'immigration_sceptical': {'precision': 0.6105263157894737, 'recall': 0.5742574257425742, 'f1-score': 0.5918367346938775, 'support': 101}, 'immigration_supportive': {'precision': 0.5959595959595959, 'recall': 0.45384615384615384, 'f1-score': 0.5152838427947597, 'support': 130}, 'integration_neutral': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6}, 'integration_sceptical': {'precision': 0.21739130434782608, 'recall': 0.5, 'f1-score': 0.30303030303030304, 'support': 10}, 'integration_supportive': {'precision': 0.38080495356037153, 'recall': 0.6949152542372882, 'f1-score': 0.49200000000000005, 'support': 177}, 'no_topic': {'precision': 0.9808204581779435, 'recall': 0.9205, 'f1-score': 0.9497033789012123, 'support': 2000}, 'accuracy': 0.8473684210526315, 'macro avg': {'precision': 0.4235699724746272, 'recall': 0.4708132495527849, 'f1-score': 0.430937163110442, 'support': 2470}, 'weighted avg': {'precision': 0.8820289966823666, 'recall': 0.8473684210526315, 'f1-score': 0.8598623243677584, 'support': 2470}} 

## deberta-v3-mnli-fever-anli, three texts, one quote (improved hypotheses and hp search over hypotheses and text format)
# zero-shot
Aggregate metrics:  {'f1_macro': 0.32835651597554777, 'f1_micro': 0.8153846153846154, 'accuracy_balanced': 0.3817549674193735, 'accuracy_not_b': 0.8153846153846154, 'precision_macro': 0.3820342652265923, 'recall_macro': 0.3817549674193735, 'precision_micro': 0.8153846153846154, 'recall_micro': 0.8153846153846154, 
Detailed metrics:  {'immigration_neutral': {'precision': 0.15789473684210525, 'recall': 0.32608695652173914, 'f1-score': 0.2127659574468085, 'support': 46}, 'immigration_sceptical': {'precision': 0.75, 'recall': 0.1485148514851485, 'f1-score': 0.24793388429752064, 'support': 101}, 'immigration_supportive': {'precision': 0.3668639053254438, 'recall': 0.47692307692307695, 'f1-score': 0.4147157190635452, 'support': 130}, 'integration_neutral': {'precision': 0.024096385542168676, 'recall': 0.3333333333333333, 'f1-score': 0.0449438202247191, 'support': 6}, 'integration_sceptical': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 10}, 'integration_supportive': {'precision': 0.4029126213592233, 'recall': 0.4689265536723164, 'f1-score': 0.43342036553524804, 'support': 177}, 'no_topic': {'precision': 0.9724722075172049, 'recall': 0.9185, 'f1-score': 0.9447158652609925, 'support': 2000}, 'accuracy': 0.8153846153846154, 'macro avg': {'precision': 0.3820342652265923, 'recall': 0.3817549674193735, 'f1-score': 0.32835651597554777, 'support': 2470}, 'weighted avg': {'precision': 0.8692753007754209, 'recall': 0.8153846153846154, 'f1-score': 0.8320479344040226, 'support': 2470}} 
# 20 class max, + 100 no topic
Epoch 	Training Loss 	Validation Loss 	F1 Macro 	F1 Micro 	Accuracy Balanced 	Accuracy Not B 	Precision Macro 	Recall Macro 	Precision Micro 	Recall Micro 	Label Gold Raw 	Label Predicted Raw
1 	1.162600 	0.256249 	0.346348 	0.866802 	0.352843 	0.866802 	0.363805 	0.352843 	0.866802 	0.866802 	
2 	0.613000 	0.239409 	0.340307 	0.867611 	0.363666 	0.867611 	0.345177 	0.363666 	0.867611 	0.867611 	
3 	0.428600 	0.281639 	0.357629 	0.871255 	0.372895 	0.871255 	0.378624 	0.372895 	0.871255 	0.871255 	
4 	0.335500 	0.251458 	0.360869 	0.872874 	0.386164 	0.872874 	0.374532 	0.386164 	0.872874 	0.872874 	
5 	0.262200 	0.273547 	0.377810 	0.870445 	0.397838 	0.870445 	0.404044 	0.397838 	0.870445 	0.870445 	
6 	0.241900 	0.251180 	0.376714 	0.868421 	0.405295 	0.868421 	0.391603 	0.405295 	0.868421 	0.868421 	
7 	0.222400 	0.305825 	0.381040 	0.874089 	0.397735 	0.874089 	0.389478 	0.397735 	0.874089 	0.874089 	
8 	0.171100 	0.312161 	0.381654 	0.863158 	0.401931 	0.863158 	0.392828 	0.401931 	0.863158 	0.863158 	
9 	0.134800 	0.407807 	0.389354 	0.871660 	0.409485 	0.871660 	0.402078 	0.409485 	0.871660 	0.871660 	# ! best
10 	0.134300 	0.305247 	0.386310 	0.870040 	0.397678 	0.870040 	0.413247 	0.397678 	0.870040 	0.870040

## cross-lingual transfer, only eng training & hypos, mdeberta-v3-mnli-xnli, hp search (3 text), eval on all lang, up to 1k examples per class
# zero-shot
Aggregate metrics:  {'f1_macro': 0.1719893691402557, 'f1_micro': 0.25299739357080797, 'accuracy_balanced': 0.21984094561458065, 'accuracy_not_b': 0.25299739357080797, 'precision_macro': 0.2868417953588206, 'recall_macro': 0.21984094561458065, 'precision_micro': 0.25299739357080797, 'recall_micro': 0.25299739357080797}
Detailed metrics:  {'immigration_neutral': {'precision': 0.14772727272727273, 'recall': 0.023255813953488372, 'f1-score': 0.0401854714064915, 'support': 559}, 'immigration_sceptical': {'precision': 0.35507246376811596, 'recall': 0.049, 'f1-score': 0.08611599297012303, 'support': 1000}, 'immigration_supportive': {'precision': 0.21465773809523808, 'recall': 0.577, 'f1-score': 0.3129067245119306, 'support': 1000}, 'integration_neutral': {'precision': 0.09090909090909091, 'recall': 0.030927835051546393, 'f1-score': 0.046153846153846156, 'support': 388}, 'integration_sceptical': {'precision': 0.31520223152022314, 'recall': 0.27970297029702973, 'f1-score': 0.2963934426229508, 'support': 808}, 'integration_supportive': {'precision': 0.2843237704918033, 'recall': 0.555, 'f1-score': 0.3760162601626017, 'support': 1000}, 'no_topic': {'precision': 0.6, 'recall': 0.024, 'f1-score': 0.04615384615384615, 'support': 1000}, 'accuracy': 0.25299739357080797, 'macro avg': {'precision': 0.2868417953588206, 'recall': 0.21984094561458065, 'f1-score': 0.1719893691402557, 'support': 5755}, 'weighted avg': {'precision': 0.3173917720505248, 'recall': 0.25299739357080797, 'f1-score': 0.1913206075172488, 'support': 5755}} 
# 20 class max, + 50 no topic
Aggregate metrics:  {'f1_macro': 0.29884915222692277, 'f1_micro': 0.36854908774978284, 'accuracy_balanced': 0.33716972126814243, 'accuracy_not_b': 0.3685490877497828, 'precision_macro': 0.3588243610849638, 'recall_macro': 0.33716972126814243, 'precision_micro': 0.3685490877497828, 'recall_micro': 0.3685490877497828}
Detailed metrics:  {'immigration_neutral': {'precision': 0.16531961792799413, 'recall': 0.40250447227191416, 'f1-score': 0.234375, 'support': 559}, 'immigration_sceptical': {'precision': 0.33100069979006297, 'recall': 0.473, 'f1-score': 0.3894606834088102, 'support': 1000}, 'immigration_supportive': {'precision': 0.4288256227758007, 'recall': 0.241, 'f1-score': 0.3085787451984635, 'support': 1000}, 'integration_neutral': {'precision': 0.11555555555555555, 'recall': 0.06701030927835051, 'f1-score': 0.08482871125611745, 'support': 388}, 'integration_sceptical': {'precision': 0.2909698996655518, 'recall': 0.10767326732673267, 'f1-score': 0.15718157181571815, 'support': 808}, 'integration_supportive': {'precision': 0.6172248803827751, 'recall': 0.129, 'f1-score': 0.21339950372208438, 'support': 1000}, 'no_topic': {'precision': 0.562874251497006, 'recall': 0.94, 'f1-score': 0.7041198501872659, 'support': 1000}, 'accuracy': 0.3685490877497828, 'macro avg': {'precision': 0.3588243610849638, 'recall': 0.33716972126814243, 'f1-score': 0.29884915222692277, 'support': 5755}, 'weighted avg': {'precision': 0.4017859870291424, 'recall': 0.3685490877497828, 'f1-score': 0.33127552693503004, 'support': 5755}} 
# ! it predicts "no_topic" too often. probably because it takes this as the "semantically dissimilar class" and puts other languages/scripts into it

## multiling training & only eng hypos, mdeberta-v3-mnli-xnli, hp search, eval on all lang, up to 1k examples per class
# zero-shot
Aggregate metrics:  {'f1_macro': 0.17577228246366625, 'f1_micro': 0.2647276769173768, 'accuracy_balanced': 0.22240540044285978, 'accuracy_not_b': 0.2647276769173768, 'precision_macro': 0.28014060603616253, 'recall_macro': 0.22240540044285978, 'precision_micro': 0.2647276769173768, 'recall_micro': 0.2647276769173768}
Detailed metrics:  {'immigration_neutral': {'precision': 0.14473684210526316, 'recall': 0.025056947608200455, 'f1-score': 0.04271844660194175, 'support': 439}, 'immigration_sceptical': {'precision': 0.4330708661417323, 'recall': 0.055, 'f1-score': 0.09760425909494233, 'support': 1000}, 'immigration_supportive': {'precision': 0.23151750972762647, 'recall': 0.595, 'f1-score': 0.3333333333333333, 'support': 1000}, 'integration_neutral': {'precision': 0.07627118644067797, 'recall': 0.03237410071942446, 'f1-score': 0.045454545454545456, 'support': 278}, 'integration_sceptical': {'precision': 0.3088, 'recall': 0.2834067547723935, 'f1-score': 0.2955589586523737, 'support': 681}, 'integration_supportive': {'precision': 0.29783783783783785, 'recall': 0.551, 'f1-score': 0.3866666666666667, 'support': 1000}, 'no_topic': {'precision': 0.46875, 'recall': 0.015, 'f1-score': 0.02906976744186046, 'support': 1000}, 'accuracy': 0.2647276769173768, 'macro avg': {'precision': 0.28014060603616253, 'recall': 0.22240540044285978, 'f1-score': 0.17577228246366625, 'support': 5398}, 'weighted avg': {'precision': 0.31978730589513077, 'recall': 0.2647276769173768, 'f1-score': 0.19995173009886727, 'support': 5398}} 
# 20 class max per lang, + 50 no topic
Aggregate metrics:  {'f1_macro': 0.414688984645728, 'f1_micro': 0.5355687291589477, 'accuracy_balanced': 0.4363537212393535, 'accuracy_not_b': 0.5355687291589477, 'precision_macro': 0.46201265089166155, 'recall_macro': 0.4363537212393535, 'precision_micro': 0.5355687291589477, 'recall_micro': 0.5355687291589477}
Detailed metrics:  {'immigration_neutral': {'precision': 0.15384615384615385, 'recall': 0.004555808656036446, 'f1-score': 0.008849557522123894, 'support': 439}, 'immigration_sceptical': {'precision': 0.5542168674698795, 'recall': 0.414, 'f1-score': 0.47395535203205497, 'support': 1000}, 'immigration_supportive': {'precision': 0.4044421487603306, 'recall': 0.783, 'f1-score': 0.5333787465940055, 'support': 1000}, 'integration_neutral': {'precision': 0.3, 'recall': 0.04316546762589928, 'f1-score': 0.07547169811320754, 'support': 278}, 'integration_sceptical': {'precision': 0.48596491228070177, 'recall': 0.4067547723935389, 'f1-score': 0.442845723421263, 'support': 681}, 'integration_supportive': {'precision': 0.5386119257086999, 'recall': 0.551, 'f1-score': 0.5447355412753336, 'support': 1000}, 'no_topic': {'precision': 0.7970065481758652, 'recall': 0.852, 'f1-score': 0.8235862735621073, 'support': 1000}, 'accuracy': 0.5355687291589477, 'macro avg': {'precision': 0.46201265089166155, 'recall': 0.4363537212393535, 'f1-score': 0.414688984645728, 'support': 5398}, 'weighted avg': {'precision': 0.514293823067135, 'recall': 0.5355687291589477, 'f1-score': 0.5005742754614794, 'support': 5398}} 


# In[ ]:


450 + 176 + 170 + 155 + 140 + 137 + 116


# In[ ]:


## save study  # https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-save-and-resume-studies
import joblib
os.getcwd()
#joblib.dump(experiment_details_dic, f"./results/{TRAINING_DIRECTORY}/results_minilm.pkl")
## load results
#experiment_details_dic = joblib.load(f"./results/{TRAINING_DIRECTORY}/results_minilm.pkl")


# In[ ]:


for experiment_key, experiment_value in experiment_details_dic.items():
  label_gold_trial = []
  label_predicted_trial = []
  #for dic in study.best_trial.user_attrs["results_trainer"]:
  for seed_key, seed_value in experiment_value.items():  # iterate over each random repetition and add predictions to get aggregate scores
    label_gold_trial = label_gold_trial + seed_value["metrics"]["eval_label_gold_raw"]
    label_predicted_trial = label_predicted_trial + seed_value["metrics"]["eval_label_predicted_raw"]
  assert len(label_gold_trial) == len(label_predicted_trial)

  # map numerical labels to label texts
  label_num_text_map = {label_num: label_text for label_num, label_text in enumerate(np.sort(df_cl.label_text.unique()))}
  label_gold_trial_text = [label_num_text_map[label] for label in label_gold_trial]
  label_predicted_trial_text = [label_num_text_map[label] for label in label_predicted_trial]

  # sklearn plot https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import matplotlib.pyplot as plt
  conf_mat = confusion_matrix(label_gold_trial_text, label_predicted_trial_text, normalize=None)  #labels=np.sort(df_cl.label_text.unique())  
  disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=np.sort(df_cl.label_text.unique()))
  plt.rcParams["figure.figsize"] = (9,9)
  disp.plot(xticks_rotation="vertical")
  print(experiment_key)
  plt.show()


# In[ ]:


df_train_test_dist["predicted"] = pd.Series(label_predicted_trial_text).value_counts()
df_train_test_dist


# In[ ]:


#print(metrics_mean_lst)
#print(metrics_std_lst)
#print(N_MAX_SAMPLE_PER_CLASS_TEST)

n_samples = [0, 16, 32, 64, 128]
n_classes = len(LABEL_TEXT_ALPHABETICAL)
x_axis_values = [f"{n} (total {n*n_classes})" if n != 0 else "total 0 " for n in n_samples]
print(x_axis_values)

## minilm-l6-mnli-binary
metric_mean_nli = [0.3475035427566548, 0.4625005742342056, 0.49383885471246053, 0.5439162358413432, 0.5736261204898448]
metric_std_nli = [0.0, 0.01852183425314923, 0.01884262684778947, 0.008509306174886674, 0.028343294051863872]
## minilm-l6
metric_mean_standard = [0.00402549480040255, 0.026954602974704724, 0.03488313042861732, 0.04668561628429598, 0.08554193202830433]
metric_std_standard = [0.0, 0.00669269208694209, 0.01135578906594988, 0.004719000422789446, 0.008543591490249747]


# In[ ]:


### visualisation   # https://plotly.com/python/continuous-error-bars/
import plotly.graph_objs as go

fig = go.Figure([
    go.Scatter(
        name='MiniLM-NLI',
        x=x_axis_values,
        y=metric_mean_nli,
        mode='lines',
        line=dict(color='#f20808'),        
        showlegend=True
    ),
    go.Scatter(
        name='Upper Bound MiniLM-NLI',
        x=x_axis_values,
        y=pd.Series(metric_mean_nli) + pd.Series(metric_std_nli),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound MiniLM-NLI',
        x=x_axis_values,
        y=pd.Series(metric_mean_nli) - pd.Series(metric_std_nli),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name='MiniLM',
        x=x_axis_values,
        y=metric_mean_standard,
        mode='lines',
        line=dict(color='#0872f2'),        
        showlegend=True
    ),
    go.Scatter(
        name='Upper Bound MiniLM',
        x=x_axis_values,
        y=pd.Series(metric_mean_standard) + pd.Series(metric_std_standard),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound MiniLM',
        x=x_axis_values,
        y=pd.Series(metric_mean_standard) - pd.Series(metric_std_standard),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
])
fig.update_layout(
    yaxis_title='F1 macro',
    xaxis_title=f'N random examples given {n_classes} classes',
    title='CAP SOTU dataset',
    hovermode="x",
    autosize=False,
    width=600,
    height=500,
    title_x=0.5,
)
fig.update_xaxes(
    ticktext=x_axis_values,
    tickvals=x_axis_values,
    tickangle=-45,
)
fig.update_yaxes(range = [0, max(pd.Series(metric_mean_nli)+pd.Series(metric_std_nli) + 0.1)])  # adjust range of Y axis display
fig.update_yaxes(automargin=True)
fig.update_xaxes(type="category")
fig.show()



# #### sklearn hyperparameter tuning with cross-validation

# In[ ]:


#!pip install spacy
#!python -m spacy download en_core_web_sm


# In[ ]:


from sklearn.model_selection import ShuffleSplit, train_test_split, cross_val_score
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler, Normalizer
import spacy

df_cl_samp = df_cl.groupby(by="label_cap2", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), 20), random_state=42))
print(len(df_cl_samp))

## lemmatize text
nlp = spacy.load("en_core_web_sm")
texts_lemma = []
for doc in nlp.pipe(df_cl_samp.text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"]):
    doc_lemmas = " ".join([token.lemma_ for token in doc])
    texts_lemma.append(doc_lemmas)

## count-vectorize & tfidf texts
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), norm="l2", use_idf=True, smooth_idf=True, analyzer="word", max_df=1.0, min_df=10)
X = vectorizer.fit_transform(texts_lemma)
y = df_cl_samp.label

clf = svm.SVC(kernel='linear', C=1, random_state=42)

split_strategy = ShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
scores = cross_val_score(clf, X, y, cv=split_strategy, scoring='f1_macro')

print(scores)
print("%0.2f f1_macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[ ]:


## Notes: 
# first tests on short-hypo

## 0-shot, minilm-l6-mnli-binary - single sentences
Aggregate metrics:  {'accuracy_balanced': 0.6434277364132586, 'accuracy_not_b': 0.5625785849715598, 'f1_macro': 0.43404536640754016, 'f1_micro': 0.5625785849715598, 'precision_macro': 0.5317636994458071, 'recall_macro': 0.6434277364132586, 'precision_micro': 0.5625785849715598, 'recall_micro': 0.5625785849715598}
Detailed metrics:  {0: {'precision': 0.971018276762402, 'recall': 0.5518622941089182, 'f1-score': 0.7037562683319141, 'support': 47173}, 1: {'precision': 0.09250912212921228, 'recall': 0.734993178717599, 'f1-score': 0.16433446448316621, 'support': 2932}, 'accuracy': 0.5625785849715598, 'macro avg': {'precision': 0.5317636994458071, 'recall': 0.6434277364132586, 'f1-score': 0.43404536640754016, 'support': 50105}, 'weighted avg': {'precision': 0.919610456357562, 'recall': 0.5625785849715597, 'f1-score': 0.6721908611093907, 'support': 50105}} 
## 0-shot, minilm-l6-mnli-binary - 2 sentences
Aggregate metrics:  {'accuracy_balanced': 0.6397582144071516, 'accuracy_not_b': 0.5452423317390098, 'f1_macro': 0.42215428593853055, 'f1_micro': 0.5452423317390098, 'precision_macro': 0.5296340553336686, 'recall_macro': 0.6397582144071516, 'precision_micro': 0.5452423317390098, 'recall_micro': 0.5452423317390098}
Detailed metrics:  {0: {'precision': 0.9724999211580309, 'recall': 0.5332998979644779, 'f1-score': 0.6888486798016352, 'support': 57823}, 1: {'precision': 0.08676818950930626, 'recall': 0.7462165308498254, 'f1-score': 0.15545989207542593, 'support': 3436}, 'accuracy': 0.5452423317390098, 'macro avg': {'precision': 0.5296340553336686, 'recall': 0.6397582144071516, 'f1-score': 0.42215428593853055, 'support': 61259}, 'weighted avg': {'precision': 0.9228194786117109, 'recall': 0.5452423317390098, 'f1-score': 0.6589310534181283, 'support': 61259}} 

## 0-shot, ynie/roberta-large-allnli
## 0-shot, DeBERTa-v3-large-mnli


## 8-shot, minilm-l6-mnli-binary - 2 sentences, 4 epochs
Aggregate metrics:  {'accuracy_balanced': 0.6772191034914727, 'accuracy_not_b': 0.5544654662988295, 'f1_macro': 0.4329059405108773, 'f1_micro': 0.5544654662988295, 'precision_macro': 0.5375852134929361, 'recall_macro': 0.6772191034914727, 'precision_micro': 0.5544654662988295, 'recall_micro': 0.5544654662988295}
Detailed metrics:  {0: {'precision': 0.9800616390967986, 'recall': 0.538955087076077, 'f1-score': 0.6954620010934937, 'support': 57823}, 1: {'precision': 0.09510878788907369, 'recall': 0.8154831199068685, 'f1-score': 0.17034987992826092, 'support': 3436}, 'accuracy': 0.5544654662988295, 'macro avg': {'precision': 0.5375852134929361, 'recall': 0.6772191034914727, 'f1-score': 0.4329059405108773, 'support': 61259}, 'weighted avg': {'precision': 0.9304248837343254, 'recall': 0.5544654662988295, 'f1-score': 0.6660086105986481, 'support': 61259}} 


## 20-shot, minilm-l6-mnli-binary - single sentences, 4 epochs
Aggregate metrics:  {'accuracy_balanced': 0.6483584999558532, 'accuracy_not_b': 0.47308651831154575, 'f1_macro': 0.38741173623210196, 'f1_micro': 0.47308651831154575, 'precision_macro': 0.5333010365612552, 'recall_macro': 0.6483584999558532, 'precision_micro': 0.47308651831154575, 'recall_micro': 0.47308651831154575}
Detailed metrics:  {0: {'precision': 0.979280110752192, 'recall': 0.4498547898162084, 'f1-score': 0.6165042197463795, 'support': 47173}, 1: {'precision': 0.08732196237031827, 'recall': 0.846862210095498, 'f1-score': 0.15831925271782443, 'support': 2932}, 'accuracy': 0.47308651831154575, 'macro avg': {'precision': 0.5333010365612552, 'recall': 0.6483584999558532, 'f1-score': 0.38741173623210196, 'support': 50105}, 'weighted avg': {'precision': 0.9270852940461615, 'recall': 0.47308651831154575, 'f1-score': 0.5896925577699754, 'support': 50105}} 
## 20-shot, minilm-l6-mnli-binary - 2 sentences, 4 epochs
Aggregate metrics:  {'accuracy_balanced': 0.6689166753254889, 'accuracy_not_b': 0.5310403369300838, 'f1_macro': 0.41935083622813174, 'f1_micro': 0.5310403369300838, 'precision_macro': 0.5357763340060794, 'recall_macro': 0.6689166753254889, 'precision_micro': 0.5310403369300838, 'recall_micro': 0.5310403369300838}
Detailed metrics:  {0: {'precision': 0.98006798006798, 'recall': 0.5136191480898604, 'f1-score': 0.6740122097905272, 'support': 57823}, 1: {'precision': 0.09148468794417884, 'recall': 0.8242142025611175, 'f1-score': 0.16468946266573622, 'support': 3436}, 'accuracy': 0.5310403369300838, 'macro avg': {'precision': 0.5357763340060794, 'recall': 0.6689166753254889, 'f1-score': 0.41935083622813174, 'support': 61259}, 'weighted avg': {'precision': 0.9302275943003805, 'recall': 0.5310403369300838, 'f1-score': 0.6454444408239953, 'support': 61259}} 
## 20-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs
Aggregate metrics:  {'accuracy_balanced': 0.6776378702119674, 'accuracy_not_b': 0.5513802053575801, 'f1_macro': 0.43136559893784365, 'f1_micro': 0.5513802053575801, 'precision_macro': 0.5376553824622781, 'recall_macro': 0.6776378702119674, 'precision_micro': 0.5513802053575801, 'recall_micro': 0.5513802053575801}
Detailed metrics:  {0: {'precision': 0.9803983660027233, 'recall': 0.5354270791899417, 'f1-score': 0.69260195521353, 'support': 57823}, 1: {'precision': 0.09491239892183288, 'recall': 0.819848661233993, 'f1-score': 0.1701292426621573, 'support': 3436}, 'accuracy': 0.5513802053575801, 'macro avg': {'precision': 0.5376553824622781, 'recall': 0.6776378702119674, 'f1-score': 0.43136559893784365, 'support': 61259}, 'weighted avg': {'precision': 0.9307317083215674, 'recall': 0.5513802053575801, 'f1-score': 0.6632966084020163, 'support': 61259}} 
## 20-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs, & 20 nsp samples per class
Aggregate metrics:  {'accuracy_balanced': 0.6787854653266561, 'accuracy_not_b': 0.5380433895427611, 'f1_macro': 0.42454133751153933, 'f1_micro': 0.5380433895427611, 'precision_macro': 0.5378622345978273, 'recall_macro': 0.6787854653266561, 'precision_micro': 0.5380433895427611, 'recall_micro': 0.5380433895427611}
Detailed metrics:  {0: {'precision': 0.9817570654657006, 'recall': 0.5202601041108209, 'f1-score': 0.6801107782738935, 'support': 57823}, 1: {'precision': 0.09396740372995395, 'recall': 0.8373108265424912, 'f1-score': 0.16897189674918509, 'support': 3436}, 'accuracy': 0.5380433895427611, 'macro avg': {'precision': 0.5378622345978273, 'recall': 0.6787854653266561, 'f1-score': 0.42454133751153933, 'support': 61259}, 'weighted avg': {'precision': 0.9319611942023103, 'recall': 0.5380433895427611, 'f1-score': 0.6514411428420567, 'support': 61259}} 
## 20-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs, & 40 nsp samples per class
Aggregate metrics:  {'accuracy_balanced': 0.6841061363310622, 'accuracy_not_b': 0.5470543103870452, 'f1_macro': 0.4301106410126855, 'f1_micro': 0.5470543103870452, 'precision_macro': 0.5390018861711284, 'recall_macro': 0.6841061363310622, 'precision_micro': 0.5470543103870452, 'recall_micro': 0.5470543103870452}
Detailed metrics:  {0: {'precision': 0.9822035528762907, 'recall': 0.5297373017657333, 'f1-score': 0.6882674785695829, 'support': 57823}, 1: {'precision': 0.09580021946596615, 'recall': 0.8384749708963911, 'f1-score': 0.17195380345578798, 'support': 3436}, 'accuracy': 0.5470543103870452, 'macro avg': {'precision': 0.5390018861711284, 'recall': 0.6841061363310622, 'f1-score': 0.4301106410126855, 'support': 61259}, 'weighted avg': {'precision': 0.9324854403769376, 'recall': 0.5470543103870452, 'f1-score': 0.6593075904275794, 'support': 61259}} 
## 20-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs, & 80 nsp samples per class
Aggregate metrics:  {'accuracy_balanced': 0.6870666507724177, 'accuracy_not_b': 0.5456667591700811, 'f1_macro': 0.4298358809592259, 'f1_micro': 0.5456667591700811, 'precision_macro': 0.5396233351952257, 'recall_macro': 0.6870666507724177, 'precision_micro': 0.5456667591700811, 'recall_micro': 0.5456667591700811}
Detailed metrics:  {0: {'precision': 0.9829935259445357, 'recall': 0.5278003562596199, 'f1-score': 0.6868234499831214, 'support': 57823}, 1: {'precision': 0.09625314444591553, 'recall': 0.8463329452852154, 'f1-score': 0.17284831193533048, 'support': 3436}, 'accuracy': 0.5456667591700811, 'macro avg': {'precision': 0.5396233351952257, 'recall': 0.6870666507724177, 'f1-score': 0.4298358809592259, 'support': 61259}, 'weighted avg': {'precision': 0.9332565085131499, 'recall': 0.5456667591700811, 'f1-score': 0.6579947297243478, 'support': 61259}} 

## 20-shot, ynie/roberta-large-allnli
## 20-shot, DeBERTa-v3-large-mnli
## 20-shot, standard_dl, minilm-l6


## 200-shot, minilm-l6-mnli-binary - single sentences, 4 epochs
Aggregate metrics:  {'accuracy_balanced': 0.6889562681660947, 'accuracy_not_b': 0.514898712703323, 'f1_macro': 0.41618594239990564, 'f1_micro': 0.514898712703323, 'precision_macro': 0.5417939105100477, 'recall_macro': 0.6889562681660947, 'precision_micro': 0.514898712703323, 'recall_micro': 0.514898712703323}
Detailed metrics:  {0: {'precision': 0.9858083705120034, 'recall': 0.4918279524304157, 'f1-score': 0.6562482321660915, 'support': 47173}, 1: {'precision': 0.09777945050809184, 'recall': 0.8860845839017736, 'f1-score': 0.17612365263371974, 'support': 2932}, 'accuracy': 0.514898712703323, 'macro avg': {'precision': 0.5417939105100477, 'recall': 0.6889562681660947, 'f1-score': 0.41618594239990564, 'support': 50105}, 'weighted avg': {'precision': 0.9338434809111359, 'recall': 0.514898712703323, 'f1-score': 0.628152727382359, 'support': 50105}} 
## 200-shot, minilm-l6-mnli-binary - 2 sentences, 4 epochs
Aggregate metrics:  {'accuracy_balanced': 0.7073986670654859, 'accuracy_not_b': 0.5197113893468714, 'f1_macro': 0.4188255678318088, 'f1_micro': 0.5197113893468714, 'precision_macro': 0.5440528137517363, 'recall_macro': 0.7073986670654859, 'precision_micro': 0.5197113893468714, 'recall_micro': 0.5197113893468714}
Detailed metrics:  {0: {'precision': 0.9903656894229773, 'recall': 0.4959964028154886, 'f1-score': 0.6609665598856905, 'support': 57823}, 1: {'precision': 0.09773993808049536, 'recall': 0.9188009313154831, 'f1-score': 0.17668457577792704, 'support': 3436}, 'accuracy': 0.5197113893468714, 'macro avg': {'precision': 0.5440528137517363, 'recall': 0.7073986670654859, 'f1-score': 0.4188255678318088, 'support': 61259}, 'weighted avg': {'precision': 0.9402985632600825, 'recall': 0.5197113893468714, 'f1-score': 0.6338033202409971, 'support': 61259}} 
## 200-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs
Aggregate metrics:  {'accuracy_balanced': 0.7016113029854243, 'accuracy_not_b': 0.5023261888049103, 'f1_macro': 0.40840872827445235, 'f1_micro': 0.5023261888049103, 'precision_macro': 0.5430522210241567, 'recall_macro': 0.7016113029854243, 'precision_micro': 0.5023261888049103, 'recall_micro': 0.5023261888049103}
Detailed metrics:  {0: {'precision': 0.9908777474500791, 'recall': 0.4771457724434913, 'f1-score': 0.6441220073073645, 'support': 57823}, 1: {'precision': 0.09522669459823432, 'recall': 0.9260768335273574, 'f1-score': 0.17269544924154023, 'support': 3436}, 'accuracy': 0.5023261888049103, 'macro avg': {'precision': 0.5430522210241567, 'recall': 0.7016113029854243, 'f1-score': 0.40840872827445235, 'support': 61259}, 'weighted avg': {'precision': 0.9406409329803858, 'recall': 0.5023261888049103, 'f1-score': 0.6176798248767964, 'support': 61259}} 
## 200-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs, & 100 nsp samples per class
Aggregate metrics:  {'accuracy_balanced': 0.691587361350041, 'accuracy_not_b': 0.489087317781877, 'f1_macro': 0.39967960667903196, 'f1_micro': 0.489087317781877, 'precision_macro': 0.5411265819572748, 'recall_macro': 0.691587361350041, 'precision_micro': 0.489087317781877, 'recall_micro': 0.489087317781877}
Detailed metrics:  {0: {'precision': 0.9898068471396388, 'recall': 0.46350068311917403, 'f1-score': 0.6313545347467608, 'support': 57823}, 1: {'precision': 0.09244631677491078, 'recall': 0.919674039580908, 'f1-score': 0.1680046786113031, 'support': 3436}, 'accuracy': 0.489087317781877, 'macro avg': {'precision': 0.5411265819572748, 'recall': 0.691587361350041, 'f1-score': 0.39967960667903196, 'support': 61259}, 'weighted avg': {'precision': 0.9394741485592962, 'recall': 0.489087317781877, 'f1-score': 0.6053653722452276, 'support': 61259}} 
## 200-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs, & 200 nsp samples per class
Aggregate metrics:  {'accuracy_balanced': 0.7053898269934077, 'accuracy_not_b': 0.5169526110449076, 'f1_macro': 0.417027327986376, 'f1_micro': 0.5169526110449076, 'precision_macro': 0.5436524358112617, 'recall_macro': 0.7053898269934077, 'precision_micro': 0.5169526110449076, 'recall_micro': 0.5169526110449076}
Detailed metrics:  {0: {'precision': 0.9901729286756025, 'recall': 0.4931428670252322, 'f1-score': 0.6583853799886864, 'support': 57823}, 1: {'precision': 0.09713194294692092, 'recall': 0.9176367869615832, 'f1-score': 0.1756692759840655, 'support': 3436}, 'accuracy': 0.5169526110449076, 'macro avg': {'precision': 0.5436524358112617, 'recall': 0.7053898269934077, 'f1-score': 0.417027327986376, 'support': 61259}, 'weighted avg': {'precision': 0.940082512133319, 'recall': 0.5169526110449076, 'f1-score': 0.6313099701165064, 'support': 61259}} 
## 200-shot, minilm-l6-mnli-binary - 2 sentences, 2 epochs, & 400 nsp samples per class
Aggregate metrics:  {'accuracy_balanced': 0.7009768525670357, 'accuracy_not_b': 0.5081049315202664, 'f1_macro': 0.41153434322881044, 'f1_micro': 0.5081049315202664, 'precision_macro': 0.5428197931671495, 'recall_macro': 0.7009768525670357, 'precision_micro': 0.5081049315202664, 'recall_micro': 0.5081049315202664}
Detailed metrics:  {0: {'precision': 0.9900538015007787, 'recall': 0.4837348459955381, 'f1-score': 0.6499215800174266, 'support': 57823}, 1: {'precision': 0.09558578483352016, 'recall': 0.9182188591385332, 'f1-score': 0.17314710644019426, 'support': 3436}, 'accuracy': 0.5081049315202664, 'macro avg': {'precision': 0.5428197931671495, 'recall': 0.7009768525670357, 'f1-score': 0.41153434322881044, 'support': 61259}, 'weighted avg': {'precision': 0.9398833431963876, 'recall': 0.5081049315202664, 'f1-score': 0.623179434516988, 'support': 61259}} 


## 200-shot, ynie/roberta-large-allnli
## 200-shot, DeBERTa-v3-large-mnli
## 200-shot, standard_dl, minilm-l6





# In[ ]:


#trainer.save_model(output_dir = f'./results/{training_directory}')


# In[ ]:


### tensor board
#%load_ext tensorboard
#%tensorboard --logdir logs/coronanet_classi_distilbert  # "logs" is the directory with  my training logs


# ## Classical Machine Learning

# In[ ]:



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import svm, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import spacy

df_cl_samp = df_cl.groupby(by="label_cap2", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), 200_000_0), random_state=42))
print(len(df_cl_samp))

## lemmatize text
nlp = spacy.load("en_core_web_sm")
texts_lemma = []
for doc in nlp.pipe(df_cl_samp.text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"]):
    doc_lemmas = " ".join([token.lemma_ for token in doc])
    texts_lemma.append(doc_lemmas)

## count-vectorize & tfidf texts
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), norm="l2", use_idf=True, smooth_idf=True, analyzer="word", max_df=1.0, min_df=10)
X = vectorizer.fit_transform(texts_lemma)
y = df_cl_samp.label

# ! is it data leakage to do TFIDF on entire set? (but if only on train, some vocab missing. important issue of method?)

## create train-test-val split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, shuffle=True, random_state=42, stratify=y_train)


# In[ ]:


### fit model with standard hyperparams

#clf = svm.SVC()
#clf = naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
clf = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

clf.fit(X_train, y_train)


# In[ ]:


### Hyperparamter search
# GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import GridSearchCV
"""hyperparams = {'kernel': ["linear", "poly", "rbf", "sigmoid"], 
               'C': [1, 2, 5, 10],
               "gamma": ["scale", "auto"],
               "class_weight": ["balanced", None],
               "decision_function_shape": ["ovo", "ovr"],
               }
svc = svm.SVC()
# scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
clf = GridSearchCV(svc, hyperparams, 
                   scoring=["balanced_accuracy", "accuracy", "f1_macro", "f1_micro"],
                   n_jobs=-1,
                   refit="f1_macro",
                   cv=3,
                   verbose=3,
                   error_score="raise", # ‘raise’ or numeric, default=np.nan
                   return_train_score=False,
                   )
clf.fit(X_train, y_train)

# output
clf.best_score_
clf.best_params_
clf.best_estimator_  # available if refit=True
clf.cv_results_"""


# In[ ]:


from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report

def compute_metrics_sklearn(X, y):
    labels = y
    preds_max = clf.predict(X)
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds_max, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds_max, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(labels, preds_max)
    acc_not_balanced = accuracy_score(labels, preds_max)
    metrics = {'accuracy_balanced': acc_balanced,
               'accuracy_not_b': acc_not_balanced,
               'f1_macro': f1_macro,
               'f1_micro': f1_micro,
               'precision_macro': precision_macro,
               'recall_macro': recall_macro,
               'precision_micro': precision_micro,
               'recall_micro': recall_micro
               }
    print("Aggregate metrics: ", metrics)
    print("Detailed metrics: ", classification_report(labels, preds_max, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn'), "\n")
    return metrics


# In[ ]:


compute_metrics_sklearn(X_val, y_val)


# In[ ]:


## Support Vector Machine, standard params
Aggregate metrics:  {'accuracy_balanced': 0.5609516353748466, 'accuracy_not_b': 0.7421020028907702, 'f1_macro': 0.6190483985692911, 'precision_macro': 0.7749940419813355, 'recall_macro': 0.5609516353748466, 'f1_micro': 0.7421020028907702, 'precision_micro': 0.7421020028907702, 'recall_micro': 0.7421020028907702}
Detailed metrics:  {'0': {'precision': 1.0, 'recall': 0.1724137931034483, 'f1-score': 0.29411764705882354, 'support': 29}, '1': {'precision': 0.675, 'recall': 0.4778761061946903, 'f1-score': 0.5595854922279793, 'support': 113}, '2': {'precision': 0.6188118811881188, 'recall': 0.6410256410256411, 'f1-score': 0.6297229219143576, 'support': 195}, '3': {'precision': 0.717948717948718, 'recall': 0.5490196078431373, 'f1-score': 0.6222222222222223, 'support': 51}, '4': {'precision': 0.7282608695652174, 'recall': 0.46206896551724136, 'f1-score': 0.5654008438818566, 'support': 145}, '5': {'precision': 0.8846153846153846, 'recall': 0.35384615384615387, 'f1-score': 0.5054945054945055, 'support': 65}, '6': {'precision': 0.9230769230769231, 'recall': 0.5853658536585366, 'f1-score': 0.7164179104477613, 'support': 41}, '7': {'precision': 0.7706422018348624, 'recall': 0.828169014084507, 'f1-score': 0.7983706720977597, 'support': 710}, '8': {'precision': 0.9421487603305785, 'recall': 0.7755102040816326, 'f1-score': 0.8507462686567164, 'support': 147}, '9': {'precision': 0.7567567567567568, 'recall': 0.5957446808510638, 'f1-score': 0.6666666666666666, 'support': 47}, '10': {'precision': 0.8518518518518519, 'recall': 0.42592592592592593, 'f1-score': 0.5679012345679013, 'support': 54}, '11': {'precision': 1.0, 'recall': 0.05, 'f1-score': 0.09523809523809523, 'support': 20}, '12': {'precision': 0.8333333333333334, 'recall': 0.36585365853658536, 'f1-score': 0.5084745762711864, 'support': 41}, '13': {'precision': 0.8361669242658424, 'recall': 0.8754045307443366, 'f1-score': 0.8553359683794466, 'support': 618}, '14': {'precision': 0.8076923076923077, 'recall': 0.8048780487804879, 'f1-score': 0.806282722513089, 'support': 287}, '15': {'precision': 0.925, 'recall': 0.7254901960784313, 'f1-score': 0.8131868131868133, 'support': 51}, '16': {'precision': 0.643652561247216, 'recall': 0.8592666005946482, 'f1-score': 0.7359932088285228, 'support': 1009}, '17': {'precision': 0.56, 'recall': 0.30434782608695654, 'f1-score': 0.3943661971830986, 'support': 46}, '18': {'precision': 0.6426666666666667, 'recall': 0.7280966767371602, 'f1-score': 0.6827195467422097, 'support': 331}, '19': {'precision': 0.8, 'recall': 0.8266666666666667, 'f1-score': 0.8131147540983606, 'support': 150}, '20': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 23}, '21': {'precision': 0.75, 'recall': 0.27906976744186046, 'f1-score': 0.4067796610169491, 'support': 43}, '22': {'precision': 0.8709677419354839, 'recall': 0.627906976744186, 'f1-score': 0.7297297297297297, 'support': 43}, '23': {'precision': 0.7945205479452054, 'recall': 0.5132743362831859, 'f1-score': 0.6236559139784946, 'support': 113}, '24': {'precision': 0.89171974522293, 'recall': 0.8484848484848485, 'f1-score': 0.8695652173913044, 'support': 165}, '25': {'precision': 0.8, 'recall': 0.7787610619469026, 'f1-score': 0.7892376681614349, 'support': 113}, '26': {'precision': 0.7375, 'recall': 0.5462962962962963, 'f1-score': 0.6276595744680851, 'support': 108}, '27': {'precision': 0.9375, 'recall': 0.7058823529411765, 'f1-score': 0.8053691275167786, 'support': 85}, 'accuracy': 0.7421020028907702, 'macro avg': {'precision': 0.7749940419813355, 'recall': 0.5609516353748466, 'f1-score': 0.6190483985692911, 'support': 4843}, 'weighted avg': {'precision': 0.7526129242076539, 'recall': 0.7421020028907702, 'f1-score': 0.7321056774204774, 'support': 4843}} 

## GaussianNB, standard params
Aggregate metrics:  {'accuracy_balanced': 0.33964269281299353, 'accuracy_not_b': 0.6582696675614289, 'f1_macro': 0.37891683746700233, 'f1_micro': 0.6582696675614289, 'precision_macro': 0.6600952797353202, 'recall_macro': 0.33964269281299353, 'precision_micro': 0.6582696675614289, 'recall_micro': 0.6582696675614289}
Detailed metrics:  {'0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 29}, '1': {'precision': 0.7058823529411765, 'recall': 0.21238938053097345, 'f1-score': 0.32653061224489793, 'support': 113}, '2': {'precision': 0.6357615894039735, 'recall': 0.49230769230769234, 'f1-score': 0.5549132947976879, 'support': 195}, '3': {'precision': 0.7142857142857143, 'recall': 0.09803921568627451, 'f1-score': 0.1724137931034483, 'support': 51}, '4': {'precision': 0.7894736842105263, 'recall': 0.3103448275862069, 'f1-score': 0.44554455445544555, 'support': 145}, '5': {'precision': 1.0, 'recall': 0.06153846153846154, 'f1-score': 0.11594202898550725, 'support': 65}, '6': {'precision': 1.0, 'recall': 0.14634146341463414, 'f1-score': 0.2553191489361702, 'support': 41}, '7': {'precision': 0.6865671641791045, 'recall': 0.8422535211267606, 'f1-score': 0.7564832384566731, 'support': 710}, '8': {'precision': 0.9313725490196079, 'recall': 0.6462585034013606, 'f1-score': 0.7630522088353414, 'support': 147}, '9': {'precision': 0.6666666666666666, 'recall': 0.0425531914893617, 'f1-score': 0.08, 'support': 47}, '10': {'precision': 1.0, 'recall': 0.05555555555555555, 'f1-score': 0.10526315789473684, 'support': 54}, '11': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 20}, '12': {'precision': 1.0, 'recall': 0.04878048780487805, 'f1-score': 0.09302325581395349, 'support': 41}, '13': {'precision': 0.6706443914081146, 'recall': 0.9093851132686084, 'f1-score': 0.771978021978022, 'support': 618}, '14': {'precision': 0.71875, 'recall': 0.8013937282229965, 'f1-score': 0.7578253706754531, 'support': 287}, '15': {'precision': 0.9090909090909091, 'recall': 0.19607843137254902, 'f1-score': 0.3225806451612903, 'support': 51}, '16': {'precision': 0.5562579013906448, 'recall': 0.8721506442021804, 'f1-score': 0.6792744114241605, 'support': 1009}, '17': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 46}, '18': {'precision': 0.5853018372703412, 'recall': 0.6737160120845922, 'f1-score': 0.6264044943820225, 'support': 331}, '19': {'precision': 0.7642857142857142, 'recall': 0.7133333333333334, 'f1-score': 0.7379310344827587, 'support': 150}, '20': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 23}, '21': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 43}, '22': {'precision': 0.6666666666666666, 'recall': 0.046511627906976744, 'f1-score': 0.08695652173913045, 'support': 43}, '23': {'precision': 0.9090909090909091, 'recall': 0.26548672566371684, 'f1-score': 0.4109589041095891, 'support': 113}, '24': {'precision': 0.9013157894736842, 'recall': 0.8303030303030303, 'f1-score': 0.8643533123028391, 'support': 165}, '25': {'precision': 0.8769230769230769, 'recall': 0.504424778761062, 'f1-score': 0.6404494382022471, 'support': 113}, '26': {'precision': 0.8918918918918919, 'recall': 0.3055555555555556, 'f1-score': 0.4551724137931035, 'support': 108}, '27': {'precision': 0.9024390243902439, 'recall': 0.43529411764705883, 'f1-score': 0.5873015873015872, 'support': 85}, 'accuracy': 0.6582696675614289, 'macro avg': {'precision': 0.6600952797353202, 'recall': 0.33964269281299353, 'f1-score': 0.37891683746700233, 'support': 4843}, 'weighted avg': {'precision': 0.6814636295364148, 'recall': 0.6582696675614289, 'f1-score': 0.614852474001098, 'support': 4843}} 



# ### Multi-task self-supervision - NSP

# In[ ]:


### create df for nsp
# text batches of size n, columns with next sent and random sent

# function to convert long iterable (text column/series/list) into concatenated groups of n elements https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
def group_iterable(iterable, n):
    return zip(*[iter(iterable)]*n)

# create df with n texts batched together in one cell
TEXT_BATCH_SIZE = 2
text_batch_lst = []
text_batch_doc_id_lst = []
for group_name, group_df in df_cl.groupby(by="doc_id"):
  for group in group_iterable(group_df.text_original[5:-5], TEXT_BATCH_SIZE):  # omit the first and last 5 texts because they presumable contain less meaningful content. ("thank you ...")
    text_batch_lst.append(" ".join(group))
    text_batch_doc_id_lst.append(group_name)

df_text_batches = pd.DataFrame(data={"text_batch": text_batch_lst, "doc_id": text_batch_doc_id_lst})

# create df with text batches, next sententeces (text batches) and random sentences for each text batch
sent_next_lst = []
sent_random_lst = []
for i in range(len(df_text_batches)-1):  # -1 to avoid last text which does not have a next sent
  sent_next_lst.append(df_text_batches.iloc[i+1]["text_batch"])
  sent_random_lst.append(df_text_batches[df_text_batches.doc_id != df_text_batches.iloc[i].doc_id].sample(n=1)["text_batch"].iloc[0])

df_nsp = pd.DataFrame(data={"text_batch": text_batch_lst[:-1], "sent_next": sent_next_lst, "sent_random": sent_random_lst, "doc_id": text_batch_doc_id_lst[:-1]})  # [:-1] because had to cut last text, has no next sent


# In[ ]:


## convert df_nsp to training format

def sample_df_nsp_train(df_nsp=None, n_sample=None):
  df_nsp_train_step1 = df_nsp.sample(n=n_sample, random_state=42)
  df_nsp_train_step2 = df_nsp.sample(n=n_sample, random_state=69)
  df_nsp_train = pd.DataFrame(data={"text": pd.concat([df_nsp_train_step1["text_batch"], df_nsp_train_step2["text_batch"]]),
                                    "hypothesis": pd.concat([df_nsp_train_step1["sent_next"], df_nsp_train_step2["sent_random"]]),
                                    "label": [0] * n_sample + [1] * n_sample})
  return df_nsp_train.sample(frac=1)


#df_nsp_train = sample_df_nsp_train(df_nsp=df_nsp, n_sample=5)
#DataTable(df_nsp_train, num_rows_per_page=5)


# ## Balanced dataloading?

# In[ ]:


#### test different dataloaders
## probably need to subclass and overwrite get_train_dataloader https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L625
# similar to how compute_loss is overwritten here https://huggingface.co/docs/transformers/main_classes/trainer#trainer

# standard dataloader
#train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# balanced dataloader
def make_weights_for_balanced_classes(train_examples, nclasses):                        
    count = [0] * nclasses                                                      
    for train_example in train_examples:                                                         
        count[train_example.label] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(train_examples)                                              
    for idx, val in enumerate(train_examples):                                          
        weight[idx] = weight_per_class[val.label]                                  
    return weight, weight_per_class

# each text gets a weight and each class gets a weight
weights, weight_per_class = make_weights_for_balanced_classes(train_examples, len(df_train["label_subcat"].unique())) 

# inspect weights
df_weights = pd.DataFrame(data={"subcat_freq": df_train["label_subcat"].value_counts(sort=False), "subcat_weight": weight_per_class})
df_weights["subcat_weight_norm"] = df_weights.subcat_weight/df_weights.subcat_weight.sum()

# using weight per text here
weights = torch.DoubleTensor(weights)
# docs for pytorch samplers https://pytorch.org/docs/stable/data.html                                 
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
train_dataloader = torch.utils.data.DataLoader(train_examples, batch_size=64, shuffle=False,                              
                                               sampler=sampler)  # num_workers=args.workers, pin_memory=True


# ## Other

# In[ ]:





# In[ ]:


####  (unclean)
## shorter hypotheses 
hypo_label_dic_short = {
    'Agriculture': "It is about agriculture.",
    'Culture': "It is about cultural policy",
    'Civil Rights': "It is about civil rights, or minorities, or civil liberties.",
    'Defense': "It is about defense, or military",
    'Domestic Commerce': "It is about banking, or finance, or commerce.",
    'Education': "It is about education.",
    'Energy': "It is about energy, or electricity, or fossil fules.",
    'Environment': "It is about the environment, or water, or waste, or pollution.",
    'Foreign Trade': "It is about foreign trade.",
    'Government Operations': "It is about government operations, or administration.",
    'Health': "It is about health.",
    'Housing': "It is about community development, or housing issues.",
    'Immigration': "It is about migration.",
    'International Affairs': "It is about international affairs, or foreign aid.", 
    'Labor': "It is about employment, or labour.",
    'Law and Crime': "It is about law, crime, and family issues.",
    'Macroeconomics': "It is about macroeconomics.",
    'Other': "It is about other topics, or miscellaneous",  
    'Public Lands': "It is about public lands, or water management.",
    'Social Welfare': "It is about social welfare.",
    'Technology': "It is about space, or science, or technology, or communications.",
    'Transportation': "It is about transportation", 
}
## detailed hypotheses - seem to drastically reduce performance - probably because bill text one-sidedly trigger 1-2 hypos
hypo_label_dic_long = {
    'Agriculture': "It is about agriculture, agricultural trade, subsidies to farmers, food safety, agricultural marketing, animals, fisheries",
    'Culture': "It is about cultural policy",
    'Civil Rights': "It is about civil rights, minority discrimination, gender discrimination, age and handicap discrimination, voting rights, freedom of speech, privacy, anti-government",
    'Defense': "It is about defense policy, military, intelligence, alliances, nuclear weapons, military aid, military personnel, military procurement and contractors, military hazardous waste, homeland security, foreign operations, war",
    'Domestic Commerce': "It is about domestic commerce, banking, securities and commodities, consumer finance and safety, insurance regulation, brankruptcy, small businesses, copyright and patents, disaster relief, tourism, sports regulation",
    'Education': "It is about education, universities, schools, underprivileged students, vocational training, education for physically mentally handicapped",
    'Energy': "It is about energy policy, nuclear energy, electricity, natural gas and oil, coal, renewable energy",
    'Environment': "It is about the environment, drinking water, waste disposal, hazardous waste, pollution, climate change, species and forest protection, coastal water, land conservation",
    'Foreign Trade': "It is about foreign trade, trade agreements, exports, private international investments, competitiveness, tariffs and imports, exchange rates",
    'Government Operations': "It is about government operations, government agencies, government efficiency and bureaucratic oversight, postal service, government employees, appointments and nominations, currency, government procurement and contractors, tax administration, government scandals, checks and balances, political campaigns, census, capital city, national holidays",
    'Health': "It is about health, health care, health insurance, drug industry, medical facilities, health labor, disease prevention, tobacco, alcohol and drug abuse",
    'Housing': "It is about housing, community development, urban development, rural housing and development, housing assistance, veterans, elderly and homeless housing support",
    'Immigration': "It is about immigration, migrants, refugees, citizenship",
    'International Affairs': "It is about international affairs, foreign aid, international resources exploitation, developing countries, international finance and debt, European Union, human rights, international organisations, international terrorism, diplomacy", 
    'Labor': "It is about labor, employment policy, worker safety, employment training, employee benefits, labor unions, fair labor standards, migrant and seasonal workers",
    'Law and Crime': "It is about law and crime, white collar crime, illegal drugs, trafficking, courts, prisons, child abuse, family issues, criminal and civil code, police, domestic response to terrorism",
    'Macroeconomics': "It is about macroeconomics, interest rates, unemployment, monetary policy, national budget and debt, taxation, industrial policy, price control",
    'Other': "It is about other, miscellaneous topics",  
    'Public Lands': "It is about public lands and territorial issues, water management, national parks, historic sights, indigenous affairs, natural resources, forests, dependencies and territories",
    'Social Welfare': "It is about social welfare policy, low-income assistance, elderly and disabled assistance, volunteer associations and charities, child care",
    'Technology': "It is about technology, space, science, telecommunications and broadcast, weather forecasting, computers and internet, cyber security",
    'Transportation': "It is about transportation policy, highways and cars, air travel and planes, railroads and trains, safety, maritime, infrastructure", 
}"""


# In[ ]:


#### old evaluation loop without cross-validation

import warnings
transformers.logging.set_verbosity_warning()  # https://huggingface.co/transformers/main_classes/logging.html
warnings.filterwarnings(action='ignore')

# ! for NLI is probably best to not run 4 epochs, but only 3 or less
METHOD = "nli"  # "standard_dl", "nli", "nsp"
N_MAX_SAMPLE_PER_CLASS = [200]  #[0, 16, 64, 128]  # 2_000_000, 200, 20, 0
HYPER_PARAMS = 'few_shot'  # 'few_shot', 'full_shot'
VAL_OR_TEST_SET = 'test'  # 'test', 'validation'
TRAINING_DIRECTORY = "nli-few-shot/cap/sotu"
LABEL_TEXT_ALPHABETICAL = np.sort(df_cl.label_text.unique())
HYPOTHESIS_TYPE = hypo_label_dic_short  # hypo_label_dic_short , hypo_label_dic_long, hypo_label_dic_short_subcat
REVERSE = False
N_NSP_SAMPLES = None  # 100 * len(df_cl.label_text.unique())  # None
#LANGUAGES_AUGMENT = ["de", "es", "zh"]  
#N_TEXTS_PER_CLASS_AUGMENT = 80
#translator = EasyNMT('opus-mt')
#TRUE_LABEL_POSITION = 0  # will be necessary for some third party NLI models were entailment is not label 0

MODEL_NAME = "./results/nli-few-shot/MiniLM-L6-mnli-binary/" # "textattack/bert-base-uncased-MNLI"  # "./results/nli-few-shot/MiniLM-L6-mnli-binary/",   # './results/nli-few-shot/all-nli-2c/MiniLM-L6-allnli-2c-v1'
#MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"    # "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",    # "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
#MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
#MODEL_NAME = "bert-base-uncased"  # "google/fnet-base" ,  "bert-base-uncased" , "bert-large-uncased" , "google/mobilebert-uncased"

def few_shooter(n_max_sample_per_class=None):
  print(f"The method is:  {METHOD} with {MODEL_NAME};  {HYPER_PARAMS} hyper-paramters;  testing on {VAL_OR_TEST_SET}-set; with up to {n_max_sample_per_class} samples per class.\n")
  df_train_samp, df_train, df_test = custom_train_test_split_sent_overlapp(df=df_cl, val_or_test_set=VAL_OR_TEST_SET, n_max_sample_per_class=n_max_sample_per_class, dataset="sotu")
  len_df_train_samp = len(df_train_samp)
  df_train_samp = format_nli_trainset(df_train_samp=df_train_samp, hypo_label_dic=HYPOTHESIS_TYPE, method=METHOD)  # hypo_label_dic_short , hypo_label_dic_long
  
  ## augmentation & domain adaptation via NSP
  #if N_NSP_SAMPLES != None:
  #  df_nsp_train = sample_df_nsp_train(df_nsp=df_nsp, n_sample=N_NSP_SAMPLES)
  #  df_train_samp = pd.concat([df_train_samp[["text", "hypothesis", "label"]], df_nsp_train[["text", "hypothesis", "label"]]]).sample(frac=1)
  ## adding augmentation via back-translation
  #df_train_samp = aug_back_translation(df=df_train_samp, languages=LANGUAGES_AUGMENT, n_texts_per_class_agument=N_TEXTS_PER_CLASS_AUGMENT)
  df_test = format_nli_testset(df_test=df_test, hypo_label_dic=HYPOTHESIS_TYPE, method=METHOD)  # hypo_label_dic_short , hypo_label_dic_long

  clean_memory()
  model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, df_train_samp=df_train_samp, method=METHOD)
  encoded_dataset = tokenize_datasets(df_train_samp=df_train_samp, df_test=df_test, tokenizer=tokenizer, method=METHOD, reverse=REVERSE)

  train_args = set_hyperparams(params_dic=HYPER_PARAMS, training_directory=TRAINING_DIRECTORY) 
  trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=encoded_dataset, train_args=train_args, method=METHOD)
  clean_memory()

  if n_max_sample_per_class != 0:
    trainer.train()
  results = trainer.evaluate()  # eval_dataset=encoded_dataset_test
  # dataset statistics
  dataset_stats_dic = {"share_traing_samp_to_full": round((len_df_train_samp / len(df_train)) * 100, 2), "n_training_samp": len_df_train_samp, "n_train_full": len(df_train)}

  run_parameters_dic = {"method": METHOD, "n_sample_per_class": n_max_sample_per_class, "model": MODEL_NAME, "metrics": results, "hyper_params": HYPER_PARAMS, "dataset_stats": dataset_stats_dic, "test_set": VAL_OR_TEST_SET, "hypotheses": HYPOTHESIS_TYPE}  # "trainer_args": train_args
  transformers.logging.set_verbosity_warning()  # https://huggingface.co/transformers/main_classes/logging.html

  print(results)
  return run_parameters_dic

run_parameters_dic = {}
for i, n_sample in enumerate(N_MAX_SAMPLE_PER_CLASS):
  transformers.logging.set_verbosity_warning()  # https://huggingface.co/transformers/main_classes/logging.html
  run_parameters_dic.update({"n_run": i, "run_details": few_shooter(n_max_sample_per_class=n_sample)})

print(run_parameters_dic)
warnings.filterwarnings(action='default')


# In[ ]:



label_text_map_cap4_nyt = {
    100: "General Domestic Macroeconomic Issues (includes combinations of multiple subtopics)",
    101: "Inflation, Prices, and Interest Rates",
    103: "Unemployment Rate",
    104: "Monetary Supply, Federal Reserve Board, and the Treasury",
    105: "National Budget and Debt",
    107: "Taxation, Tax policy, and Tax Reform",
    108: "Industrial Policy",
    110: "Price Control and Stabilization",
    199: "Other",
    200: "General (includes combinations of multiple subtopics)",
    201: "Ethnic Minority and Racial Group Discrimination",
    202: "Gender and Sexual Orientation Discrimination",
    204: "Age Discrimination",
    205: "Handicap or Disease Discrimination", 
    # ...
}

label_text_map_cap4 = {
        100: ["General", "Includes issues related to general domestic macroeconomic policy"],
        101: ["Interest Rates", "Includes issues related to inflation, cost of living, prices, and interest rates"],
        103: ["Unemployment Rate", "Includes issues related to the unemployment rate, impact of unemployment"],
        104: ["Monetary Policy", "Includes issues related to the monetary policy, central bank, and the treasury"],
        105: ["National Budget","Issues related to public debt, budgeting, and efforts to reduce deficits"],
        107: ["Tax Code", "Includes issues related to tax policy, the impact of taxes, and tax enforcement"],
        108: ["Industrial Policy", "Includes issues related to manufacturing policy, industrial revitalization and growth"],
        110: ["Price Control", "Includes issues related to wage or price control, emergency price controls"],
        199: ["Other", "Includes issues related to other macroeconomics subtopics"],

        200: ["General", "Includes issues related generally to civil rights and minority rights"],
        201: ["Minority Discrimination", "Includes issues related to minority, ethnic, and racial group discrimination"], 
        202: ["Gender Discrimination", "Includes issues related to sex, gender, and sexual orientation discrimination"],
        204: ["Age Discrimination", "Includes issues related to age discrimination, including mandatory retirement age policies"],
        205: ["Handicap Discrimination", "Includes issues related to handcap and disease discrimination"],
        206: ["Voting Rights", "Includes issues related to voting rights, expanding or contracting the franchise, participation and related issues"],
        207: ["Freedom of Speech", "Issues related to freedom of speech, religious freedoms, and other types of freedom of expression"],
        208: ["Right to Privacy", "Includes issues related to privacy rights, including privacy of records, access to government information, and abortion rights"],
        209: ["Anti-Government", "Includes issues related to anti-government activity groups, such as the communist party and local insurgency groups"],
        299: ["Other", "Includes issues related to other civil rights subtopics"],

        300: ["General", "Includes issues related generally to health care, including appropriations for general health care government agencies"],
        301: ["Health Care Reform", "Includes issues related to broad, comprehensive changes in the health care system"],
        302: ["Insurance", "Includes issues related to health insurance reform, regulation, availability, and cost"],
        321: ["Drug Industry", "Includes issues related to the regulation and promotion of pharaceuticals, medical devices, and clinical labs"],
        322: ["Medical Facilities", "Issues related to facilities construction, regulaton and payments, including waitlists and ambulance services"],
        323: ["Insurance Providers", "Includes issues related to provider and insurer payments and regulation, including other types of benefits or multple benefits"],
        324: ["Medical Liability", "Includes issues related to medical liability, malpractice issues, medical fraud and abuse, and unfair practices"],
        325: ["Manpower", "Issues related to the supply and quantity of labor in the health care industry, training and licensing"],
        331: ["Disease Prevention", "Issues related to disease prevention, treatment, and health promotion, including specific diseases not covered in other subtopics"],
        332: ["Infants and Children", "Includes issues related to infants and children, including coverage and quality of care, health promotion, and school health programs"],
        333: ["Mental", "Includes issues related to mental health care and mental health disease"],
        334: ["Long-term Care", "Includes issues related to long term care, home health care, the terminally ill, and rehabilitation services"],
        335: ["Drug Coverage and Cost", "Includes issues related to prescription drug coverage, programs to pay for prescription drugs, and policy to reduce the cost of prescription drugs"],
        341: ["Tobacco Abuse", "Includes issues related to tobacco abuse, treatment, education, and health effects"],
        342: ["Drug and Alcohol Abuse", "Includes issues related to alcohol and illegal drug abuse, treatment, education, and health effects"],
        398: ["R&D", "Includes issues related to health care research and development"],
        399: ["Other", "Includes issues related to other health care topics"],

        400: General
              Description: Includes issues related to general agriculture policy, including appropriations for general agriculture government agencies
        401: Trade
              Description: Includes issues related to  the regulation and impact of agricultural foreign trade
        402: Subsidies to Farmers
              Description: Includes issues related to government subsidies to farmers and ranchers, including agricultural disaster insurance
        403: Food Inspection & Safety
              Description: Includes issues related to food inspection and safety, including seafood, and labeling requirements
        404: Marketing & Promotion
              Description: Includes issues related to efforts to provide information on agricultural products to consumers and the regulation of agricultural marketing
        405: Animal and Crop Disease
              Description: Includes issues related to animal and crop disease, pest control and pesticide regulation, and welfare for domesticated animals
        408: Fisheries & Fishing
              Description: Includes issues related to fishing, commercial fishery regulation and conservation
        498: R&D
              Description: Includes issues related to agricultural research and development
        499: Other
              Description: Includes issues related to other agricultural subtopics
    5. Labor
        500: General
              Description: Includes issues generally related to labor, employment, and pensions, including appropriations for government agencies regulating labor policy
        501: Worker Safety
              Description: Includes issues related to worker safety and protection and compensation for work-related injury and disease
        502: Employment Training
              Description: Includes issues related to job training for adult workers, workforce development, and efforts to retrain displaced workers
        503: Employee Benefits
              Description: Includes issues related to all employee benefits, pensions, and retirement accounts, including government-provided unemployment insurance
        504: Labor Unions
              Description: Includes issues related to labor unions, collective bargaining, and employer-employee relations
        505: Fair Labor Standards
              Description: Includes issues related to fair labor standards such as the minimum wage and overtime compensation, and labor law
        506: Youth Employment
              Description: Includes issues related to youth employment, child labor and job training for youths
        529: Migrant and Seasonal
              Description: Includes issues related to migrant, guest and seasonal workers
        599: Other
              Description: Issues related to other labor policy
    6. Education
        600: General
              Description: Includes issues related to general education policy, including appropriations for government agencies regulating education policy
        601: Higher
              Description: Includes issues related to higher education, student loans and education finance, and the regulation of colleges and universities
        602: Elementary & Secondary
              Description: Includes issues related to elementary and primary schools, school reform, safety in schools, and efforts to generally improve educational standards and outcomes
        603: Underprivileged
              Description: Includes issues related to education of underprivileged students, including adult literacy programs, bilingual education needs, and rural education initiatives
        604: Vocational
              Description: Includes issues related to vocational education for children and adults and their impact
        606: Special
              Description: Includes issues related to special education and education for the physically or mentally handicapped
        607: Excellence
              Description: Includes issues related to education excellence, including efforts to increase the quality of specific areas, such as math, science or foreign language skills
        698: R&D
              Description: Includes issues related to research and development in education
        699: Other
              Description: Includes issues related to other subtopics in education policy
    7. Environment
        700: General
              Description: Includes issues related to general environmental policy, including appropriations for government agencies regulating environmental policy
        701: Drinking Water
              Description: Includes issues related to domestic drinking water safety, supply, polution, fluridation, and conservation
        703: Waste Disposal
              Description: Includes issues related to the disposal and treatment of wastewater, solid waste and runoff. 
        704: Hazardous Waste
              Description: Includes issues related to hazardous waste and toxic chemical regulation, treatment, and disposal
        705: Air Pollution
              Description: Includes issues related to air pollution, climate change, and noise pollution
        707: Recycling
              Description: Includes issues related to recycling, reuse, and resource conservation
        708: Indoor Hazards
              Description: Includes issues related to indoor environmental hazards, indoor air contamination (including on airlines), and indoor hazardous substances such as asbestos
        709: Species & Forest
              Description: Includes issues related to species and forest protection, endangered species, control of the domestic illicit trade in wildlife products, and regulation of labratory or performance animals
        711: Conservation
              Description: Includes issues related to land and water conservation
        798: R&D
              Description: Includes issues related to research and development in environmental technology, not including alternative energy
        799: Other
              Description: Includes issues related to other environmental subtopics
    8. Energy
        800: General
              Description: Includes issues generally related to energy policy, including appropriations for government agencies regulating energy policy
        801: Nuclear
              Description: Includes issues related to nuclear energy, safety and security, and disposal of nuclear waste
        802: Electricity
              Description: Includes issues related to to general electricity, hydropower, and regulation of electrical utilities 
        803: Natural Gas & Oil
              Description: Includes issues related to natural gas and oil, drilling, oil spills and flaring, oil and gas prices, shortages and gasoline regulation
        805: Coal
              Description: Includes issues related to coal production, use, trade, and regulation, including coal gasification and clean coal technologies
        806: Alternative & Renewable
              Description: Includes issues related to alternative and renewable energy, biofuels, hydrogen and geothermal power
        807: Conservation
              Description: Includes issues related to energy conservation and energy efficiency, including vehicles, homes, commerical use and government
        898: R&D
              Description: Includes issues related to energy research and development
        899: Other
              Description: Includes issues related to other energy subtopics
    9. Immigration
        900: Immigration
              Description: Includes issues related to immigration, refugees, and citizenship
    10. Transportation
        1000: General
              Description: Includes issues related generally to transportation, including appropriations for government agencies regulating transportation policy
        1001: Mass
              Description: Includes issues related to mass transportation construction, regulation, safety, and availability
        1002: Highways
              Description: Includes issues related to public highway construction, maintenance, and safety
        1003: Air Travel
              Description: Includes issues related to air travel, regulation and safety of aviation, airports, air traffic control, pilot training, and aviation technology
        1005: Railroad Travel
              Description: Includes issues related to railroads, rail travel, rail freight, and the development and deployment of new rail technologies
        1007: Maritime
              Description: Includes issues related to maritime transportation, including martime freight and shipping, safety and security, and inland waterways and channels
        1010: Infrastructure
              Description: Includes issues related to infrastructure and public works, including employment intiatives
        1098: R&D
              Description: Includes issues related to transportation research and development
        1099: Other
              Description: Includes issues related to other transportation subtopics
    12. Law and Crime
        1200: General
              Description: Includes issues related to general law, crime, and family issues
        1201: Agencies
              Description: Includes issues related to all law enforcement agencies, including border, customs, and other specialized enforcement agencies and their appropriations
        1202: White Collar Crime
              Description: Includes issues related to white collar crime, organized crime, counterfeitting and fraud, cyber-crime, and money laundering
        1203: Illegal Drugs
              Descripton: Issues related to illegal drug crime and enforcement, criminal penalties for drug crimes, including international efforts to combat drug trafficking 
        1204: Court Administration
              Description: Includes issues related to court administration, judiciary appropriations, guidelines for bail, pre-release, fines and legal representation
        1205: Prisons
              Description: Includes issues related to prisons and jails, parole systems, and appropriations
        1206: Juvenile Crime
              Description: Includes issues related to juvenile crime and justice, juvenile prisons and jails, and efforts to reduce juvenile crime and recidivism
        1207: Child Abuse
              Description: Includes issues related to child abuse, child pornography, sexual exploitation of children and parental kidnapping
        1208: Family Issues
              Description: Includes issues related to family issues, domestic violence, child welfare, family law
        1210: Criminal & Civil Code
              Description: Includes issues related to domestic criminal and civil codes, including crimes not mentioned in other subtopics
        1211: Crime Control
              Description: Includes issues related to the control, prevention, and impact of crime
        1227: Police
              Description: Includes issues related to Police and other general domestic security responses to terrorism, such as special police
        1299: Other
              Description: Includes issues related to other law, crime, and family subtopics
    13. Social Welfare
        1300: General
              Description: Includes issues generally related to social welfare policy
        1302: Low-Income Assistance
              Description: Includes issues related to poverty assitance for low-income families, including food assitance programs, programs to assess or alleviate welfare dependency and tax credits directed at low income families
        1303: Elderly Assistance
              Description: Includes issues related to elderly issues and elderly assitance, including government pensions
        1304: Disabled Assistance
              Description: Includes issues related to aid for people with physical or mental disabilities
        1305: Volunteer Associations
              Description: Includes issues related to domestic volunteer associations, charities, and youth organizations
        1308: Child Care
              Description: Includes issues related to parental leave and child care
        1399: Other
              Description: Includes issues related to other social welfare policy subtopics
    14. Housing
        1400: General
              Description: Includes issues related generally to housing and urban affairs
        1401: Community Development
              Description: Includes issues related to housing and community development, neighborhood development, and national housing policy
        1403: Urban Development
             Description: Includes issues related to urban development and general urban issues
        1404: Rural Housing
              Description: Includes issues related to rural housing
        1405: Rural Development
              Description: Includes issues related to non-housing rural economic development
        1406: Low-Income Assistance
              Description: Includes issues related to housing for low-income individuals and families, including public housing projects and housing affordability programs
        1407: Veterans
              Description: Includes issues related to housing for military veterans and their families, including subsidies for veterans
        1408: Elderly
              Description: Includes issues related to housing for the elderly, including housing facilities for the handicapped elderly
        1409: Homeless
              Description: Includes issues related to housing for the homeless and efforts to reduce homelessness 
        1498: R&D
              Description: Includes issues related to housing and community development research and development
        1499: Other
              Description: Other issues related to housing and community development
    15. Domestic Commerce
        1500: General
              Description: Includes issues generally related to domestic commerce, including approprations for government agencies regulating domstic commerce
        1501: Banking
              Description: Includes issues related to the regulation of national banking systems and other non-bank financial institutions
        1502: Securities & Commodities
              Description: Includes issues related to the regulation and facilitation of securities and commodities trading, regulation of investments and related industries, and exchanges
        1504: Consumer Finance
              Description: Includes issues related to consumer finance, mortages, credit cards, access to credit records, and consumer credit fraud
        1505: Insurance Regulation
              Description: Includes issues related to insurance regulation, fraud and abuse in the insurance industry, the financial health of the insurance industry, and insurance availability and cost
        1507: Bankruptcy
              Description: Includes issues related to personal, commercial, and municipal bankruptcies
        1520: Corporate Management
              Description: Includes issues related to corporate mergers, antitrust regulation, corporate accounting and governance, and corporate  management
        1521: Small Businesses
              Description: Includes issues related to small businesses, including programs to promote and subsidize small businesses
        1522: Copyrights and Patents
              Description: Includes issues related to copyrights and patents, patent reform, and intellectual property
        1523: Disaster Relief
              Description: Includes issues related to domestic natual disaster relief, disaster or flood insurance, and natural disaster preparedness
        1524: Tourism
              Decription: Issues related to tourism regulation, promotion, and impact
        1525: Consumer Safety
              Description: Includes issues related to consumer fraud and safety in domestic commerce 
        1526: Sports Regulation
              Description: Includes issues related to the regulation and promotion of sports, gambling, and personal fitness
        1598: R&D
              Description: Includes issues related to domestic commerce research and development
        1599: Other
              Description: Includes issues related to other domestic commerce policy subtopics
    16. Defense
        1600: General
              Description: Includes issues related generally to defense policy, and appropriations for agencies that oversee general defense policy
        1602: Alliances
              Description: Includes issues related to defense alliance and agreement, security assistance, and UN peacekeeping activities
        1603: Intelligence
              Description: Includes issues related to military intelligence, espionage, and covert operations
        1604: Readiness
              Description: Includes issues related to military readiness, coordination of armed services air support and sealift capabilities, and national stockpiles of strategic materials.
        1605: Nuclear Arms
              Description: Includes issues related to nuclear weapons, nuclear proliferation, moderization of nuclear equipment
        1606: Military Aid
              Description: Includes issues related to military aid to other countries and the control of arms sales to other countries
        1608: Personnel Issues
              Description: Includes issues related to military manpower, military personel and their defendents, military courts, and general veterans issues
        1610: Procurement
              Description: Includes issues related to military procurement, conversion of old equipment, and weapons systems evaluation
        1611: Installations & Land
              Description: Includes issues related to military installations, construction, and land transfers
        1612: Reserve Forces
              Description; Issues related to military reserves and reserve affairs
        1614: Hazardous Waste
              Description: Includes issues related to military nuclear and hazardous waste disposal and military environmental compliance
        1615: Civil
              Description: Includes issues related to domestic civil defense, national security responses to terrorism, and other issues related to homeland security
        1616: Civilian Personnel
              Description: Includes issues related to non-contractor civilian personell, civilian employment in the defense industry, and military base closings
        1617: Contractors
              Description: Includes issues related to military contractors and contracting, oversight of military contrators and fraud by military contractors
        1619: Foreign Operations
              Description: Includes issues related to direct war-related foreign military operations, prisoners of war and collateral damage to civilian populations
        1620: Claims against Military
              Description: Includes issues related to claims against the military, settlements for military dependents, and compensation for civilizans injured in military operations
        1698: R&D
              Description: Includes issues related to defense research and development
        1699: Other
              Description: Includes issues related to other defense policy subtopics
    17. Technology
        1700: General
              Description: Includes issues related to general space, science, technology, and communications
        1701: Space
              Description: Includes issues related to the government use of space and space resource exploitation agreements, government space programs and space exploration, military use of space
        1704: Commercial Use of Space
              Description: Includes issues related to the regulation and promotion of commerical use of space, commercial satellite technology, and government efforts to encourage commercial space development
        1705: Science Transfer
              Description: Includes issues related to science and technology transfer and international science cooperation
        1706: Telecommunications
              Description: Includes issues related to telephone and telecommunication regulation, infrastructure for high speed internet, and other forms fo telecommunication
        1707: Broadcast
              Description: Includes issues related to the regulation of the newspaper, publishing, radio, and broadcast televsion industries
        1708: Weather Forecasting
              Description: Includes issues related to weather forecasting, oceanography, geological surveys, and weather forecasting research and technology
        1709: Computers
              Description: Includes issues related generally to the computer industry, regulation of the internet, and computer security
        1798: R&D
              Description: Includes issues related to space, science, technology, and communication research and development not mentioned in other subtopics.
        1799: Other
              Description: Includes issues related to other space, science, technology, and communication research and development
    18. Foreign Trade
        1800: General
              Description: Includes issues generally related to foreign trade and appropriations for government agencies generally regulating foreign trade
        1802: Trade Agreements
              Description: Includes issues related to trade negotiatons, disputes, and agreements, including tax treaties
        1803: Exports
              Description: Includes issues related to export regulation, subsidies, promotion, and control
        1804: Private Investments
              Description: Includes issues related to international private business investment and corporate development
        1806: Competitiveness
              Description: Includes issues related to productivity of competitiveness of domestic businsses and balance of payments issues
        1807: Tariff & Imports
              Description: Includes issues related to tariffs and other barriers to imports, import regulation and impact of imports on domestic industries
        1808: Exchange Rates
              Description: Includes issues related to exchange rate and related issues
        1899: Other
              Description: Includes issues related to other foreign trade policy subtopics
    19. International Affairs
        1900: General
              Description: Includes issues related to general international affairs and foreign aid, including appropriations for general government foreign affairs agencies
        1901: Foreign Aid
              Description: Includes issues related to foreign aid not directly targeting at increasing international development
        1902: Resources Exploitation
              Description: Includes issues related to international resources exploitation and resources agreements, law of the sea and international ocean conservation efforts
        1905: Developing Countries
              Description: Includes issues related specifically to developing countriesDeveloping Countries Issues (for Financial Issues see 1906)
        1906: International Finance
              Description: Includes issues related to international finance and economic development, the World Bank and International Monetary Fund, regional development banks, sovereign debt and implications for international lending instututions
        1910: Western Europe
              Description: Includes issues related to Western Europe and the European Union
        1921: Specific Country
              Description: Includes issues related specifically to a foreign country or region not codable using other codes, assessment of political issues in other countries, relations between individual countries
        1925: Human Rights
              Description: Includes issues related to human rights, human rights violations, human rights treaties and conventions, UN reports on human rights, crimes associated with genocide or crimes against humanitity
        1926: Organizations
              Description: International organizations, NGOs, the United Nations, International Red Cross, UNESCO, International Olympic Committee, International Criminal Court
        1927: Terrorism
              Description: Includes issues related to international terrorism, hijacking, and acts of piracy in other countries, efforts to fight international terrorism, international legal mechanisms to combat terrorism
        1929: Diplomats
              Description: Includes issues related to diplomats, diplomacy, embassies, citizens abroad, foreign diplomats in the country, visas and passports
        1999: Other
              Description: Includes issues related to other international affairs policy subtopics
    20. Government Operations
        2000: General
              Description: Includes issues related to general government operations, including appropriations for multiple government agencies
        2001: Intergovernmental Relations
              Description: Includes issues related to intergovernmental relations, local government issues
        2002: Bureaucracy
              Description: Includes issues related to general government efficiencies and bureaucratic oversight
        2003: Postal Service
              Description: Includes issues related to postal services, regulation of mail, and post civil service
        2004: Employees
              Description: Includes issues related to civil employees not mentioned in other subtopics, government pensions and general civil service issues
        2005: Appointments
              Description: Includes issues related to nomations and appointments not mentioned elsewhere
        2006: Currency
              Description: Includes issues related the currency, natonal mints, medals, and commemorative coins
        2007: Procurement & Contractors
              Description: Includes issues related to government procurement, government contractors, contractor and procurement fraud, and procurement processes and systems
        2008: Property Management
              Description: Includes issues related to government property management, construction, and regulation
        2009: Tax Administration
              Description: Includes issues related to tax administration, enforcement, and auditing for both individuals and corporations
        2010: Scandals
              Description: Includes issues related to public scandal and impeachment
        2011: Branch Relations
              Description: Includes issues related to government branch relatons, administrative issues, and constitutional reforms
        2012: Political Campaigns
              Description: Includes issues related to the regulation of political campaigns, campaign finance, political advertising and voter registration
        2013: Census & Statistics
              Description: Includes issues related to census and statistics collection by government
        2014: Capital City
              Description: Includes issues related to the capital city
        2015: Claims against
              Description: Includes issues related to claims againt the government, compensation for the victims of terrorist attacks, compensation policies without other substantive provisions
        2030: National Holidays
              Description: Includes issues related to national holidays and their observation
        2099: Other
              Description: Includes issues related to other government operations subtopics
    21. Public Lands
        2100: General
              Description: Includes issues related to general public lands, water management, and terrorial issues
        2101: National Parks
              Description: Includes issues related to national parks, memorials, historic sites, and recreation, including the management and staffing of cultural sites
        2102: Indigenous Affairs
              Description: Includes issues related to indigenous affairs, indigenous lands, and assistance to indigenous people
        2103: Public Lands
              Description: Includes issues related to natural resources, public lands, and forest management, including forest fires, livestock grazing
        2104: Water Resources
              Description: Includes issues related to water resources, water resource development and civil works, flood control, and research
        2105: Dependencies & Territories
              Description: Includes issues related to terroritial and dependency issues and devolution
        2199: Other
              Description: Includes issues related to other public lands policy subtopics
    23. Culture
        2300: General
            Description: Includes issues related to general cultural policy issues

