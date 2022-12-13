
### This scripts downloads and cleans the data for the Sentiment-News-Econ dataset

# Install and load packages
import pandas as pd
import numpy as np
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


### Load & prepare data
#set working directory for local runs
"""print(os.getcwd())
if "NLI-experiments" not in os.getcwd().split("/")[-1]:
    os.chdir("./NLI-experiments")  #os.chdir("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments")
print(os.getcwd())"""



## load data
# data repository: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MXKRDE
df_test_raw = pd.read_csv("./data_raw/sentiment-econ/ground-truth-dataset-cf.csv", sep=",", encoding='utf-8') #encoding='utf-8',  # low_memory=False  #lineterminator='\t',
df_train_raw = pd.read_csv("./data_raw/sentiment-econ/5AC.csv", sep=",", encoding='utf-8') #encoding='utf-8',  # low_memory=False  #lineterminator='\t',

df = pd.concat([df_test_raw, df_train_raw])

print(df.columns)
print(len(df))

df_overlap = df_test_raw[df_test_raw.text.isin(df_train_raw.text)]
print(f"Numers of rows in df_test which have same text as df_train: {len(df_overlap)}")
print(f"This is {round((len(df_overlap) / len(df_test_raw)) * 100, 1)}% of the entire test set (lenght: {len(df_test_raw)})")


#### Data Cleaning
### data cleaning

## initial cleaning
df_train_raw_cl = df_train_raw.copy(deep=True)
df_train_raw_cl["split"] = "train"
df_train_raw_cl = df_train_raw_cl.rename(columns={"text": "text_body"})
df_train_raw_cl.positivity = [99 if str(element) == "nan" else int(element) for element in df_train_raw_cl.positivity]  # convert nan to 99 and all labels to int for easier downstream processing

df_test_raw_cl = df_test_raw.copy(deep=True)
df_test_raw_cl["split"] = "test"
df_test_raw_cl = df_test_raw_cl.rename(columns={"text": "text_body"})

## exclude texts which are not relevant, or neutral sentiment from train set - to maintain comparability with original paper
# df train
df_train_raw_cl = df_train_raw_cl[df_train_raw_cl.relevance == "yes"]  # exclude not relevant texts (nan is for test set)
df_train_raw_cl = df_train_raw_cl[~df_train_raw_cl.positivity.isin([5, 99])]  # exclude neutral texts and those with no positivity score (no indication about performance)
# could make a complex task out of recognising non-relevant or neutral texts (but these categories don't exist in ground-truth test dataset)(would need to create train test split. also: neutral category is problematic, because of continuous score and annotator disagreement)

## harmonise labels and label text
# they only used data with > 5 for positive and < 5 for negative (5 is neutral) and discard neutral texts. based on 05-SML-classifier.R https://codeocean.com/capsule/9240688/tree/v1
# they also merged headlines with body. based on classifier.py https://codeocean.com/capsule/9240688/tree/v1
# unclear from the code how they handle duplicate codings of the same text. seems like key just process duplicates, which can mean that one text is one positive and one negative (if once 4 and once 6 labeled)

df_train_raw_cl["label"] = [0 if label < 5 else 1 for label in df_train_raw_cl.positivity]
df_train_raw_cl["label_text"] = ["negative" if label < 5 else "positive" for label in df_train_raw_cl.positivity]

df_test_raw_cl["label"] = [0 if label == "negative" else 1 for label in df_test_raw_cl.positivity]
df_test_raw_cl["label_text"] = df_test_raw_cl.positivity

## concatenate test and train set
df_cl = pd.concat([df_train_raw_cl, df_test_raw_cl])

#df_cl["text"] = [headline + ". " + body for headline, body in zip(df_cl["headline"], df_cl["body"])]
df_cl["text"] = df_cl["headline"] + ". " + df_cl["text_body"]

# remove very short and long strings - too much noise - no issue in this dataset
#df_cl = df_cl[df_cl.text.str.len().ge(30)]  
#print(len(df_cl))
#df_cl = df_cl[~df_cl.text.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
#print(len(df_cl))

# remove raw text columns to avoid memory issues (and date)
df_cl = df_cl[["label", "label_text", "text", "split", "articleid", "relevance", "positivity"]]




### ! some texts are both in train and in test set !
# inspect duplicates: 
df_train_test_leakage = df_cl.groupby(by="text", group_keys=False, as_index=False, sort=False).filter(lambda x: not all(isinstance(label, type(x.positivity.iloc[0])) for label in x.positivity))
print("Leaked texts from train to test", len(df_train_test_leakage[~df_train_test_leakage.text.duplicated(keep="first")]))
df_train_test_leakage.drop_duplicates(subset=["text", "split"], keep="first")
## remove the rows where text is both in train and in test
print("df_cl before removing leaked texts", len(df_cl))
df_cl = df_cl.groupby(by="text", group_keys=False, as_index=False, sort=False).filter(lambda x: all(isinstance(label, type(x.positivity.iloc[0])) for label in x.positivity))  # for df_test positivity is a string, for df_train positivity is int
print("df_cl after removing leaked texts", len(df_cl))
# I've removed the overlapping texts in both train and test. Possibly better solution could have been to only remove in train.



### remove duplicate texts and average the sentiment score
# ! decision on threshold is relevant here. can remove around 1316 texts 
def deduplicated_and_average(df):
  # only do deduplication and averaging on train set. for test set, positivity is string
  if isinstance(df["positivity"].iloc[0], int):  
    mean_sentiment = df.positivity.mean()
    df = df[~df.text.duplicated(keep="first")]
    df.positivity = mean_sentiment
    # adapt label and label text to mean sentiment
    threshold_upper = 5.9
    threshold_lower = 4.1
    if mean_sentiment > threshold_upper: 
      df.label = 1
      df.label_text = "positive"
    elif mean_sentiment < threshold_lower:
      df.label = 0
      df.label_text = "negative"
    elif (mean_sentiment <= threshold_upper) or (mean_sentiment >= threshold_lower):
      df.label = 99
      df.label_text = "neutral"
    else:
      raise Exception(f"Something was not caught by code. mean sentiment: {mean_sentiment}")
  return df


df_cl = df_cl.groupby(by="text", group_keys=False, as_index=False, sort=False).apply(lambda x: deduplicated_and_average(x))
print("df_cl deduplicated, but including new neutral texts: ", len(df_cl))
print("Number of new neutral texts (based on threshold): ", len(df_cl[df_cl.label == 99]))
df_cl = df_cl[df_cl.label != 99]  # remove rows where the mean too close to 5 / neutral
print("final df_cl: ", len(df_cl))

# reset index
df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream 

print("\n")
print(df_cl.label_text.value_counts(), "\n")
print(df_cl.split.value_counts())

len(df_cl[df_cl.text.duplicated(keep=False)])


### length of texts
# ! some texts are quite long. processing with transformers max_len=512 will probably lead to truncation for a few texts
text_length = [len(text) for text in df_cl.text]
pd.Series(text_length).value_counts(bins=10).plot.bar()


#### Train-Test-Split

### separate train and test here

## split is predefined from dataset
df_train = df_cl[df_cl.split == "train"]
df_test = df_cl[df_cl.split == "test"]
assert len(df_train) + len(df_test) == len(df_cl)

# show train-test distribution
df_train_test_dist = pd.DataFrame([df_train.label_text.value_counts().rename("train"), df_test.label_text.value_counts().rename("test"), df_cl.label_text.value_counts().rename("data_all")]).transpose()
df_train_test_dist


### Save Data

# dataset statistics
text_length = [len(text) for text in df_cl.text]
print("Average number of characters in text: ", int(np.mean(text_length)), "\n")

print(os.getcwd())

df_cl.to_csv("./data_clean/df_sentiment_news_econ_all.csv")
df_train.to_csv("./data_clean/df_sentiment_news_econ_train.csv")
df_test.to_csv("./data_clean/df_sentiment_news_econ_test.csv")

