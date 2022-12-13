
### This scripts downloads and cleans the data for the CAP- SotU dataset

# load packages
import pandas as pd
import numpy as np
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


### Load & prepare data

#set working directory for local runs
"""print(os.getcwd())
os.chdir("./NLI-experiments")
print(os.getcwd())"""


## load data
# overview of CAP data: https://www.comparativeagendas.net/datasets_codebooks
# overall CAP master codebook: https://www.comparativeagendas.net/pages/master-codebook
# SOTU codebook 2015: https://comparativeagendas.s3.amazonaws.com/codebookfiles/State_of_the_Union_Address_Codebook.pdf

df = pd.read_csv("https://comparativeagendas.s3.amazonaws.com/datasetfiles/US-Executive_State_of_the_Union_Speeches-20.csv")
print(df.columns)
print(len(df))



#### Data Cleaning

## data cleaning

# contains two types of CAP topics
# based on codebook, seems like PAP is older policy agendas project code from US, while CAP is newer, more international project code
# ! in CAP-us-courts it made more sense to use pap_majortopic
df_cl = df[["description", 'majortopic', 'subtopic', "year", "president", "pres_party", "id"]].copy(deep=True)
print(len(df_cl))

# remove NAs
df_cl = df_cl[~df_cl.description.isna()]
print(len(df_cl))
# remove very short strings
df_cl = df_cl[df_cl.description.str.len().ge(30)]  # removes X. mostly noise, some content like "Amen.	"
print(len(df_cl))
df_cl = df_cl[~df_cl.description.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
print(len(df_cl))
# are there unique texts which are annotated with more than one type? Yes, 105. String like " ", ".", "Thank you very much", "#NAME?", "It's the right thing to do."
#df_cl = df_cl.groupby(by="description").filter(lambda x: len(x.value_counts("majortopic")) == 1)
#print(len(df_cl))
# remove duplicates
# maintain duplicates to maintain sequentiality of texts
#df_cl = df_cl[~df_cl.description.duplicated(keep="first")]  # 170 duplicates
#print(len(df_cl))

# renumber "Other" cateogry label from -555 to 99
df_cl.majortopic = df_cl.majortopic.replace(-555, 99)
df_cl.subtopic = df_cl.subtopic.replace(-555, 99)

# rename columns
df_cl = df_cl.rename(columns={"majortopic": "label_cap2", "subtopic": "label_cap4", "description": "text", "id": "id_original"})

df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream 


### adding label_text to label ids
# label names from master codebook as of Oct. 2021, https://www.comparativeagendas.net/pages/master-codebook
label_text_map_cap2 = {  
    1: "Macroeconomics",
    2: "Civil Rights", 
    3: "Health",
    4: "Agriculture",
    5: "Labor",
    6: "Education",
    7: "Environment",
    8: "Energy",
    9: "Immigration",
    10: "Transportation",
    12: "Law and Crime",  
    13: "Social Welfare",
    14: "Housing",  
    15: "Domestic Commerce", 
    16: "Defense",
    17: "Technology",  
    18: "Foreign Trade",
    19: "International Affairs",  
    20: "Government Operations",
    21: "Public Lands",  
    23: "Culture", 
    99: "Other",  
}

df_cl["label_cap2_text"] = df_cl.label_cap2.map(label_text_map_cap2)
print(f"Maybe label_cap4 later too. Very fine-grained number of classes: {len(df_cl.label_cap4.unique())}. Makes for interesting data")

# labels numbers in alphabetical order of text
df_cl["label"] = pd.factorize(df_cl["label_cap2_text"], sort=True)[0]
df_cl["label_text"] = df_cl["label_cap2_text"]

df_cl = df_cl[["label", "label_text", "text", 'label_cap2', "label_cap2_text", 'label_cap4',  "year", "president", "pres_party", "id_original"]]

# test that label_cap2 and label_cap2_text correspond
assert len(df_cl[df_cl.label_cap2_text.isna()]) == 0  # each label_cap2 could be mapped to a label text. no label text is missing. 
print(np.sort(df_cl["label_cap2_text"].value_counts().tolist()) == np.sort(df_cl["label_cap2"].value_counts().tolist()))

df_cl.label_cap2_text.value_counts()



### augmenting text column

## new column where every sentence is merged with previous sentence
n_unique_doc_lst = []
n_unique_doc = 0
text_preceding = []
text_following = []
for name_group, df_group in df_cl.groupby(by=["president", "year"], sort=False):  # over each speech to avoid merging sentences accross manifestos
    n_unique_doc += 1
    df_group = df_group.reset_index(drop=True)  # reset index to enable iterating over index
    #text_ext = []
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
    #text_ext2_all.append(text_ext)
    n_unique_doc_lst.append([n_unique_doc] * len(df_group["text"]))
n_unique_doc_lst = [item for sublist in n_unique_doc_lst for item in sublist]

# create new columns
df_cl["text_original"] = df_cl["text"]
df_cl = df_cl.drop(columns=["text"])
df_cl["text_preceding"] = text_preceding
df_cl["text_following"] = text_following
df_cl["doc_id"] = n_unique_doc_lst  # column with unique doc identifier



## test how many sentences have same type as preceding / following sentence
test_lst = []
test_lst2 = []
test_lst_after = []
for name_df, group_df in df_cl.groupby(by="doc_id", group_keys=False, as_index=False, sort=False):
  for i in range(len(group_df)):
    # one preceding text
    if i == 0:
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1]:
      test_lst.append("same_before")
    else:
      #test_lst.append(f"different label before: {group_df['label_text'].iloc[i-1]}")
      test_lst.append(f"different label before")
    # two preceding texts
    """if i < 2:
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1] == group_df["label_text"].iloc[i-2]:
      test_lst2.append("same_two_before")
    else:
      test_lst2.append("different_two_before")"""
    # for following texts
    if i >= len(group_df)-1:
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i+1]:
      test_lst_after.append("same_after")
    else:
      #test_lst_after.append(f"different label after: {group_df['label_text'].iloc[i+1]}")
      test_lst_after.append(f"different label after")


print(pd.Series(test_lst).value_counts(normalize=True), "\n")  
print(pd.Series(test_lst_after).value_counts(normalize=True), "\n")  
# SOTU: 75 % of sentences have the same type as the preceeding sentence. also 75% for following sentence. #  concatenating preceding/following leads to data leakage? 25% different class which can confuse the model, its's random and same for all models
# Manifesto: 56 % of sentenes have same type as preceding sentence. # including preceding sentence should not provide illegitimate advantage to classifier


#### Train-Test-Split

### simplified dataset
from sklearn.model_selection import train_test_split

# normal sample based on subcat
#df_train, df_test = train_test_split(df_cl, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl["label_text"])
# sample based on docs - to make test set composed of entirely different docs - avoid data leakage when including surrounding sentences
doc_id_train = pd.Series(df_cl.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test = df_cl[~df_cl.doc_id.isin(doc_id_train)].doc_id.unique().tolist()
print(len(doc_id_train))
print(len(doc_id_test))
assert sum([doc_id in doc_id_train for doc_id in doc_id_test]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train = df_cl[df_cl.doc_id.isin(doc_id_train)]
df_test = df_cl[~df_cl.doc_id.isin(doc_id_train)]

# sample for faster testing - full data at the very end
samp_per_class_max = 100
df_test_samp = df_test.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), samp_per_class_max), random_state=SEED_GLOBAL))

print(f"Overall train size: {len(df_train)}")
print(f"Overall test size: {len(df_test)} - sampled test size: {len(df_test_samp)}")
df_train_test_distribution = pd.DataFrame([df_train.label_text.value_counts().rename("train"), df_test.label_text.value_counts().rename("test"), 
                                           df_test_samp.label_text.value_counts().rename("test_sample"), df_cl.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution


# ## Save data

# dataset statistics
text_length = [len(text) for text in df_cl.text_original]
text_context_length = [len(text) + len(preceding) + len(following) for text, preceding, following in zip(df_cl.text_original, df_cl.text_preceding, df_cl.text_following)]
print("Average number of characters in text: ", int(np.mean(text_length)))
print("Average number of characters in text with context: ", int(np.mean(text_context_length)))

print(os.getcwd())

df_cl.to_csv("./data_clean/df_cap_sotu_all.csv")
df_train.to_csv("./data_clean/df_cap_sotu_train.csv")
df_test.to_csv("./data_clean/df_cap_sotu_test.csv")

