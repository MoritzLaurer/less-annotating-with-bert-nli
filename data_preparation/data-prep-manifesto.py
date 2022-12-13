
### This scripts loads and cleans the data for the Manifesto dataset

# load packages
import pandas as pd
import numpy as np
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


#set working directory for local runs
"""print(os.getcwd())
if "NLI-experiments" not in os.getcwd().split("/")[-1]:
    os.chdir("./NLI-experiments")  #os.chdir("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments")
print(os.getcwd())"""


## load dfs
# correct v5 codebook: https://manifesto-project.wzb.eu/down/papers/handbook_2021_version_5.pdf - the PDF on the following website is wrong, but html is correct: https://manifesto-project.wzb.eu/coding_schemes/mp_v5
# we are working with v4 for backwards compatibility https://manifesto-project.wzb.eu/down/papers/handbook_2011_version_4.pdf
# overview of changes from v4 to v5: https://manifesto-project.wzb.eu/down/papers/Evolution_of_the_Manifesto_Coding_Instructions_and_the_Category_Scheme.pdf
# switch was 2016/2017
# working with version provided by Manifesto team

df = pd.read_csv("./data_raw/manifesto/all_annotated_manifestos.zip", index_col="Unnamed: 0")

print(df.columns)
print(len(df))

df.cmp_code_hb4.value_counts()
df.cmp_code.value_counts()
df.eu_code.value_counts()
df.columns


# ### Data Cleaning

# deep copy
df_cl = df.copy(deep=True)
df_cl = df_cl[["text", "cmp_code", "cmp_code_hb4", "manifesto_id", "party", "date", "country_name", "testresult"]]  # "eu_code", "Text_CharsReplaced"

print(len(df_cl))

# only English texts from English speaking countries
country_lst = ["New Zealand", "United Kingdom", "Ireland", "Australia", "United States", "South Africa"] # Canada
df_cl = df_cl[df_cl.country_name.isin(country_lst)]
print(len(df_cl))

# check for NAs
df_cl = df_cl[~df_cl["text"].isna()]
print(len(df_cl))

# remove headlines
df_cl = df_cl[~df_cl["cmp_code_hb4"].isna()]  # 13k NA in English data. seem to be headlines and very short texts. they can have meaning
print(len(df_cl))
df_cl = df_cl[~df_cl["cmp_code_hb4"].str.match("H", na=False)]  # 7.6k headlines
print(len(df_cl))

# remove very short and long strings - too much noise
df_cl = df_cl[df_cl.text.str.len().ge(30)]  # removes  67
print(len(df_cl))
df_cl = df_cl[~df_cl.text.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
print(len(df_cl))

## duplicates
# remove texts where exact same string has different code? Can keep it for experiments with context - shows value of context for disambiguation
#df_cl = df_cl.groupby(by="text").filter(lambda x: len(x.cmp_code.unique()) == 1)
#print(len(df_cl))
# maintain duplicates to maintain sequentiality of texts
#df_cl = df_cl[~df_cl.text.duplicated(keep="first")]  # around 7k
#print(len(df_cl))


### Wrangling of labels and label text

# translating label codes to label text with codebook mapping. MPDS2020a-1
# see codebook https://manifesto-project.wzb.eu/down/papers/handbook_2011_version_4.pdf
# Note that the "main" codes are from v4 for backwards compatibility with older data
# for new v5 categories: everything was aggregated up into the old v4 categories, except for 202.2, 605.2 und 703.2, which where added to 000. 
df_label_map = pd.read_csv("./data_raw/manifesto/codebook_categories_MPDS2020a-1.csv")

df_label_map.domain_name = df_label_map.domain_name.fillna("No other category applies")  # for some reason domain_name in case of no label is NaN. replace with expressive string

# translating label codes to label text with codebook mapping
# info on two column cmp_codes (v5 codebook) and cmp_code_hb4 (v4 codebook - backwardscompatible): "Außerdem enthält die Spalte cmp_code jetzt einfach die unmodifizierten original cmp_codes (also auch die neuen handbuch 5 Kategorien, wo sie angewendet wurden). Dafür gibt es jetzt cmp_code_hb4, in der dann alles in hb4 umgewandelt wurde (also 605.2 zu "000", 202.2 zu "000" und 703.2 zu "000", alle übrigen 5er Kategorien hochaggregiert)

# mapping of numeric codes to domain and subcat titles. only use v4 codebook numeric codes with XX.0 floats, ignore XX.1 codes from codebook because not present in masterfile shared by Tobias due to backwords compatibility
code_to_domain_map = {int(row["code"]): row["domain_name"] for i, row in df_label_map.iterrows() if str(row["code"])[-1] == "0"}  # only take labels which don't have old sub category. old subcategories indicated by XX.1 floats, main categories indicated by XX.0 floats
code_to_subcat_map = {int(row["code"]): row["title"] for i, row in df_label_map.iterrows() if str(row["code"])[-1] == "0"}

# labels were name changed from v4 to v5 - but not changing it because working with v4. 
#code_to_subcat_map.update({416: 'Anti-Growth Economy and Sustainability', 605: 'Law and Order', 703: 'Agriculture and Farmers'})
#code_to_subcat_map["Anti-Growth Economy and Sustainability"] = code_to_subcat_map.pop("Anti-Growth Economy: Positive")
#code_to_subcat_map["Law and Order"] = code_to_subcat_map.pop("Law and Order: Positive")
#code_to_subcat_map["Agriculture and Farmers"] = code_to_subcat_map.pop("Agriculture and Farmers: Positive")

df_cl["label_domain_text"] = df_cl.cmp_code_hb4.astype(int).map(code_to_domain_map)
df_cl["label_subcat_text"] = df_cl.cmp_code_hb4.astype(int).map(code_to_subcat_map)
print(len(df_cl.label_domain_text.value_counts()))
print(len(df_cl.label_subcat_text.value_counts()))


## ! decide on label level to use for downstream analysis
df_cl["label_text"] = df_cl["label_subcat_text"]
df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]

# test that label and label_text correspond
assert len(df_cl[df_cl.label_text.isna()]) == 0  # each label_cap2 could be mapped to a label text. no label text is missing. 
print(np.sort(df_cl["label_text"].value_counts().tolist()) == np.sort(df_cl["label"].value_counts().tolist()))

# final update
df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream 

print(df_cl.label_text.value_counts(), "\n")
print(df_cl.country_name.value_counts())



### augmenting text column

## new column where every sentence is merged with previous sentence
n_unique_doc_lst = []
n_unique_doc = 0
text_preceding = []
text_following = []
for name_group, df_group in df_cl.groupby(by=["manifesto_id"], sort=False):  # over each speech to avoid merging sentences accross manifestos
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

df_cl = df_cl[["label", "label_text", "text_original", "label_domain_text", "label_subcat_text", "text_preceding", "text_following", "manifesto_id", "doc_id", "country_name", "date", "party", "cmp_code_hb4", "cmp_code"]]




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
# Manifesto: 57 % of sentences have same type as preceding sentence (57 class). # including preceding sentence should not provide illegitimate advantage to classifier




#### simplify labeling scheme to make complex reasoning a separate task (?)
# ! careful: 000 includes 202.2, 605.2 und 703.2 - based on manifesto project recommendations - makes 000 category harder

# reducing 57 labels to same 44 simplified labels as Osnabrügge et al.
# eventually did not use this column in paper
codes_v4_to_simple = {
    # simplified
    "Foreign Special Relationships: Positive": "Foreign Special Relationships", "Foreign Special Relationships: Negative": "Foreign Special Relationships",
    "Military: Positive": "Military", "Military: Negative": "Military",
    "Internationalism: Positive": "Internationalism", "Internationalism: Negative": "Internationalism",
    "European Community/Union: Positive": "European Community/Union", "European Community/Union: Negative": "European Community/Union",
    "Constitutionalism: Positive": "Constitutionalism", "Constitutionalism: Negative": "Constitutionalism",
    "Decentralization": "Centralisation & Decentralisation", "Centralisation": "Centralisation & Decentralisation",
    "Protectionism: Positive": "Protectionism", "Protectionism: Negative": "Protectionism",
    "Welfare State Expansion": "Welfare State", "Welfare State Limitation": "Welfare State",  # 1000
    "Education Expansion": "Education", "Education Limitation": "Education",  # < 200
    "National Way of Life: Positive": "National Way of Life", "National Way of Life: Negative": "National Way of Life",
    "Traditional Morality: Positive": "Traditional Morality", "Traditional Morality: Negative": "Traditional Morality",
    "Multiculturalism: Positive": "Multiculturalism", "Multiculturalism: Negative": "Multiculturalism",
    "Labour Groups: Positive": "Labour Groups", "Labour Groups: Negative": "Labour Groups",  # < 300
    # categories with confusing change vom v4 to v5 in csv codebook 
    "Anti-Growth Economy: Positive": "Anti-Growth Economy: Positive",  # new v5 sustainability category is included here. but keeping the v4 label
    "Law and Order: Positive": "Law and Order: Positive",  # law and order negative not included, because not part of v4 codebook. was added to 000 / other category
    "Agriculture and Farmers: Positive": "Agriculture and Farmers: Positive",  # negative not included here, because not part of v4 codebook. was added to 000 / other category
    "No other category applies": "No other category applies",  #"No meaningful category applies": "No other category applies",  # label text slightly different in csv version of codebook ... going with csv version    
    # name unchanged
    "Civic Mindedness: Positive": "Civic Mindedness: Positive",
    "Anti-Imperialism": "Anti-Imperialism", "Marxist Analysis": "Marxist Analysis",
    "Peace": "Peace", 
    "Freedom and Human Rights": "Freedom and Human Rights",
    "Democracy": "Democracy",  # democracy negative was introduced in v5, so was added to 000
    "Governmental and Administrative Efficiency": "Governmental and Administrative Efficiency",
    "Political Corruption": "Political Corruption",
    "Political Authority": "Political Authority",
    "Environmental Protection": "Environmental Protection",
    "Culture: Positive": "Culture: Positive",
    "Equality: Positive": "Equality: Positive",
    "Middle Class and Professional Groups": "Middle Class and Professional Groups",
    "Underprivileged Minority Groups": "Underprivileged Minority Groups",
    "Non-economic Demographic Groups": "Non-economic Demographic Groups",
    "Technology and Infrastructure: Positive": "Technology and Infrastructure: Positive",
    # too many overlapping economic categories
    "Free Market Economy": "Free Market Economy",
    "Incentives: Positive": "Incentives: Positive",
    "Market Regulation": "Market Regulation",
    "Economic Planning": "Economic Planning",
    "Corporatism/Mixed Economy": "Corporatism/Mixed Economy",
    "Economic Goals": "Economic Goals",
    "Keynesian Demand Management": "Keynesian Demand Management",  #162
    "Economic Growth: Positive": "Economic Growth: Positive",
    "Controlled Economy": "Controlled Economy",
    "Nationalisation": "Nationalisation",
    "Economic Orthodoxy": "Economic Orthodoxy",
}


## mapping 56 sub-types to simplified merged types
df_cl["label_subcat_text_simple"] = [codes_v4_to_simple[sub_type_v4] for sub_type_v4 in df_cl.label_subcat_text]
print("Number of simplified sub-types: ", len(df_cl.label_subcat_text_simple.unique()))

# remove types with low frequency ? - no better to keep
thresh_removal = 300
infreq_types = df_cl.label_subcat_text_simple.value_counts()[df_cl.label_subcat_text_simple.value_counts() < thresh_removal].index.to_list()
print("Infrequent types: ", infreq_types)
#df_cl = df_cl[~df_cl.label_subcat_text_simple.isin(infreq_types)]
#print(f"Number of sub-types after removing low frequency {thresh_removal}, unimportant types: ", len(df_cl.label_subcat_text_simple.unique()))



### df_complex with only complex labels
## isolating specifically complex categories
"""code_v4_complex_dic = {
    "Foreign Special Relationships: Positive": "Foreign Special Relationships", "Foreign Special Relationships: Negative": "Foreign Special Relationships",
    "Military: Positive": "Military", "Military: Negative": "Military",
    "Internationalism: Positive": "Internationalism", "Internationalism: Negative": "Internationalism",
    "European Community/Union: Positive": "European Community/Union", "European Community/Union: Negative": "European Community/Union",
    "Constitutionalism: Positive": "Constitutionalism", "Constitutionalism: Negative": "Constitutionalism",
    "Decentralization": "Centralisation & Decentralisation", "Centralisation": "Centralisation & Decentralisation",
    "Protectionism: Positive": "Protectionism", "Protectionism: Negative": "Protectionism",
    "Welfare State Expansion": "Welfare State", "Welfare State Limitation": "Welfare State",  # 1000
    "Education Expansion": "Education", "Education Limitation": "Education",  # < 200
    "National Way of Life: Positive": "National Way of Life", "National Way of Life: Negative": "National Way of Life",
    "Traditional Morality: Positive": "Traditional Morality", "Traditional Morality: Negative": "Traditional Morality",
    "Multiculturalism: Positive": "Multiculturalism", "Multiculturalism: Negative": "Multiculturalism",
    "Labour Groups: Positive": "Labour Groups", "Labour Groups: Negative": "Labour Groups",  # < 300
}
code_v4_complex_labels = [key for key in code_v4_complex_dic]

df_complex = df_cl[df_cl.label_subcat_text.isin(code_v4_complex_labels)]

## adapt label numbering - has to start from 0 and count up linearly
df_complex["label_text"] = df_complex["label_subcat_text"]
df_complex["label"] = pd.factorize(df_complex["label_text"], sort=True)[0]

print(len(df_complex.label_text.value_counts()))
df_complex.label_text.value_counts()

#DataTable(df_complex, num_rows_per_page=5, max_rows=10_000)"""



# ### Train-Test-Split

### simplified dataset
from sklearn.model_selection import train_test_split

# normal sample based on subcat
#df_train, df_test = train_test_split(df_cl, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl["label_subcat"])
# sample based on docs - to make test set composed of entirely different docs - avoid data leakage when including surrounding sentences
doc_id_train = pd.Series(df_cl.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test = df_cl[~df_cl.doc_id.isin(doc_id_train)].doc_id.unique().tolist()
print(len(doc_id_train))
print(len(doc_id_test))
assert sum([doc_id in doc_id_train for doc_id in doc_id_test]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train = df_cl[df_cl.doc_id.isin(doc_id_train)]
df_test = df_cl[~df_cl.doc_id.isin(doc_id_train)]

# small sample just for faster testing
#samp_per_class_max = 100
#df_test_samp = df_test.groupby(by="label_subcat_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), samp_per_class_max), random_state=SEED_GLOBAL))

print(f"Overall train size: {len(df_train)}")
#print(f"Overall test size: {len(df_test)} - sampled test size: {len(df_test_samp)}")
print(f"Overall test size: {len(df_test)}")
df_train_test_distribution = pd.DataFrame([df_train.label_domain_text.value_counts().rename("train"), df_test.label_domain_text.value_counts().rename("test"),
                                           #df_test_samp.label_domain_text.value_counts().rename("test_sample"),
                                           df_cl.label_domain_text.value_counts().rename("all")]).transpose()
df_train_test_distribution


### complex dataset
from sklearn.model_selection import train_test_split

"""# normal sample based on subcat
#df_train_complex, df_test_complex = train_test_split(df_complex, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_complex["label_subcat_text"])
# sample based on docs - to make test set composed of entirely different docs - avoid data leakage when including surrounding sentences
doc_id_train_complex = pd.Series(df_complex.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_complex = df_complex[~df_complex.doc_id.isin(doc_id_train)].doc_id.unique().tolist()
print(len(doc_id_train_complex))
print(len(doc_id_test_complex))
assert sum([doc_id in doc_id_train_complex for doc_id in doc_id_test_complex]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_complex = df_complex[df_complex.doc_id.isin(doc_id_train)]
df_test_complex = df_complex[~df_complex.doc_id.isin(doc_id_train)]

# sample for faster testing - full data at the very end
samp_per_class_max = 100
df_test_complex_samp = df_test_complex.groupby(by="label_subcat_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), samp_per_class_max), random_state=SEED_GLOBAL))

print(f"Overall train size: {len(df_train_complex)}")
print(f"Overall test size: {len(df_test_complex)} - sampled test size: {len(df_test_complex_samp)}")
df_train_test_distribution_complex = pd.DataFrame([df_train_complex.label_subcat_text.value_counts().rename("train"), df_test_complex.label_subcat_text.value_counts().rename("test"), 
                                           df_test_complex_samp.label_subcat_text.value_counts().rename("test_sample"), df_complex.label_subcat_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_complex"""



#### Small complex tasks

### Military positive vs. negative

df_cl_military = df_cl.copy(deep=True)

label_text_military = [label if label in ["Military: Positive", "Military: Negative"] else "Other" for label in df_cl_military.label_subcat_text]

df_cl_military["label_text"] = label_text_military
df_cl_military["label"] = pd.factorize(df_cl_military["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_military, df_test_military = train_test_split(df_cl_military, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_military["label_text"])
# better train test split to avoid data leakage
doc_id_train_military = pd.Series(df_cl_military.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_military = df_cl_military[~df_cl_military.doc_id.isin(doc_id_train_military)].doc_id.unique().tolist()
print(len(doc_id_train_military))
print(len(doc_id_test_military))
assert sum([doc_id in doc_id_train_military for doc_id in doc_id_test_military]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_military = df_cl_military[df_cl_military.doc_id.isin(doc_id_train_military)]
df_test_military = df_cl_military[~df_cl_military.doc_id.isin(doc_id_train_military)]

# down sampling the "other" category
df_train_military = df_train_military.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_military.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_military = df_test_military.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_military.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_military = pd.concat([df_train_military, df_test_military])

# show data distribution
print(f"Overall train size: {len(df_train_military)}")
print(f"Overall test size: {len(df_test_military)}")
df_train_test_distribution_military = pd.DataFrame([df_train_military.label_text.value_counts().rename("train"), df_test_military.label_text.value_counts().rename("test"), 
                                                   df_cl_military.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_military



### Protectionism positive vs. negative
df_cl_protectionism = df_cl.copy(deep=True)

label_text_protectionism = [label if label in ["Protectionism: Positive", "Protectionism: Negative"] else "Other" for label in df_cl_protectionism.label_subcat_text]

df_cl_protectionism["label_text"] = label_text_protectionism
df_cl_protectionism["label"] = pd.factorize(df_cl_protectionism["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_protectionism, df_test_protectionism = train_test_split(df_cl_protectionism, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_protectionism["label_text"])
# better train test split to avoid data leakage
doc_id_train_protectionism = pd.Series(df_cl_protectionism.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_protectionism = df_cl_protectionism[~df_cl_protectionism.doc_id.isin(doc_id_train_protectionism)].doc_id.unique().tolist()
print(len(doc_id_train_protectionism))
print(len(doc_id_test_protectionism))
assert sum([doc_id in doc_id_train_protectionism for doc_id in doc_id_test_protectionism]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_protectionism = df_cl_protectionism[df_cl_protectionism.doc_id.isin(doc_id_train_protectionism)]
df_test_protectionism = df_cl_protectionism[~df_cl_protectionism.doc_id.isin(doc_id_train_protectionism)]

# down sampling the "other" category
df_train_protectionism = df_train_protectionism.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_protectionism.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_protectionism = df_test_protectionism.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_protectionism.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_protectionism = pd.concat([df_train_protectionism, df_test_protectionism])

# show data distribution
print(f"Overall train size: {len(df_train_protectionism)}")
print(f"Overall test size: {len(df_test_protectionism)}")
df_train_test_distribution_protectionism = pd.DataFrame([df_train_protectionism.label_text.value_counts().rename("train"), df_test_protectionism.label_text.value_counts().rename("test"), 
                                                   df_cl_protectionism.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_protectionism




### Traditional Morality positive vs. negative

df_cl_morality = df_cl.copy(deep=True)

label_text_morality = [label if label in ["Traditional Morality: Positive", "Traditional Morality: Negative"] else "Other" for label in df_cl_morality.label_subcat_text]

df_cl_morality["label_text"] = label_text_morality
df_cl_morality["label"] = pd.factorize(df_cl_morality["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_morality, df_test_morality = train_test_split(df_cl_morality, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_morality["label_text"])
# better train test split to avoid data leakage
doc_id_train_morality = pd.Series(df_cl_morality.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_morality = df_cl_morality[~df_cl_morality.doc_id.isin(doc_id_train_morality)].doc_id.unique().tolist()
print(len(doc_id_train_morality))
print(len(doc_id_test_morality))
assert sum([doc_id in doc_id_train_morality for doc_id in doc_id_test_morality]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_morality = df_cl_morality[df_cl_morality.doc_id.isin(doc_id_train_morality)]
df_test_morality = df_cl_morality[~df_cl_morality.doc_id.isin(doc_id_train_morality)]

# down sampling the "other" category
df_train_morality = df_train_morality.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_morality.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_morality = df_test_morality.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_morality.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_morality = pd.concat([df_train_morality, df_test_morality])

# show data distribution
print(f"Overall train size: {len(df_train_morality)}")
print(f"Overall test size: {len(df_test_morality)}")
df_train_test_distribution_morality = pd.DataFrame([df_train_morality.label_text.value_counts().rename("train"), df_test_morality.label_text.value_counts().rename("test"), 
                                                   df_cl_morality.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_morality




### National Way of Life positive vs. negative

df_cl_nationalway = df_cl.copy(deep=True)

label_text_nationalway = [label if label in ["National Way of Life: Positive", "National Way of Life: Negative"] else "Other" for label in df_cl_nationalway.label_subcat_text]

df_cl_nationalway["label_text"] = label_text_nationalway
df_cl_nationalway["label"] = pd.factorize(df_cl_nationalway["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_nationalway, df_test_nationalway = train_test_split(df_cl_nationalway, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_nationalway["label_text"])
# better train test split to avoid data leakage
doc_id_train_nationalway = pd.Series(df_cl_nationalway.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_nationalway = df_cl_nationalway[~df_cl_nationalway.doc_id.isin(doc_id_train_nationalway)].doc_id.unique().tolist()
print(len(doc_id_train_nationalway))
print(len(doc_id_test_nationalway))
assert sum([doc_id in doc_id_train_nationalway for doc_id in doc_id_test_nationalway]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_nationalway = df_cl_nationalway[df_cl_nationalway.doc_id.isin(doc_id_train_nationalway)]
df_test_nationalway = df_cl_nationalway[~df_cl_nationalway.doc_id.isin(doc_id_train_nationalway)]

# down sampling the "other" category
df_train_nationalway = df_train_nationalway.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_nationalway.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_nationalway = df_test_nationalway.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_nationalway.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_nationalway = pd.concat([df_train_nationalway, df_test_nationalway])

# show data distribution
print(f"Overall train size: {len(df_train_nationalway)}")
print(f"Overall test size: {len(df_test_nationalway)}")
df_train_test_distribution_nationalway = pd.DataFrame([df_train_nationalway.label_text.value_counts().rename("train"), df_test_nationalway.label_text.value_counts().rename("test"), 
                                                   df_cl_nationalway.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_nationalway



#### Save data

# dataset statistics
text_length = [len(text) for text in df_cl.text_original]
text_context_length = [len(text) + len(preceding) + len(following) for text, preceding, following in zip(df_cl.text_original, df_cl.text_preceding, df_cl.text_following)]
print("Average number of characters in text: ", int(np.mean(text_length)))
print("Average number of characters in text with context: ", int(np.mean(text_context_length)))

print(os.getcwd())


## datasets used in paper
df_cl.to_csv("./data_clean/df_manifesto_all.csv")
df_train.to_csv("./data_clean/df_manifesto_train.csv")
df_test.to_csv("./data_clean/df_manifesto_test.csv")

df_cl_military.to_csv("./data_clean/df_manifesto_military_cl.csv")
df_train_military.to_csv("./data_clean/df_manifesto_military_train.csv")
df_test_military.to_csv("./data_clean/df_manifesto_military_test.csv")

df_cl_protectionism.to_csv("./data_clean/df_manifesto_protectionism_cl.csv")
df_train_protectionism.to_csv("./data_clean/df_manifesto_protectionism_train.csv")
df_test_protectionism.to_csv("./data_clean/df_manifesto_protectionism_test.csv")

df_cl_morality.to_csv("./data_clean/df_manifesto_morality_cl.csv")
df_train_morality.to_csv("./data_clean/df_manifesto_morality_train.csv")
df_test_morality.to_csv("./data_clean/df_manifesto_morality_test.csv")


## datasets not used in paper
"""
#df_complex.to_csv("./data_clean/df_manifesto_complex_all.csv")
#df_train_complex.to_csv("./data_clean/df_manifesto_complex_train.csv")
#df_test_complex.to_csv("./data_clean/df_manifesto_complex_test.csv")
#df_test_complex_samp.to_csv("./data_clean/df_manifesto_complex_test.csv")

#df_cl_nationalway.to_csv("./data_clean/df_manifesto_nationalway_cl.csv")
#df_train_nationalway.to_csv("./data_clean/df_manifesto_nationalway_train.csv")
#df_test_nationalway.to_csv("./data_clean/df_manifesto_nationalway_test.csv")
"""

