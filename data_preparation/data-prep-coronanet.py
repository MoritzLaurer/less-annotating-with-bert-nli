
### This scripts downloads and cleans the data for the CoronaNet dataset

# load packages
import pandas as pd
import numpy as np
import random
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


### Load & prepare data

#set working directory for local runs
"""print(os.getcwd())
os.chdir("./NLI-experiments")
print(os.getcwd())"""

## load data
# Overview of CoronaNet - https://www.coronanet-project.org/
# using event data format
# ! dataset is updated regularly. Working with github version commit 24.01.22
# https://github.com/CoronaNetDataScience/corona_tscs/blob/64b0ef942aea98057e41fed16794284847e34cd6/data/CoronaNet/data_bulk/coronanet_release.csv.gz
df = pd.read_csv("https://github.com/CoronaNetDataScience/corona_tscs/raw/64b0ef942aea98057e41fed16794284847e34cd6/data/CoronaNet/data_bulk/coronanet_release.csv.gz")
print(df.columns)
print(len(df))


# ### Data Cleaning

## Select relevant columns
df_cl = df[['record_id', 'policy_id', 'entry_type', 'update_type', #'update_level', 'update_level_var',
       'description', #'date_announced', 'date_start', 'date_end', 'date_end_spec', 
       'ISO_A3', # 'country',  'ISO_A2',
       #'init_country_level', 'domestic_policy', 'province', 'ISO_L2', 'city',
       'type', 'type_sub_cat', 
       #'type_new_admin_coop', 'type_vac_cat', 'type_vac_mix', 'type_vac_reg', 'type_vac_purchase', 'type_vac_group',
       #'type_vac_group_rank', 'type_vac_who_pays', 'type_vac_dist_admin',
       #'type_vac_loc', 'type_vac_cost_num', 'type_vac_cost_scale',
       #'type_vac_cost_unit', 'type_vac_cost_gov_perc', 'type_vac_amt_num',
       #'type_vac_amt_scale', 'type_vac_amt_unit', 'type_vac_amt_gov_perc',
       #'type_text', 'institution_cat', 'institution_status',
       #'institution_conditions', #'target_init_same', 'target_country',
       #'target_geog_level', 'target_region', 'target_province', 'target_city',
       #'target_other', 'target_who_what', 'target_who_gen', 'target_direction',
       #'travel_mechanism', 'compliance', 'enforcer', 'dist_index_high_est',
       #'dist_index_med_est', 'dist_index_low_est', 'dist_index_country_rank',
       'pdf_link', 'link', #'date_updated', 'recorded_date'
       ]].copy(deep=True)


## data cleaning
print(len(df_cl))

# remove NAs - no NAs
#df_cl = df_cl[~df_cl.description.isna()]
#df_cl = df_cl[~df_cl.type.isna()]
#print(len(df_cl))

# remove very short and long strings - too much noise
df_cl["description"] = df_cl["description"].str.replace("\n", " ")
df_cl = df_cl[df_cl.description.str.len().ge(30)]  # removes  67
print(len(df_cl))
df_cl = df_cl[~df_cl.description.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
print(len(df_cl))

# are there unique texts which are annotated with more than one type? Yes. removes around 10k
df_cl = df_cl.groupby(by="description").filter(lambda x: len(x.value_counts("type")) == 1)
print(len(df_cl))

# duplicates
# remove updates/duplicate policy ids - only work with unique policy_id. each update to a policy measure is a new row, with some slight updates at the end of the description text.
# best to only work with unique policy measures, not unique description strings (otherwise small description updates multiply one policy). ! deduplication on policy_id removes 27k rows (updates) of same policy measure
df_cl = df_cl[~df_cl.policy_id.duplicated(keep="first")]
print(len(df_cl))
# also remove duplicate texts
df_cl = df_cl[~df_cl.description.duplicated(keep="first")]  # removes around 8.5k
print(len(df_cl))

# remove very low n types
df_cl = df_cl[df_cl.type != "Missing/Not Applicable"]  # type only has 6 entries
print(len(df_cl))

# could remove badly performing labels with negative transfer risk
# "Other Policy Not Listed Above"  # seems  to contain: economic, social measures; other (like "day of  prayer against covid")
# ! should maintain this for realism
#df_cl = df_cl[~df_cl.type.str.contains("Other Policy Not Listed Above")]  #"Anti-Disinformation Measures|Public Awareness Measures|Other Policy Not Listed Above"
#print(len(df_cl))

# maintain and rename only key columns & rename colums so that trainer recognises them
df_cl = df_cl.rename(columns={"description": "text", "type": "label_text"})
# add numeric label column based on alphabetical label_text order
df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]

# final update
df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream 

print("\n")
df_cl.label_text.value_counts()


# ### Train-Test-Split
print(df_cl.columns)
df_cl = df_cl[["label", "label_text", "text", "type_sub_cat", "record_id", "policy_id", "ISO_A3", "pdf_link", "link"]]

### simplified dataset
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df_cl, test_size=0.3, random_state=SEED_GLOBAL, stratify=df_cl["label_text"])

# sample for faster testing before final runs
samp_per_class_max = 100
df_test_samp = df_test.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), samp_per_class_max), random_state=SEED_GLOBAL))

print(f"Overall train size: {len(df_train)}")
print(f"Overall test size: {len(df_test)} - sampled test size: {len(df_test_samp)}")
df_train_test_distribution = pd.DataFrame([df_train.label_text.value_counts().rename("train"), df_test.label_text.value_counts().rename("test"), 
                                           df_test_samp.label_text.value_counts().rename("test_sample"), df_cl.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution


### Save data

# dataset statistics
text_length = [len(text) for text in df_cl.text]
print("Average number of characters in text: ", int(np.mean(text_length)), "\n")

print(os.getcwd())

df_cl.to_csv("./data_clean/df_coronanet_20220124_all.csv")
df_train.to_csv("./data_clean/df_coronanet_20220124_train.csv")
df_test.to_csv("./data_clean/df_coronanet_20220124_test.csv")

