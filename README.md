## Paper Repository: "Less Annotating, More Classifying"

This is the replication code for the paper "Less Annotating, More Classifying 
– Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI"

This repository contains the full code and data for reproducing the paper. 
An earlier pre-print version of the paper is available [here](https://osf.io/wqc86/) 
and an improved version is currently under review. 
The repository will be updated upon final acceptance. 

We also provide an easy-to-use Google Colab notebook for testing BERT-NLI with free access to a GPU. 
We invite anyone to run and copy this notebook and to train their own BERT-NLI model 
on their own data: https://colab.research.google.com/drive/1-y7o-QRWp-OwGMe64CxQwQk2-o2jZFm3?usp=sharing 


### Description of programs/code
This section provides a high level descript of the analysis pipeline and each of its components. 

- First, the scripts in the directory "data_preparation" execute the following actions for each dataset:
  - downloading of data (or loading of local data from the "data_raw" directory for the datasets: Manifesto and Sentiment-Economy)
  - cleaning data (e.g. removing duplicates, removing very long texts etc.)
  - train-test-split for downstream analyses
  - The data is then saved in the "data_clean" directory.
- The batch scripts in the directory "batch-scripts" run the main analysis pipeline on the cleaned datasets.
  - One batch script was created for each dataset for the runs with Transformers on a GPU.
  For runs with classical machine learning, separate scripts were created to increase efficiency.
  - The batch scripts call on "analysis-classical-hyperparams.py" and "analysis-transf-hyperparams.py" for
  hyperparameter search for classical algorithms and Transformers respectively. 
  Moreover, they call on "analysis-classical-run.py" 
  and "analysis-transf-run.py" for the final run with optimal hyperparameters (on 3 separate random seeds
  for each dataset and sample size).
  - These scripts then save the raw data (the raw predictions as well as other meta-data) 
  to the "results-raw" directory. One sub-directory per dataset was created. The files
  were saved in Pickle format, which can be loaded in Python.
    - The result files titled "experiment_results..."
    contain the results for different algorithms, sample sizes (e.g. 00100samp for 100 randomly sampled texts).
    - The result files titled "optuna_study..." contain the hyperparameter study object produced by the Python
    Optuna library. These files can be loaded with Python Pickle and Optuna.
- The script "helpers.py" contains helper functions used by the analysis and hyperparameter search scripts.
- The script "hypothesis-hyperparams.py" contains multiple NLI hypotheses for each dataset.
The hyperparameter script uses this script to select the best hypotheses per dataset and sample size.
- The "data-analysis-viz.py" script uses the raw data from "data-raw" to calculate metrics and use them 
to create the plots for the main manuscript and the appendix. The resulting files are saved in the 
directory "/results/figures/"
- The "appendix.py" script uses the raw data from "data-raw" to calculate metrics and save them in tables
for the appendix (also called 'supplementary materials'). 
The resulting files are saved in the directory "/results/appendix/".


### Computational requirements

The full results for this paper were calculated with an A100 Nvidia GPU on 
the [Dutch Snellius High-Performance-Compute (HPC) system](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+hardware+and+file+systems).
A complete run requiresa a hyperparameter search for each of 4 algorithms on each of 8 datasets 
on each of ~5 sample sizes (depending on dataset size), as well as a final run with the best hyperparameters
with 3 random seeds for each of 8 datasets on each of ~5 sample sizes. This entire process takes around
1-2 weeks on an A100 GPU.

This replication code was adapted for a Code Ocean capsule which only provides 10 hours of GPU time.
The capsule therefore only reproduces the calculations on the raw outputs provided by Snellius HPC system.
This reproduction on intermediate outputs only requires a standard CPU (as of December 2022, e.g. 2,3 GHz Quad-Core Intel i5, 8GB RAM). 
See the CPUs available through Code Ocean here:
https://help.codeocean.com/en/articles/1120508-what-machine-will-my-code-run-on


### Instructions to Replicators

The Code Ocean capsule can be run by clicking on "Reproducible Run" to the top right.
The capsule will automatically install all dependencies and run the analysis on the raw outputs. 
The resulting figures and tables are written to the "results" directory.

- The "/results/figures/" directory contains the two main figures from the main text.
- The "/results/appendix/" directory contains all tables and figures from the appendix. 
- Each file is numbered based on the number of the table/figure in the main text.
Some additional Excel files where added to provide more details on disaggregated metrics.
From some files in the appendix, the appendix chapter (e.g. "B6") was added to help identify
the corresponding figure in the appendix. 
- For cross-checking numbers provided in the main text, I recommend checking table 22 from appendix D3
(file "D3_appendix_mean_difference_f1_macro.xlsx" in the "/results/appendix/" directory)

### Details on each data source

The paper does not produce its own data. 
It is based on multiple datasets from existing research projects.
If the data is publicly available, the scripts in the /code/data_preparation/ repository download the data
directly from the source.
If the data was provided to the authors by the dataset creators (Manifesto and Sentiment-Economy),
then the raw data is saved in the /code/data_raw/ repository in .csv format. 

An overview of each data source is provided in the table below. Details are provided in the full manuscript.

|Dataset|Task|Domain|Unit of Analysis|
|:---:|:---:|:---:|:---:|
|Manifesto Corpus (Burst et al. 2020)|Classify text in 8 general topics|Party Manifestos|Quasi-sentences|
|Sentiment Economy News (Barberá et al. 2021)|Differentiate if economy is performing well or badly according to the text (2 classes)|News articles|News headline & first paragraphs|
|US State of the Union Speeches (Policy Agendas Project 2015)|Classify text in policy topics (22 classes)|Presidential Speeches|Quasi-sentences|
|US Supreme Court Cases (Policy Agendas Project 2014)|Classify text in policy topics (20 classes)|Law, summaries of court cases and rulings|Court case summaries (multiple paragraphs)|
|CoronaNet (Cheng et al. 2020)|Classify text in types of policy measures against COVID-19 (20 classes)|Research assistant texts and copies from news & government sources|One or multiple sentences|
|Manifesto stances towards the military (subsets of Burst et al. 2020).|Identify stance towards the simple topic “military” (3 classes: positive/negative/unrelated).|Party Manifestos|Quasi-sentences|
|Manifesto stances towards protectionism (subsets of Burst et al. 2020).|Identify stance towards the concept “protectionism” (3 classes: positive/negative/unrelated).|Party Manifestos|Quasi-sentences|
|Manifesto stances towards traditional morality (subsets of Burst et al. 2020).|Identify stance towards the complex concept “traditional morality” (3 classes: positive/negative/unrelated).|Party Manifestos|Quasi-sentences|


### Software Requirements

- Python 3.9
  - the file "`requirements.txt`" lists all dependencies. You can run "`pip install -r requirements.txt`" to install these dependencies.
- This code was written on MacOS and is compatible with Linux. Portions of the code use bash scripts. 
The code was not tested on Windows.


### Controlled Randomness

Randomness is controlled with the random seed 42. When multiple random seeds are necessary to control for randomness, a random number generator 
initialised with the value 42 is used to create new random seeds. 


###  License for Code

MIT


### Acknowledgements

This README file is partly based on the [Social Science Data Editors template](https://social-science-data-editors.github.io/template_README/template-README.html). 