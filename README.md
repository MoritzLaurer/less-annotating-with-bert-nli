## Paper Repository: "Less Annotating, More Classifying"

This is the replication code for the paper "Less Annotating, More Classifying – Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI"

This repository contains the full code and data for reproducing the paper.[^1] An earlier pre-print version of the paper is available [here](https://osf.io/wqc86/) and an improved version is currently under review. The repository will be updated upon final acceptance. 

We also provide an easy-to-use Google Colab notebook for testing BERT-NLI with free access to a GPU. We invite anyone to run and copy this notebook and to train their own BERT-NLI model on their own data: https://colab.research.google.com/drive/1-y7o-QRWp-OwGMe64CxQwQk2-o2jZFm3?usp=sharing 

[^1]: Note that the raw data file for the Manifesto corpus has not been uploaded, as its 400+ MB size exceed the 100 MB data limit by GitHub. 


## Data Availability and Provenance Statements

> INSTRUCTIONS: Every README should contain a description of the origin (provenance), location and accessibility (data availability) of the data used in the article. These descriptions are generally referred to as "Data Availability Statements" (DAS). However, in some cases, there is no external data used.

> INSTRUCTIONS: 
> - When the authors are **secondary data users** (they did not generate the data), the provenance and DAS coincide, and should describe the condition under which (a) the current authors (b) any future users might access the data. 
> - When the data were generated (by the authors) in the course of conducting (lab or field) **experiments**, or were collected as part of **surveys**, then the description of the provenance should describe the data generating process, i.e., survey or experimental procedures:
>   - Experiments: complete sets of experimental instructions, questionnaires, stimuli for all conditions, potentially screenshots, scripts for experimenters or research assistants, as well as for subject eligibility criteria (e.g. selection criteria, exclusions), recruitment waves, demographics of subject pool used. 
>   - For lab experiments specifically, a description of any pilot sessions/studies, and computer programs, configuration files, or scripts used to run the experiment. 
>   - For surveys, the whole questionnaire (code or images/PDF) including survey logic if not linear, interviewer instructions, enumeration lists, sample selection criteria.
>
>  The information should describe ALL data used, regardless of whether they are provided as part of the replication archive or not, and regardless of size or scope. The DAS should provide enough information that a replicator can obtain the data from the original source, even if the file is provided. 
>
> For instance, if using GDP deflators, the source of the deflators (e.g. at the national statistical office) should also be listed here. If any of this information has been provided in a pre-registration, then a link to that registration may (partially) suffice.
>
> DAS can be complex and varied. Examples are provided [here](https://social-science-data-editors.github.io/guidance/Requested_information_dcas.html), and below.
>
> Importantly, if providing the data as part of the replication package, authors should be clear about whether they have the **rights** to distribute the data. Data may be subject to distribution restrictions due to sensitivity, IRB, proprietary clauses in the data use agreement, etc.
>
> NOTE: DAS do not replace Data Citations (see [Guidance](Data_citation_guidance.md)). Rather, they augment them. Depending on journal requirements and to some extent stylistic considerations, data citations should appear in the main article, in an appendix, or in the README. However, data citations only provide information **where** to find the data, not **how to access** those data. Thus, DAS augment data citations by going into additional detail that allow a researcher to assess cost, complexity, and availability over time of the data used by the original author.


### Statement about Rights

- [ ] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript. 
- [ ] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. Appropriate permission are documented in the [LICENSE.txt](LICENSE.txt) file.


### (Optional, but recommended) License for Data

> INSTRUCTIONS: Most data repositories provide for a default license, but do not impose a specific license. Authors should actively select a license. This should be provided in a LICENSE.txt file, separately from the README, possibly combined with the license for any code. Some data may be subject to inherited license requirements, i.e., the data provider may allow for redistribution only if the data is licensed under specific rules - authors should check with their data providers. For instance, a data use license might require that users - the current author, but also any subsequent users - cite the data provider. Licensing can be complex. Some non-legal guidance may be found [here](https://social-science-data-editors.github.io/guidance/Licensing_guidance.html). For multiple licenses within a data package, the `LICENSE.txt` file might contain the concatenation of all the licenses that apply (for instance, a custom license for one file, plus a CC-BY license for another file).
>
> NOTE: In many cases, it is not up to the creator of the replication package to simply define a license, a license may be *sticky* and be defined by the original data creator.

*Example:* The data are licensed under a Creative Commons/CC-BY-NC license. See LICENSE.txt for details.


### Summary of Availability

- [x] All data **are** publicly available.


### Details on each Data Source

All datasets are provided in .csv format. An overview is provided in the table below.
Details on all datasets and the full references are provided in the manuscript.

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


## Computational requirements

> INSTRUCTIONS: In general, the specific computer code used to generate the results in the article will be within the repository that also contains this README. However, other computational requirements - shared libraries or code packages, required software, specific computing hardware - may be important, and is always useful, for the goal of replication. Some example text follows. 

> INSTRUCTIONS: We strongly suggest providing setup scripts that install/set up the environment. Sample scripts for [Stata](https://github.com/gslab-econ/template/blob/master/config/config_stata.do),  [R](https://github.com/labordynamicsinstitute/paper-template/blob/master/programs/global-libraries.R), [Julia](https://github.com/labordynamicsinstitute/paper-template/blob/master/programs/packages.jl) are easy to set up and implement. Specific software may have more sophisticated tools: [Python](https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability), [Julia](https://julia.quantecon.org/more_julia/tools_editors.html#Package-Environments).


### Software Requirements

> INSTRUCTIONS: List all of the software requirements, up to and including any operating system requirements, for the entire set of code. It is suggested to distribute most dependencies together with the replication package if allowed, in particular if sourced from unversioned code repositories, Github repos, and personal webpages. In all cases, list the version *you* used. 


- Python 3.8
  - the file "`requirements.txt`" lists all dependencies, please run "`pip install -r requirements.txt`" as the first step. See [https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability](https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability) for further instructions on creating and using the "`requirements.txt`" file.

Portions of the code use bash scripting, which may require Linux.


### Controlled Randomness

Randomness is controlled with the random seed 42. 

When multiple random seeds are necessary to control for randomness, a random number generator 
initialised with the value 42 is used to create new random seeds. 


### Memory and Runtime Requirements

> INSTRUCTIONS: Memory and compute-time requirements may also be relevant or even critical. Some example text follows. It may be useful to break this out by Table/Figure/section of processing. For instance, some estimation routines might run for weeks, but data prep and creating figures might only take a few minutes.



#### Summary

Approximate time needed to reproduce the analyses on a standard (CURRENT YEAR) desktop machine:

- [ ] XXX hours


#### Details

The code was last run on a **4-core Intel-based laptop with MacOS version 10.14.4**. 

Portions of the code were last run on a **32-core Intel server with 1024 GB of RAM, 12 TB of fast local storage**. Computation took 734 hours. 

Portions of the code were last run on a **12-node AWS R3 cluster, consuming 20,000 core-hours**.  

> INSTRUCTIONS: Identifiying hardware and OS can be obtained through a variety of ways:
> Some of these details can be found as follows:
>
> - (Windows) by right-clicking on "This PC" in File Explorer and choosing "Properties"
> - (Mac) Apple-menu > "About this Mac"
> - (Linux) see code in [tools/linux-system-info.sh](https://github.com/AEADataEditor/replication-template/blob/master/tools/linux-system-info.sh)`


## Description of programs/code

> INSTRUCTIONS: Give a high-level overview of the program files and their purpose. Remove redundant/ obsolete files from the Replication archive.

- Programs in `programs/01_dataprep` will extract and reformat all datasets referenced above. The file `programs/01_dataprep/main.do` will run them all.
- Programs in `programs/02_analysis` generate all tables and figures in the main body of the article. The program `programs/02_analysis/main.do` will run them all. Each program called from `main.do` identifies the table or figure it creates (e.g., `05_table5.do`).  Output files are called appropriate names (`table5.tex`, `figure12.png`) and should be easy to correlate with the manuscript.
- Programs in `programs/03_appendix` will generate all tables and figures  in the online appendix. The program `programs/03_appendix/main-appendix.do` will run them all. 
- Ado files have been stored in `programs/ado` and the `main.do` files set the ADO directories appropriately. 
- The program `programs/00_setup.do` will populate the `programs/ado` directory with updated ado packages, but for purposes of exact reproduction, this is not needed. The file `programs/00_setup.log` identifies the versions as they were last updated.
- The program `programs/config.do` contains parameters used by all programs, including a random seed. Note that the random seed is set once for each of the two sequences (in `02_analysis` and `03_appendix`). If running in any order other than the one outlined below, your results may differ.

### (Optional, but recommended) License for Code

> INSTRUCTIONS: Most journal repositories provide for a default license, but do not impose a specific license. Authors should actively select a license. This should be provided in a LICENSE.txt file, separately from the README, possibly combined with the license for any data provided. Some code may be subject to inherited license requirements, i.e., the original code author may allow for redistribution only if the code is licensed under specific rules - authors should check with their sources. For instance, some code authors require that their article describing the econometrics of the package be cited. Licensing can be complex. Some non-legal guidance may be found [here](https://social-science-data-editors.github.io/guidance/Licensing_guidance.html).

The code is licensed under a MIT/BSD/GPL [choose one!] license. See [LICENSE.txt](LICENSE.txt) for details.

## Instructions to Replicators

> INSTRUCTIONS: The first two sections ensure that the data and software necessary to conduct the replication have been collected. This section then describes a human-readable instruction to conduct the replication. This may be simple, or may involve many complicated steps. It should be a simple list, no excess prose. Strict linear sequence. If more than 4-5 manual steps, please wrap a main program/Makefile around them, in logical sequences. Examples follow.

- Edit `programs/config.do` to adjust the default path
- Run `programs/00_setup.do` once on a new system to set up the working environment. 
- Download the data files referenced above. Each should be stored in the prepared subdirectories of `data/`, in the format that you download them in. Do not unzip. Scripts are provided in each directory to download the public-use files. Confidential data files requested as part of your FSRDC project will appear in the `/data` folder. No further action is needed on the replicator's part.
- Run `programs/01_main.do` to run all steps in sequence.


## List of tables and programs


> INSTRUCTIONS: Your programs should clearly identify the tables and figures as they appear in the manuscript, by number. Sometimes, this may be obvious, e.g. a program called "`table1.do`" generates a file called `table1.png`. Sometimes, mnemonics are used, and a mapping is necessary. In all circumstances, provide a list of tables and figures, identifying the program (and possibly the line number) where a figure is created.
>
> NOTE: If the public repository is incomplete, because not all data can be provided, as described in the data section, then the list of tables should clearly indicate which tables, figures, and in-text numbers can be reproduced with the public material provided.

The provided code reproduces:

- [ ] All numbers provided in text in the paper
- [ ] All tables and figures in the paper
- [ ] Selected tables and figures in the paper, as explained and justified below.


| Figure/Table #    | Program                  | Line Number | Output file                      | Note                            |
|-------------------|--------------------------|-------------|----------------------------------|---------------------------------|
| Table 1           | 02_analysis/table1.do    |             | summarystats.csv                 ||
| Table 2           | 02_analysis/table2and3.do| 15          | table2.csv                       ||
| Table 3           | 02_analysis/table2and3.do| 145         | table3.csv                       ||
| Figure 1          | n.a. (no data)           |             |                                  | Source: Herodus (2011)          |
| Figure 2          | 02_analysis/fig2.do      |             | figure2.png                      ||
| Figure 3          | 02_analysis/fig3.do      |             | figure-robustness.png            | Requires confidential data      |

## References

> INSTRUCTIONS: As in any scientific manuscript, you should have proper references. For instance, in this sample README, we cited "Ruggles et al, 2019" and "DESE, 2019" in a Data Availability Statement. The reference should thus be listed here, in the style of your journal:

Steven Ruggles, Steven M. Manson, Tracy A. Kugler, David A. Haynes II, David C. Van Riper, and Maryia Bakhtsiyarava. 2018. "IPUMS Terra: Integrated Data on Population and Environment: Version 2 [dataset]." Minneapolis, MN: *Minnesota Population Center, IPUMS*. https://doi.org/10.18128/D090.V2

Department of Elementary and Secondary Education (DESE), 2019. "Student outcomes database [dataset]" *Massachusetts Department of Elementary and Secondary Education (DESE)*. Accessed January 15, 2019.

U.S. Bureau of Economic Analysis (BEA). 2016. “Table 30: "Economic Profile by County, 1969-2016.” (accessed Sept 1, 2017).

Inglehart, R., C. Haerpfer, A. Moreno, C. Welzel, K. Kizilova, J. Diez-Medrano, M. Lagos, P. Norris, E. Ponarin & B. Puranen et al. (eds.). 2014. World Values Survey: Round Six - Country-Pooled Datafile Version: http://www.worldvaluessurvey.org/WVSDocumentationWV6.jsp. Madrid: JD Systems Institute.

---

## Acknowledgements

This README file is based on the [Social Science Data Editors template](https://social-science-data-editors.github.io/template_README/template-README.html). 