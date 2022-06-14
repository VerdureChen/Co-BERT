# Co-BERT 
This repository contains the code and resources for our paper:
- [Incorporating Ranking Context for End-to-End BERT Re-ranking](https://drive.google.com/file/d/1R50ctHlXYcz3z-6nD1PTPyZWbAQXx9Gs/view?usp=sharing). In *ECIR 2022*.

Co-BERT is a groupwise BERT-based ranking model,  wherein
the query-document interaction representation and the ranking context are learned
jointly. 

## Requirements
You can install required packages by running:
```
pip install -r requirements.txt
```
## Getting started
In this repo, we provide instructions on how to run Co-BERT on Robust04, GOV2 and ClueWeb09-B datasets. \
For the first step of the data pre-processing, We recommend you refer to [BERT-QE](https://github.com/zh-zheng/BERT-QE#resources).
It gives a detailed data preparation and pre-processing procedure for Robust04 and Gov2, 
and you can easily adapt it to get the `collection.txt` of ClueWeb09-B. 

After getting the `collection.txt` files, you need to add some files to `corpus` under the project directory. The directory structure is as following:
```
├── corpus
│   ├── clue
│   │   ├── collection.txt
│   │   ├── cv
│   │   ├── DPH_KL_title.res
│   │   ├── qrel
│   │   └── title_queries.tsv
│   ├── gov
│   │   ├── collection.txt
│   │   ├── cv
│   │   ├── DPH_KL_title.res
│   │   ├── qrel
│   │   └── title_queries.tsv
│   └── rob
│       ├── collection.txt
│       ├── cv
│       ├── DPH_KL_title.res
│       ├── qrel
│       └── title_queries.tsv
```
If you don't want to run our model on all three datasets, just create the folder which you need.

Here is the description of the items in the `corpus` directory:

 - **clue/gov/rob**: The folders named abbreviated version of the dataset name.
 - **collection.txt**: The file generated according to our previous instructions.
 - **cv**: The five-fold division of the queries.
 - **DPH_KL_title.res**: The run file of DPH+KL using the title queries.
 - **qrel**: The qrel file of the dataset.
 - **title_queries.tsv**: The title queries of the dataset.
 
 Then, config your PLM path in the `data_process.sh` , and then run it to complete the data pre-procession:
 ```
bert_base_ms_model=Path of Your Pre-trained Language Model.
```
 


