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

After getting the `collection.txt` files, you need to add some files to `corpus` under the project directory. The directory structure is as follows:
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

The data pre-procession step will create a directory named `data` under your project path. And the `cobert.csv` in the sub-directory of each fold will by used in the next steps.
 

## Model Training
You can train CO-BERT easily by running `cobert_train.sh`.
Before the model training, you should first settle on the parameters of the model:

- **dataset_name**: clue/gov/rob
- **DATA_NAME**: cobert.csv
- **MODEL_NAME**: Your custom model name, e.g. "cobert-220615".
- **MODEL_TYPE**: Default to "cobert". You can change it to "prf_only" or "group_only" to try the ablation versions of our model.
- **total_output_path**: The output path of the model checkpoints. This path is dataset agnostic. 
- **total_process_data_path**: Default to "./data". This path is dataset agnostic.
- **ENCODER_MODEL**: Path of your encoding PLM.
- **GROUPWISE_MODEL**: Path of your groupwise scorer PLM.
- **PRF_MODEL**: Path of your PRF calibrator PLM.

## Model Validation
After training COBERT, the project default to save five checkpoints for each fold. By running `cobert_dev.sh`, you can get the results of different checkpoints on the validation sets.
The parameters are as follows:
- **dataset_name**: clue/gov/rob
- **DATA_NAME**: cobert.csv
- **CKPT_NAME**: Your custom model name, e.g. "cobert-220615".
- **OUTDIR_NAME**: Your validation result directory name, it can be the same as _CKPT_NAME_, e.g. "cobert-0114".
- **MODEL_TYPE**: Default to "cobert", this parameter need to be consistent with the _MODEL_TYPE_ of the model training.
- **ENCODER_MODEL**: Path of your encoding PLM, this parameter need to be consistent with the _ENCODER_MODEL_ of the model training.
- **GROUPWISE_MODEL**: Path of your groupwise scorer PLM, this parameter need to be consistent with the _GROUPWISE_MODEL_ of the model training.
- **PRF_MODEL**: Path of your PRF calibrator PLM, this parameter need to be consistent with the _PRF_MODEL_ of the model training.
- **total_ckpt_dir**: Default to "./output". The path of model checkpoints, this parameter need to be consistent with the _total_output_path_ of the model training.
- **total_data_dir**: Default to "./data". This path is dataset agnostic.
- **total_corpus_path**: Default to "./corpus".

## Model Test
When testing the effectiveness of COBERT, `cobert_test.sh` first selects the best checkpoint of each fold, and then test it on the test sets. It finally outputs the run file and gives the metrics in the `test_score/record.txt` 
which is located in the output directory of the corresponding dataset.

You need to confirm these parameters:
- **dataset_name**: clue/gov/rob
- **DATA_NAME**: cobert.csv
- **CKPT_NAME**: Your custom model name, e.g. "cobert-220615".
- **MODEL_TYPE**: Default to "cobert", this parameter need to be consistent with the _MODEL_TYPE_ of the model training.
- **ENCODER_MODEL**: Path of your encoding PLM, this parameter need to be consistent with the _ENCODER_MODEL_ of the model training.
- **GROUPWISE_MODEL**: Path of your groupwise scorer PLM, this parameter need to be consistent with the _GROUPWISE_MODEL_ of the model training.
- **PRF_MODEL**: Path of your PRF calibrator PLM, this parameter need to be consistent with the _PRF_MODEL_ of the model training.
- **total_ckpt_dir**: Default to "./output". The path of model checkpoints, this parameter need to be consistent with the _total_output_path_ of the model training.
- **total_data_dir**: Default to "./data". This path is dataset agnostic.
- **total_corpus_path**: Default to "./corpus".

## Citation
If you use our code or resources, please cite this paper:
```
@inproceedings{DBLP:conf/ecir/ChenHHHSY22,
  author    = {Xiaoyang Chen and
               Kai Hui and
               Ben He and
               Xianpei Han and
               Le Sun and
               Zheng Ye},
  editor    = {Matthias Hagen and
               Suzan Verberne and
               Craig Macdonald and
               Christin Seifert and
               Krisztian Balog and
               Kjetil N{\o}rv{\aa}g and
               Vinay Setty},
  title     = {Incorporating Ranking Context for End-to-End {BERT} Re-ranking},
  booktitle = {Advances in Information Retrieval - 44th European Conference on {IR}
               Research, {ECIR} 2022, Stavanger, Norway, April 10-14, 2022, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13185},
  pages     = {111--127},
  publisher = {Springer},
  year      = {2022},
  url       = {https://doi.org/10.1007/978-3-030-99736-6\_8},
  doi       = {10.1007/978-3-030-99736-6\_8},
  timestamp = {Thu, 07 Apr 2022 18:19:50 +0200},
  biburl    = {https://dblp.org/rec/conf/ecir/ChenHHHSY22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```