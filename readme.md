# Co-BERT 
This repository contains the code and resources for our paper:
- Incorporating Ranking Context for End-to-End BERT Re-ranking. In *ECIR 2022*.

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

