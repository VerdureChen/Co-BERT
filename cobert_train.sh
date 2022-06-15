#!/bin/sh
#####first time install apex for FP16#####
#cd apex
#pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#cd -
#################################
dataset_name='clue'
DATA_NAME=cobert.csv
MODEL_NAME="cobert-0118"
MODEL_TYPE="cobert"
total_output_path=/data/chenxiaoyang/MS/data/cobert_github/output/
total_process_data_path=/data/chenxiaoyang/MS/data/cobert_github/data

ENCODER_MODEL=/data/chenxiaoyang/MS/BERT_Base_trained_on_MSMARCO
GROUPWISE_MODEL=/data/chenxiaoyang/MS/2-512/uncased_L-4_H-768_A-12
PRF_MODEL=/data/chenxiaoyang/MS/2-512/uncased_L-2_H-768_A-12

OUTPUT_DIR=$total_output_path/$dataset_name
DATA_DIR=$total_process_data_path/"$dataset_name"_data


CUDA_VISIBLE_DEVICES=4 python code/bert_finetune.py --data_dir $DATA_DIR \
                                                 --data_name $DATA_NAME \
                                                 --model_name $MODEL_TYPE \
                                                 --groupwise_model $GROUPWISE_MODEL \
                                                 --encoder_model $ENCODER_MODEL \
                                                 --prf_model $PRF_MODEL \
                                                 --task_name $dataset_name \
                                                 --fold 5 \
                                                 --output_dir $OUTPUT_DIR \
                                                 --outdir_name $MODEL_NAME \
                                                 --max_seq_length 256 \
                                                 --top_num 4 \
                                                 --overlap 4 \
                                                 --fp16 \
                                                 --train_batch_size 64 \
                                                 --learning_rate 3e-06 \
                                                 --total_epoch 5.0\
                                                 --seed 42 \
                                                 --eval_step 50 \
                                                 --save_step 50000

wait
echo "finish train cobert!"
