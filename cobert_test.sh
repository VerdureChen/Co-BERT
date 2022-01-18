#!/bin/sh
dataset_name='clue'  #rob, gov, clue
DATA_NAME=cobert.csv
CKPT_NAME="cobert-0114"
DEV_NAME="$CKPT_NAME"
OUTDIR_NAME="$CKPT_NAME"
MODEL_TYPE=cobert #cobert, group_only, prf_only, no_resi
ENCODER_MODEL=/data/chenxiaoyang/MS/BERT_Base_trained_on_MSMARCO
GROUPWISE_MODEL=/data/chenxiaoyang/MS/2-512/uncased_L-4_H-768_A-12
PRF_MODEL=/data/chenxiaoyang/MS/2-512/uncased_L-2_H-768_A-12
MODEL_NAME=$CKPT_NAME/best4test
total_ckpt_dir=./output
total_data_dir=./data
total_corpus_path=./corpus
TREC_EVAL=./code/bin/trec_eval
MODEL_DIR=$total_ckpt_dir/$dataset_name
DATA_DIR=$total_data_dir/"$dataset_name"_data
OUTPUT_DIR=$MODEL_DIR
REF_PATH=$total_corpus_path/$dataset_name/qrel


RESULT_PATH=$OUTPUT_DIR/{}/dev/$DEV_NAME
CKPT_PATH=$OUTPUT_DIR/{}/train/$CKPT_NAME
python code/inference_utils.py --result_path $RESULT_PATH \
                           --ckpt_path $CKPT_PATH \
                           --ref_file $REF_PATH \
                           --trec_eval $TREC_EVAL \
                           --best_for_test
wait

for integer in `seq 1 1 5`
do

CUDA_VISIBLE_DEVICES=4 python code/inference.py --model_dir $MODEL_DIR \
                   --run_name $MODEL_NAME \
                   --groupwise_model $GROUPWISE_MODEL \
                   --encoder_model $ENCODER_MODEL \
                   --prf_model $PRF_MODEL \
                   --top_num 4 \
                   --overlap 4 \
                   --fold $integer \
                   --data_dir $DATA_DIR \
                   --data_name $DATA_NAME \
		               --task_name $dataset_name \
		               --model_name $MODEL_TYPE \
		               --do_test\
                   --output_dir $OUTPUT_DIR \
                   --outdir_name $OUTDIR_NAME \
                   --max_seq_length 256 \
		               --eval_batch_size 64 \
		               --ref_file $REF_PATH \
		               --trec_eval $TREC_EVAL

done
wait
echo "finish cobert test!"
wait

RESULT_PATH=$OUTPUT_DIR/{}/test/$OUTDIR_NAME
OUTPUT_PATH=$OUTPUT_DIR/test_score

python code/inference_utils.py --result_path $RESULT_PATH \
                                         --output_path $OUTPUT_PATH \
                                         --ref_file $REF_PATH \
                                         --trec_eval $TREC_EVAL

