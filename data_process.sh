#!/bin/sh

dataset_name='clue'  #rob, gov, clue
text_length=150
total_corpus_path=./corpus
total_process_data_path=./data
bert_base_ms_model=/data/chenxiaoyang/MS/BERT_Base_trained_on_MSMARCO
trec_eval=./code/bin/trec_eval
token_data_name=tokens.csv
fold_data_name=cobert.csv

raw_text_split_path=$total_process_data_path/"$dataset_name"_data
input_raw_text=$total_process_data_path/"$dataset_name"_data/pre_split_raw.csv
output_token_path=$total_process_data_path/"$dataset_name"_data
data_dir=$total_process_data_path/"$dataset_name"_data
top_outpath=$total_process_data_path/"$dataset_name"_data
qrel_file=$total_corpus_path/"$dataset_name"/qrel
dph_kl=$total_corpus_path/"$dataset_name"/DPH_KL_title.res

#Split data
python code/data_process/text_split.py --output_path $raw_text_split_path \
                                          --corpus_path $total_corpus_path/"$dataset_name" \
                                          --datasets $dataset_name \
                                          --text_split_length $text_length

wait

python code/data_process/text_to_tokens.py --input_path $input_raw_text \
                                              --output_path $output_token_path \
                                              --datasets_name $dataset_name
wait

CUDA_VISIBLE_DEVICES=3,4,5 python code/inference.py  --model_dir $bert_base_ms_model \
                               --do_preinfer \
                               --task_name gov \
                               --data_dir $data_dir \
                               --data_name $token_data_name \
                               --output_dir $top_outpath \
                               --ref_file $qrel_file \
                               --eval_batch_size 320 \
                               --trec_eval $trec_eval
wait

python code/select_fold_data.py --token_file $data_dir/$token_data_name \
                                   --top1_res $top_outpath/top1_res.txt \
                                   --total_top $top_outpath/total_res.txt \
                                   --first_stage_res $dph_kl \
                                   --task_name $dataset_name \
                                   --cobert_tokens $data_dir/batched_tokens.csv \
                                   --outpath $data_dir \
                                   --outname $fold_data_name


