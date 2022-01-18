'''
将triple中的正例转换为tokens，使用逗号分隔topic_id，doc_id，passge_id，token_id，input_mask，seg_id
'''

from transformers import BertTokenizerFast
import os
import argparse
import csv
import sys
from tqdm import tqdm
import datetime
import linecache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


def pos_tokenize(inpath, outpath):
    '''
    为筛选数据之前的inference进行tokenize
    :return:
    '''
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    triple_path = inpath
    output_path = outpath
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, 'tokens.csv')
    num_examples = int(len(linecache.getlines(triple_path)))
    print('number of examples: ', str(num_examples))
    with open(triple_path, 'r', encoding='utf-8') as f,\
         open(output_path, 'w', encoding='utf-8') as out_f:
        count = 0
        for i, line in enumerate(tqdm(f, total=num_examples, desc="Tokenize examples")):
            # top_id, qtext, pos_docid, pos_title, pos_body, pos_bias, qrel_score = line.rstrip().split('\t')
            top_id, qtext, pos_docid, pos_body, pos_bias, qrel_score = line.rstrip().split('\t')
            sentence_a = qtext
            sentence_b = pos_body
            encoded = tokenizer.encode_plus(
                text=sentence_a,  # the sentence to be encoded
                text_pair=sentence_b,
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length=256,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask=True,  # Generate the attention mask
                return_tensors='pt',  # ask the function to return PyTorch tensors
                truncation=True,
            )
            input_ids = encoded['input_ids']
            attn_mask = encoded['attention_mask']
            seg_mask = encoded['token_type_ids']
            input_id = ' '.join([str(item) for item in input_ids.tolist()[0]])
            attn = ' '.join([str(item) for item in attn_mask.tolist()[0]])
            seg = ' '.join([str(item) for item in seg_mask.tolist()[0]])
            out_f.write(top_id+","+pos_docid+","+pos_bias+","+
                        input_id+","+attn+","+seg+","+qrel_score+"\n")
            count = count + 1
        logger.info('total tokenize number: {}'.format(str(count)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        default=None,
                        type=str,
                        required=True,
                        help="input path")
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True,
                        help="output path")
    parser.add_argument("--datasets_name",
                        default=None,
                        type=str,
                        required=True,
                        help="rob, gov or clue")
    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    pos_tokenize(inpath=args.input_path, outpath=args.output_path)
    logger.info('{}: tokenize finished!'.format(args.datasets_name))