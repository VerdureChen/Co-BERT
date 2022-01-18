import random
import os
from tqdm import tqdm
import linecache
import datetime
import math
import argparse
import logging
import sys
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()
random.seed(42)

def get_token_dict(token_file):
    pre_tokens = {}
    with open(token_file, 'r', encoding='utf-8') as pre:
        num_examples = int(len(linecache.getlines(token_file)))
        logger.info(f'number of token-examples: {str(num_examples)}')
        for i, line in enumerate(tqdm(pre, total=num_examples, desc="Pre-infer examples")):
            tokens = line.strip().split(',')
            qid = tokens[0]
            did = tokens[1]
            bias = tokens[2]
            pre_tokens.setdefault(qid, {})
            pre_tokens[qid][did + '_' + bias] = line
    return pre_tokens

def get_top1_dict(top1_file, pre_token):
    top1dict = {}
    total_list = []
    with open(top1_file, 'r', encoding= 'utf-8') as top1f:
        num_examples = int(len(linecache.getlines(top1_file)))
        logger.info(f'number of token-examples: {str(num_examples)}')
        for i, line in enumerate(tqdm(top1f, total=num_examples, desc="top1 examples")):
            qid, _, did, _, pscore, _ = line.strip().split(' ')
            top1dict.setdefault(qid, {})
            docid, bias = did.split('_')
            top1dict[qid][docid] = pre_token[qid][did]
            total_list.append(qid+'_'+did)
    return top1dict, total_list

def get_order_dict(top1dict, topfile, rerank_num):
    # topfile = r'/data/chenxiaoyang/MS/data/origin/ClueWeb09B/clueweb_DPH_KL_d_5_t_20_0.res'
    order_dict = {}
    qid_list = []
    with open(topfile, 'r', encoding='utf-8') as topf:
        num_examples = int(len(linecache.getlines(topfile)))
        logger.info(f'number of token-examples: {str(num_examples)}')
        for i, line in enumerate(tqdm(topf, total=num_examples, desc="preranking examples")):
            qid, _, docid, _, pscore, _ = line.strip().split(' ')
            # tokens = line.rstrip().split(',')
            # qid = tokens[0]
            # docid = tokens[1]
            if qid in top1dict:
                if qid not in qid_list:
                    qid_list.append(qid)
                order_dict.setdefault(qid, [])
                if docid in top1dict[qid]:
                    if len(order_dict[qid]) > rerank_num:
                        continue
                    order_dict[qid].append(docid)
            if i==100:
                logger.info(str(order_dict))
    logger.info(f'total qid:{str(len(qid_list))}')
    return order_dict, qid_list

def get_totaltop_dict(total_top_file, pre_token, total_num):
    total_top_dict = {}
    with open(total_top_file, 'r', encoding= 'utf-8') as totaltopf:
        num_examples = int(len(linecache.getlines(total_top_file)))
        logger.info(f'number of token-examples: {str(num_examples)}')
        for i, line in enumerate(tqdm(totaltopf, total=num_examples, desc="top1 examples")):
            qid, _, did, _, pscore, _ = line.strip().split(' ')
            total_top_dict.setdefault(qid, [])
            if len(total_top_dict[qid]) < total_num:
                total_top_dict[qid].append(pre_token[qid][did])
            else:
                continue
        logger.info(f'totaltop_dict query:{len(total_top_dict)}')
    return total_top_dict

def get_padding_text():
    input_id = ' '.join([str(item) for item in [0] * 256])
    attn = ' '.join([str(item) for item in [0] * 256])
    seg = ' '.join([str(item) for item in [0] * 256])
    top_id = '0'
    pos_docid = '0'
    pos_bias = '0'
    qrel_score = '0'
    padding_text = top_id + "," + pos_docid + "," + pos_bias + "," + input_id + "," + attn + "," + seg + "," + qrel_score + "\n"
    return padding_text

def get_cobert_data(outfile, order_dict, total_top_dict, top1dict, top_num, overlap, seq_num):
    # outfile = r'/data/chenxiaoyang/MS/data/origin/data/ms_data/dev/tokens/order_tokens.csv'
    p = os.path.dirname(outfile)
    if not os.path.exists(p):
        os.makedirs(p)
    qid_list_now = []
    same_dict = {}
    same_dict_collection = {}
    top_num = top_num
    overlap = overlap
    seq_num = seq_num
    padding_text = get_padding_text()

    for p,qid in enumerate(order_dict):
        same_dict[qid] = order_dict[qid]
        total_doc_per_q = len(same_dict[qid])
        batch_num = math.ceil((total_doc_per_q-overlap)/(seq_num-overlap))
        same_dict_collection.setdefault(qid, [])
        start = 0
        end = start + seq_num
        for i in range(int(batch_num)):
            if p == 0:
                logger.info(f'start:{str(start)}, end:{str(end)}')
            if len(same_dict[qid]) > end:
                same_dict_collection[qid].append(same_dict[qid][start:end])

            else:
                same_dict_collection[qid].append(same_dict[qid][start: len(same_dict[qid])])
            start = end - overlap
            end = start + seq_num
            qid_list_now.append(qid)
        random.shuffle(same_dict_collection[qid])

    random.shuffle(qid_list_now)

    with open(outfile, 'w', encoding='utf-8') as outf:
        for qid in qid_list_now:
            same_list = same_dict_collection[qid][0]
            del same_dict_collection[qid][0]

            for i in range(top_num):
                outf.write(total_top_dict[qid][i])
            if len(total_top_dict[qid]) < top_num:
                pad = top_num - len(total_top_dict[qid])
                logger.info(f'top qid:{qid}, paddind:{pad}')
                for t in range(pad):
                    outf.write(padding_text)
            for i in range(len(same_list)):
                did = same_list[i]
                outf.write(top1dict[qid][did])
            padding_num = seq_num - len(same_list)
            if padding_num !=0:
                logger.info(f'qid:{qid}, paddind:{padding_num}')
                for t in range(padding_num):
                    outf.write(padding_text)
    num_examples = int(len(linecache.getlines(outfile)))
    return num_examples

def generate_train_val_data(fold_num, task_name, infile,  outpath, outname, qid_list):
    split_qid_files = [f'corpus/{task_name}/cv/{fold_num}/train_query',
                       f'corpus/{task_name}/cv/{fold_num}/dev_query',
                       f'corpus/{task_name}/cv/{fold_num}/test_query']
    outfile_train = outpath + f'/{str(fold_num)}/tokens/train/{outname}'
    outfile_val = outpath + f'/{str(fold_num)}/tokens/dev/{outname}'
    outfile_test = outpath + f'/{str(fold_num)}/tokens/test/{outname}'
    for outfile in [outfile_train, outfile_val, outfile_test]:
        p = os.path.dirname(outfile)
        if not os.path.exists(p):
            os.makedirs(p)

    train_qids = []
    val_qids = []
    test_qids = []
    for num, split_qid_file in enumerate(split_qid_files):
        with open(split_qid_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='loading split query', leave=False):
                qid = line.strip().split()[0]
                if num == 0:
                    if qid not in train_qids:
                        train_qids.append(qid)
                    else:
                        logger.info('Repeated train qid!')
                        raise NotImplementedError
                elif num == 1:
                    if qid not in val_qids:
                        val_qids.append(qid)
                    else:
                        logger.info('Repeated dev qid!')
                        raise NotImplementedError
                elif num == 2:
                    if qid not in test_qids:
                        test_qids.append(qid)
                    else:
                        logger.info('Repeated test qid!')
                        raise NotImplementedError
                else:
                    raise NotImplementedError
    with open(infile, 'r', encoding='utf-8') as inf,\
         open(outfile_train, 'w', encoding='utf-8') as out_train,\
         open(outfile_val, 'w', encoding='utf-8') as out_val, \
         open(outfile_test, 'w', encoding='utf-8') as out_test:
        num_examples = int(len(linecache.getlines(infile)))
        logger.info(f'number of token-examples: {str(num_examples)}')
        train_count = 0
        train_qid=[]
        val_count = 0
        val_qid=[]
        test_count = 0
        test_qid = []
        last_id = ''
        for i, line in enumerate(tqdm(inf, total=num_examples, desc="Tokenize examples")):
            tokens = line.strip().split(',')
            query_id = tokens[0]
            if query_id == '0':
                query_id = last_id
            if query_id in train_qids:
                if query_id not in train_qid:
                    train_qid.append(query_id)
                out_train.write(line)
                train_count += 1
            elif query_id in val_qids:
                if query_id not in val_qid:
                    val_qid.append(query_id)
                out_val.write(line)
                val_count += 1
            elif query_id in test_qids:
                if query_id not in test_qid:
                    test_qid.append(query_id)
                out_test.write(line)
                test_count += 1
            else:
                raise NotImplementedError
            if query_id != '0':
                last_id = query_id
        logger.info(f'train:{str(len(set(train_qid)))}')
        logger.info(set(train_qid))
        logger.info(f'val:{str(len(set(val_qid)))}')
        logger.info(set(val_qid))
        logger.info(f'test:{str(len(set(test_qid)))}')
        logger.info(set(test_qid))
        logger.info(f'fold {str(fold_num)},total train:{str(train_count)}, total val:{str(val_count)}, total test:{str(test_count)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_file",
                        default=None,
                        type=str,
                        required=True,
                        help="token file path")
    parser.add_argument("--top1_res",
                        default=None,
                        type=str,
                        required=True,
                        help="top1 result path")
    parser.add_argument("--total_top",
                        default=None,
                        type=str,
                        required=True,
                        help="total top file path")
    parser.add_argument("--first_stage_res",
                        default=None,
                        type=str,
                        required=True,
                        help="DPH+KL res path")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="rob, gov or clue")
    parser.add_argument("--cobert_tokens",
                        default=None,
                        type=str,
                        required=True,
                        help="five folds tokens in one file")
    parser.add_argument("--outpath",
                        default=None,
                        type=str,
                        required=True,
                        help="folds tokens out path")
    parser.add_argument("--outname",
                        default=None,
                        type=str,
                        required=True,
                        help="folds token file name")
    parser.add_argument("--overlap_num",
                        default=4,
                        type=int,
                        help="overlap doc num")
    parser.add_argument("--seq_num",
                        default=60,
                        type=int,
                        help="batch_size - prf_num")
    parser.add_argument("--prf_num",
                        default=4,
                        type=int,
                        help="prf doc num")
    parser.add_argument("--rerank_num",
                        default=1000,
                        type=int,
                        help="rerank doc num")
    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    pre_tokens = get_token_dict(args.token_file)
    top1dict, total_list = get_top1_dict(args.top1_res, pre_tokens)
    order_dict, qid_list = get_order_dict(top1dict, args.first_stage_res, args.rerank_num)
    total_top_dict = get_totaltop_dict(args.total_top, pre_tokens, args.prf_num)
    # total_token_line_pre_fold = get_cobert_data(args.cobert_tokens, order_dict, total_top_dict, top1dict, args.prf_num,
    #                                             args.overlap_num, args.seq_num)
    # logger.info(f'five fold token: {str(total_token_line_pre_fold)}')
    for fold in range(1,6):
        total_token_line_pre_fold = get_cobert_data(args.cobert_tokens, order_dict, total_top_dict, top1dict,
                                                    args.prf_num, args.overlap_num, args.seq_num)
        generate_train_val_data(fold, args.task_name, args.cobert_tokens, args.outpath, args.outname, qid_list)