import random
from tqdm import tqdm
import torch
import nltk
import sys
from nltk.tokenize import WordPunctTokenizer
import argparse
import os
import logging
sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

def read_query(files):
    queries = {}
    with open(files, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='loading queryfile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) != 3:
                tqdm.write(f'skipping query line: `{line.rstrip()}`')
                continue
            c_type, c_id, c_text = cols
            assert c_type in ('query', 'doc')
            if c_type == 'query':
                queries[c_id] = c_text

        return queries


def read_doc(files, datastes):
    docs = {}
    with open(files, 'r', encoding='utf-8') as f:
        count = 0
        for line in tqdm(f, desc='loading docfile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if datastes=='rob' and len(cols) != 3:
                # tqdm.write(f'skipping doc line: `{line.rstrip()}`')
                count = count+1
                continue
            elif datastes=='gov' or datastes=='clue':
                if len(cols) != 2:
                    tqdm.write(f'skipping doc line: `{line.rstrip()}`')
                    count = count + 1
                    continue
            if len(cols)==3:
                c_id, c_title, c_text = cols
                c_text = c_title + c_text
            elif len(cols)==2:
                c_id, c_text = cols
            docs[c_id] = c_text
        print(f'total skip doc line {count}')

        return docs


def read_qrels_dict(files):
    result = {}
    with open(files, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='loading qrels (by line)', leave=False):
            qid, _, docid, score = line.strip().split(' ')
            result.setdefault(qid, {})[docid] = int(score)
        return result


def read_top_docs(files):
    top = {}
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='load topfile (by line)', leave=False):
                qid, _, docid, _, pscore, _ = line.strip().split(' ')
                top.setdefault(qid, {})[docid] = float(pscore)
    for i in top:
        if len(top[i])!=1000:
            print(f'qid {i}, len {len(top[i])}')
    return top


def split_body(doc_text, text_len):
    # sentences = sen_tokenizer.tokenize(doc_text)
    # passages_part = []
    # len_count = 0
    # sent_now = ''
    # for sentence in sentences:
    #     sen_len = len(nltk.word_tokenize(sentence))
    #     len_count = len_count + sen_len
    #     if len_count > 75:
    #         passages_part.append(sent_now)
    #         sent_now = ''
    #         len_count = 0
    #     sent_now = sent_now + sentence

    overlap = int(text_len/2)
    tokens = doc_text.split(' ')
    sen_len = len(tokens)
    passages_part = []
    len_start = 0
    sent_now = ''
    len_end = overlap
    while len_end < sen_len:
        sent_now = ' '.join(item for item in tokens[len_start: len_end])
        passages_part.append(sent_now)
        sent_now = ''
        len_start = len_end
        if len_end + overlap < sen_len:
            len_end = len_end + overlap
        else:
            len_end = sen_len
    # print(len_start, len_end, sen_len)
    passages_part.append(' '.join(item for item in tokens[len_start: sen_len]))

    part_len = len(passages_part)
    body = []
    if part_len == 0:
        body = ['']
    elif part_len == 1:
        body = passages_part
    else:
        for i in range(1, part_len):
            body.append(passages_part[i - 1] + passages_part[i])
    return body


def get_raw_text(outpath, query_path, doc_path, qrel_path, top_paths, datasets, text_len=150):
    '''

    :param split_qid_file: 划分文件，每行是一个qid
    :param outpath: raw text输出文件路径
    :param query_path: query路径，每行为'query\t qid \t qtext'
    :param doc_path:doc路径，每行为'docid title doctext'
    :param qrel_path:qrel路径
    :param top_path:top1000文件
    :return:
    '''
    queries = read_query(query_path)
    docs = read_doc(doc_path, datasets)
    qrel = read_qrels_dict(qrel_path)
    top = read_top_docs(top_paths)
    count = 0
    doc_count = 0
    q_count =[]
    with open(outpath, 'w', encoding='utf-8') as out:
        for qid in queries:
            if qid not in q_count:
                q_count.append(qid)
            qtext = queries[qid]
            top_now = top[qid]
            for doc in top_now:
                docid = doc
                try:
                    qrel_score = qrel[qid][doc]
                except KeyError:
                    qrel_score = 0
                    count = count + 1
                try:
                    # doc_title = docs[doc][0]
                    doc_text = docs[doc]
                except KeyError:
                    doc_count = doc_count + 1
                    continue
                split_doc_text = split_body(doc_text, text_len)
                pos_bias = 0
                for pos_passage in split_doc_text:
                    out.write(qid + "\t" + qtext + "\t" +
                              docid + "\t" + pos_passage + "\t" +
                              str(pos_bias) + "\t" + str(qrel_score) + "\n")
                    # out.write(qid + "\t" + qtext + "\t" +
                    #           docid + "\t" + doc_title + "\t" + pos_passage + "\t" +
                    #           str(pos_bias) + "\t" + str(qrel_score) + "\n")
                    pos_bias = pos_bias + 1
    print(f'total qid:{len(q_count)}')
    print(f'top 1000 not in qrel:{count}')
    print(f'top 1000 not in doc:{doc_count}')



def gen_pretrain_data(outpath, datasets, corpus_path, text_len):
    outpath = os.path.join(outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    logger.info(outpath)
    outpath = os.path.join(outpath, 'pre_split_raw.csv')
    query_path = os.path.join(corpus_path, 'title_queries.tsv')
    doc_path = os.path.join(corpus_path, 'collection.txt')
    qrel_path = os.path.join(corpus_path, 'qrel')
    top_paths = [os.path.join(corpus_path, 'DPH_KL_title.res')]
    get_raw_text(outpath,query_path,doc_path,qrel_path,top_paths, datasets, text_len)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True,
                        help="output path")
    parser.add_argument("--text_split_length",
                        default=150,
                        type=int,
                        help="the text length to split.")
    parser.add_argument("--corpus_path",
                        default=None,
                        type=str,
                        required=True,
                        help="corpus path")
    parser.add_argument("--datasets",
                        default=None,
                        type=str,
                        required=True,
                        help="rob, gov, clue")
    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    gen_pretrain_data(outpath=args.output_path, datasets=args.datasets,
                              corpus_path=args.corpus_path, text_len=args.text_split_length)
    logger.info(f'finish process data.')







