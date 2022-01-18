import os
import logging
import argparse
import linecache
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from transformer.modeling import BertForSequenceClassification
from data_processing import get_labels, get_rank_task_dataloader
from ms_marco_eval import compute_metrics_from_files
import datetime
import subprocess
from get_trec_metrics import validate
from models import (cobert, cobert_group_only, cobert_prf_only,
                    cobert_no_resi, cobert_freeze2embedding, cobert_freeze)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def do_eval(model, eval_dataloader, device, args):
    scores = []
    pooled_outputs = []
    model.eval()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        # batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            if args.model_name == 'freeze':
                input_ids, input_mask, segment_ids, label_ids, query_ids, doc_ids, features= batch_
                features = features.to(device)
            else:
                input_ids, input_mask, segment_ids, label_ids, query_ids, doc_ids = batch_
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            query_ids = query_ids.cpu().numpy()
            # doc_ids = doc_ids.cpu()
            if args.do_preinfer:
                logits = model(input_ids, segment_ids, input_mask)[0]
            else:
                if args.model_name == 'freeze_embedding':
                    logits, pooled_output = model(input_ids, segment_ids, input_mask, args=args)
                    pooled_outputs.extend(pooled_output.detach().cpu())
                elif args.model_name == 'freeze':
                    logits = model(features, input_ids, segment_ids, input_mask, args=args)

                else:
                    logits = model(input_ids, segment_ids, input_mask, args=args)

            probs = F.softmax(logits, dim=1)[:, 1]
            scores.append(probs.detach().cpu().numpy())
    result = {}
    result['scores'] = np.concatenate(scores)
    if args.model_name == 'freeze_embedding':
        result['pooled_output'] = pooled_outputs

    return result


def save_probs(scores, data_file,  output_dir):

    query_psgs_ids = []
    count = 0
    with open(data_file, mode='r') as ref_file:
        for line in ref_file:
            count += 1
            tokens = line.strip().split(",")
            qid = tokens[0]
            psg_id = tokens[1]
            pos_bias = tokens[2]
            query_psgs_ids.append([qid, psg_id, pos_bias])

    print(len(scores))
    print(len(query_psgs_ids))
    assert len(scores) == len(query_psgs_ids)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictions_file_path = os.path.join(output_dir, 'total_res.txt')
    predictions_top1_path = os.path.join(output_dir, 'top1_res.txt')
    #predictions_file = open(predictions_file_path, mode='w')

    rerank_run = {}
    for idx in range(len(scores)):
         qid = query_psgs_ids[idx][0]             
         psg_id = query_psgs_ids[idx][1]
         psg_bias = query_psgs_ids[idx][2]
         did = psg_id+'-'+psg_bias

         
         score = scores[idx]
         # rerank_run.setdefault(qid, {})[did] = float(score)
         q_dict = rerank_run.setdefault(qid, {})
         q_dict.setdefault(psg_id,{})[psg_bias] = float(score)

         #predictions_file.write("\t".join((qid, doc_id, psg_id, str(float(score)))) + "\n")
    with open(predictions_file_path, 'wt') as runfile, \
         open(predictions_top1_path, 'wt') as topfile:
        q_count = 0
        psg_count = 0
        qid_count = 0
        for qid in rerank_run:
            if qid == '0':
                continue
            for did in rerank_run[qid]:
                scores = list(sorted(rerank_run[qid][did].items(), key=lambda x: (x[1], x[0]), reverse=True))
                for i, (bias, score) in enumerate(scores):
                    d_name = f'{did}_{bias}'
                    runfile.write(f'{qid} 0 {d_name} {i + 1} {score} run\n')
                    psg_count = psg_count + 1
                    if i == 0:
                        topfile.write(f'{qid} 0 {d_name} {i + 1} {score} run\n')
                q_count = q_count + 1
            qid_count = qid_count + 1
        print('total query number:{},total doc number:{}, total passage number:{}'.format(
            str(qid_count), str(q_count), str(psg_count)))


def save_probs_vt(args, scores, data_file, output_dir, model_dir):
    query_psgs_ids = []
    count = 0
    with open(data_file, mode='r') as ref_file:
        for line in ref_file:
            if args.model_name == 'freeze_embedding':
                pass
            else:
                bsz = int(args.eval_batch_size)
                top_num = int(args.top_num)
                if count % bsz < top_num:
                    count += 1
                    continue
            count += 1
            tokens = line.strip().split(",")
            qid = tokens[0]
            psg_id = tokens[1]
            pos_bias = tokens[2]
            query_psgs_ids.append([qid, psg_id, pos_bias])

    print(len(scores))
    print(len(query_psgs_ids))
    assert len(scores) == len(query_psgs_ids)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictions_file_path = os.path.join(output_dir, 'results_{}_{}.txt'.format(
        model_dir,datetime.date.today().strftime('%Y_%m_%d')))
    rerank_run = {}
    for idx in range(len(scores)):
        qid = query_psgs_ids[idx][0]
        psg_id = query_psgs_ids[idx][1]
        psg_bias = query_psgs_ids[idx][2]
        did = psg_id + '-' + psg_bias

        score = scores[idx]
        q_dict = rerank_run.setdefault(qid, {})
        if psg_id in q_dict.keys():
            if float(score) > q_dict[psg_id]:
                q_dict[psg_id] = float(score)
        else:
            q_dict[psg_id] = float(score)

    with open(predictions_file_path, 'wt') as runfile:
        q_count = 0
        psg_count = 0
        for qid in rerank_run:
            if qid == '0':
                continue
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i + 1} {score} run\n')
                psg_count = psg_count + 1
            q_count = q_count + 1
        print('total topic number:{}, total passage number:{}'.format(str(q_count), str(psg_count)))


def save_pooled_outputs(pooled_output, output_dir, model_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictions_file_path = os.path.join(output_dir, 'pooled_output_{}_{}.pt'.format(model_dir,datetime.date.today().strftime('%Y_%m_%d')))
    torch.save(pooled_output, predictions_file_path)
    print('finish save pooled_output')


def get_metrics(result_path, ref_file, output_path, trec_eval):
    dev_record_file = os.path.join(output_path, 'record_{}.txt'.format(datetime.date.today().strftime('%Y_%m_%d')))
    max_score=-1
    max_outfile = ''
    total_metrics = {}
    with open(dev_record_file, 'w', encoding='utf-8') as dev_record:
        dir_list = [item for item in os.listdir(result_path) if item.startswith('results')]

        def num(ele):
            return int(ele.split('-')[-1].split('_')[0])
        try:
            dir_list.sort(key=num, reverse=True)
        except:
            pass
        logger.info('*******')
        logger.info(dir_list)
        logger.info('*******')
        if len(dir_list)==1:
            res_file = os.path.join(result_path, dir_list[0])
            path_to_candidate = res_file
            path_to_reference = ref_file
            metrics = validate(path_to_reference, path_to_candidate, trec_eval)
            dev_record.write('##########{}###########\n'.format(dir_list[0]))
            for metric in sorted(metrics):
                dev_record.write('{}: {}\n'.format(metric, metrics[metric]))
            dev_record.write('#####################\n')
            return metrics
        else:
            for i,fil in enumerate(dir_list):
                res_file = os.path.join(result_path, fil)
                path_to_candidate = res_file
                path_to_reference = ref_file
                metrics = validate(path_to_reference, path_to_candidate, trec_eval)
                total_metrics[10-i] = metrics
                p20 = float(metrics['ndcg_cut_20'])
                if p20>max_score:
                    max_score = p20
                    max_outfile = fil
                dev_record.write('##########{}###########\n'.format(fil))
                for metric in sorted(metrics):
                    dev_record.write('{}: {}\n'.format(metric, metrics[metric]))
                dev_record.write('#####################\n')
            dev_record.write('MAX FILE:{}, MAX ndcg_cut_20:{}'.format(max_outfile, str(max_score)))
            return total_metrics


def five_folds_dev_test(args, model, fold_num, task_name, device, ckpt_name):
    if args.do_embedding:
        '''
        to get the bert embedding, the model_name must be set as freeze_embedding 
        and the embedding_task must be set as 'train', 'dev' or 'test'
        '''
        assert args.model_name == 'freeze_embedding'
        if args.embedding_task not in ['train', 'dev', 'test']:
            logger.info('embedding task must be set!')
            raise NotImplementedError
        _, dataloader = get_rank_task_dataloader(fold_num, task_name, args.embedding_task,
                                                 args, SequentialSampler, args.eval_batch_size)

        pred_res = do_eval(model, dataloader, device, args)
        # pred_scores = pred_res['scores']
        # data_file = os.path.join(args.data_dir, str(fold_num), 'tokens', args.embedding_task, args.data_name)
        output_dir = os.path.join(args.output_dir, str(fold_num), args.embedding_task, args.outdir_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # just for verify
        # save_probs_vt(args, pred_scores, data_file, output_dir, d)
        # get_metrics(output_dir, args.ref_file, output_dir)

        pooled_output = pred_res['pooled_output']
        output_dir = os.path.join(args.output_dir, str(fold_num), 'tokens', args.embedding_task,  args.outdir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_pooled_outputs(pooled_output, output_dir, ckpt_name)

    if args.do_preinfer:
        _, dataloader = get_rank_task_dataloader(fold_num, task_name, 'pre', args, SequentialSampler,
                                                 args.eval_batch_size)

        pred_res = do_eval(model, dataloader, device, args)
        pred_scores = pred_res['scores']

        data_file = os.path.join(args.data_dir, args.data_name)

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_probs(pred_scores, data_file, output_dir)

    if args.do_dev:
        _, dataloader = get_rank_task_dataloader(fold_num, task_name, 'dev', args, SequentialSampler, args.eval_batch_size)

        pred_res = do_eval(model, dataloader, device, args)
        pred_scores = pred_res['scores']

        data_file = os.path.join(args.data_dir, str(fold_num), 'tokens/dev/', args.data_name)
        output_dir = os.path.join(args.output_dir, str(fold_num), 'dev/', args.outdir_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_probs_vt(args, pred_scores, data_file, output_dir, ckpt_name)
        get_metrics(output_dir, args.ref_file, output_dir, args.trec_eval)

    if args.do_test:
        _, dataloader = get_rank_task_dataloader(fold_num, task_name, 'test', args, SequentialSampler, args.eval_batch_size)

        pred_res = do_eval(model, dataloader, device, args)
        pred_scores = pred_res['scores']
        if args.task_name=='msmarco':
            data_file = os.path.join(args.data_dir, 'tokens/', args.data_name)
        else:
            data_file = os.path.join(args.data_dir, str(fold_num), 'tokens/test/', args.data_name)
        output_dir = os.path.join(args.output_dir, str(fold_num), 'test/', args.outdir_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_probs_vt(args, pred_scores, data_file, output_dir, ckpt_name)
        get_metrics(output_dir, args.ref_file, output_dir, args.trec_eval)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The model for inference.")
    parser.add_argument("--fold",
                        default=None,
                        type=str,
                        help="foldnum")
    parser.add_argument("--run_name",
                        default=None,
                        type=str,
                        help="The name of modeldir.")
    parser.add_argument("--data_dir",
                        default=None,   
                        type=str, 
                        required=True,   
                        help="data dir")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="small or base")
    parser.add_argument("--encoder_model",
                        default=None,
                        type=str,
                        help="The qt model dir.")
    parser.add_argument("--groupwise_model",
                        default=None,
                        type=str,
                        help="The qb model dir.")
    parser.add_argument("--prf_model",
                        default=None,
                        type=str,
                        help="The attn model dir.")
    parser.add_argument("--top_num",
                        default=4,
                        type=int,
                        help="Total top doc number.")
    parser.add_argument("--overlap",
                        default=4,
                        type=int,
                        help="Total overlap number.")
    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=True,
                        help="data name")
    parser.add_argument("--do_preinfer",
                        action='store_true',
                        help="Whether to run pre-infer.")
    parser.add_argument("--do_embedding",
                        action='store_true',
                        help="Whether to save embedding.")
    parser.add_argument("--embedding_task",
                        default='unset',
                        type=str,
                        help="train, dev or test to embedding.")
    parser.add_argument("--do_dev",
                        action='store_true',
                        help="Whether to run dev.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Batch size for eval.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task, rob/gov/clue/msmarco.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--outdir_name",
                        default=None,
                        type=str,
                        help="output dir name")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--ref_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The qrel file path.")
    parser.add_argument("--trec_eval",
                        default=None,
                        type=str,
                        required=True,
                        help="trec_eval path.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    task_name = args.task_name
    # do_dev = args.do_dev
    # do_test = args.do_test

    label_list = get_labels(args.task_name.lower())
    num_labels = len(label_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_preinfer:
        model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=num_labels)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        five_folds_dev_test(args, model, 'none', task_name, device, 'none')
    else:
        folds = [int(args.fold)]
        for fold in folds:
            dir_list = [item for item in
                        os.listdir(os.path.join(args.model_dir, str(fold), 'train', args.run_name))
                        if item.startswith('checkpoint')]

            def num(ele):
                return int(ele.split('-')[-1])

            dir_list.sort(key=num,reverse=True)
            # dir_list.sort(key=num)
            logger.info('*******')
            logger.info(dir_list)
            logger.info('*******')
            for ckpt_name in dir_list:
                model_dir = os.path.join(args.model_dir, str(fold), 'train', args.run_name, ckpt_name)
                logger.info('*******')
                logger.info(model_dir)
                logger.info('*******')
                prf_model = args.prf_model
                encoder_model = args.encoder_model
                groupwise_model = args.groupwise_model
                if args.model_name == 'cobert':
                    model = cobert(encoder_model, groupwise_model, prf_model, num_labels)
                    logger.info('model_name: cobert')
                elif args.model_name == 'no_resi':
                    model = cobert_no_resi(encoder_model, groupwise_model, prf_model, num_labels)
                    logger.info('model_name: no_resi')
                elif args.model_name == 'freeze':
                    model = cobert_freeze(encoder_model, groupwise_model, prf_model, num_labels, fold=fold)
                    logger.info('model_name: freeze')
                elif args.model_name == 'freeze_embedding':
                    model = cobert_freeze2embedding(encoder_model, groupwise_model, prf_model, num_labels, fold=fold)
                    logger.info('model_name: freeze_embedding')
                elif args.model_name == 'group_only':
                    model = cobert_group_only(encoder_model, groupwise_model, prf_model, num_labels)
                    logger.info('model_name: group_only')
                elif args.model_name == 'prf_only':
                    model = cobert_prf_only(encoder_model, groupwise_model, prf_model, num_labels)
                    logger.info('model_name: prf_only')
                elif args.model_name == 'bert_base':
                    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
                    logger.info('model_name: bert_base')
                if args.model_name != 'freeze_embedding' and args.model_name != 'bert_base':
                    model.load_state_dict(torch.load(os.path.join(model_dir, 'weights.pt')))
                model.to(device)
                # if n_gpu > 1:
                #     model = torch.nn.DataParallel(model)
                five_folds_dev_test(args, model, fold, task_name, device, ckpt_name)
                if args.model_name == 'freeze_embedding':
                    break



if __name__ == "__main__":
    main()

