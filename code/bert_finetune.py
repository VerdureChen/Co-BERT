"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function
from torch import nn
import argparse
import linecache
import logging
import os
import random
import sys
import numpy as np
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
import math
from transformer.optimization import BertAdam
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from data_processing import (get_labels, output_modes, get_rank_task_dataloader,
                             get_num_examples)
from models import (cobert, cobert_group_only, cobert_prf_only,
                    cobert_no_resi, cobert_freeze2embedding, cobert_freeze)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        for key in result.keys():
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("-----------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=True,
                        help="data name")
    parser.add_argument("--encoder_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The encoder model dir.")
    parser.add_argument("--groupwise_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The groupwise model dir.")
    parser.add_argument("--prf_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The prf model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--outdir_name",
                        default=None,
                        type=str,
                        required=True,
                        help="output dir name")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="model type")
    parser.add_argument("--fold",
                        default=None,
                        type=str,
                        required=True,
                        help="foldnum")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--top_num",
                        default=4,
                        type=int,
                        help="Total top doc number.")
    parser.add_argument("--overlap",
                        default=4,
                        type=int,
                        help="Total overlap number.")
    parser.add_argument("--learning_rate",
                        default=3e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--total_epoch",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)
    parser.add_argument('--save_step',
                        type=int,
                        default=50000)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # Prepare  Data
    task_name = args.task_name.lower()
    output_mode = output_modes[task_name]
    label_list = get_labels(task_name.lower())
    num_labels = len(label_list)


    # folds = [int(args.fold)]
    folds = range(1, int(args.fold)+1)
    for fold in folds:
        if args.model_name == 'cobert':
            student_model = cobert(args.encoder_model, args.groupwise_model, args.prf_model, num_labels, fold=fold)
            logger.info('model_name: cobert')
        elif args.model_name == 'prf_only':
            student_model = cobert_prf_only(args.encoder_model, args.groupwise_model, args.prf_model, num_labels,
                                                     fold=fold)
            logger.info('model_name: prf_only')
        elif args.model_name == 'group_only':
            student_model = cobert_group_only(args.encoder_model, args.groupwise_model, args.prf_model, num_labels,
                                                     fold=fold)
            logger.info('model_name: group_only')
        elif args.model_name == 'no_resi':
            student_model = cobert_no_resi(args.encoder_model, args.groupwise_model, args.prf_model, num_labels,
                                                     fold=fold)
            logger.info('model_name: no_resi')
        elif args.model_name == 'freeze':
            student_model = cobert_freeze(args.encoder_model, args.groupwise_model, args.prf_model, num_labels,
                                                     fold=fold)
            logger.info('model_name: freeze')
        elif args.model_name == 'freeze_embedding':
            student_model = cobert_freeze2embedding(args.encoder_model, args.groupwise_model, args.prf_model, num_labels,
                                                     fold=fold)
            logger.info('model_name: freeze_embedding')
        else:
            raise NotImplementedError
        student_model.to(device)


        # Prepare task settings
        if os.path.exists(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)) \
                and os.listdir(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(
                os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)))

        if not os.path.exists(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name)):
            os.makedirs(os.path.join(args.output_dir, str(fold), 'train', args.outdir_name))
        output_loss_file = os.path.join(args.output_dir, str(fold), 'train', args.outdir_name, "train_loss.txt")

        total_examples = get_num_examples(fold, args)
        total_num_train_optimization_steps = int(total_examples / args.train_batch_size / args.gradient_accumulation_steps)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters of student_model: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_num_train_optimization_steps)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            student_model, optimizer = amp.initialize(student_model, optimizer,  opt_level="O1")
            # teacher_model.half()
            logger.info('FP16 is activated, use amp')
        else:
            logger.info('FP16 is not activated, only use BertAdam')

        total_global_step = 0
        tr_loss = 0.

        for t_epoch in range(int(args.total_epoch)):
            if args.gradient_accumulation_steps < 1:
                raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    args.gradient_accumulation_steps))

            args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

            num_examples, train_dataloader = get_rank_task_dataloader(fold, task_name, 'train', args,
                                                                      SequentialSampler, batch_size=args.train_batch_size)
            num_train_optimization_steps_per_epoch = int(
                num_examples / args.train_batch_size / args.gradient_accumulation_steps)

            logger.info("***** Running training *****")
            logger.info("  Num fold = %d", fold)
            logger.info("  Num fold epoch = %d", t_epoch)
            logger.info("  Num examples = %d", num_examples)
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps_per_epoch)
            logger.info("***** Init Query Weights *****")
            logger.info("\n")

            # Train and evaluate
            global_step = 0

            student_model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t for t in batch)
                if args.model_name == 'freeze':
                    input_ids, input_mask, segment_ids, label_ids, query_ids, doc_ids, features = batch
                    features = features.to(device)
                else:
                    input_ids, input_mask, segment_ids, label_ids, query_ids, doc_ids = batch
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                query_ids = query_ids.cpu().numpy()


                if input_ids.size()[0] != args.train_batch_size:
                    continue
                try:
                    if args.model_name == 'freeze':
                        loss, student_logits = student_model(features, input_ids, segment_ids, input_mask, label_ids,
                                                             args)
                    else:
                        loss, student_logits = student_model(input_ids, segment_ids, input_mask, label_ids, args)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        raise exception
                    else:
                        raise exception
                #
                if output_mode == "classification":
                    loss = torch.nn.functional.cross_entropy(student_logits, label_ids[int(args.top_num):], reduction='mean')


                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                clip = 1
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip)
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    total_global_step += 1

                if total_global_step % args.eval_step == 0:

                    loss = tr_loss / (total_global_step + 1)
                    result = {}
                    result['global_step'] = total_global_step
                    result['loss'] = loss
                    result_to_file(result, output_loss_file)


                if total_global_step % args.save_step == 0 or global_step == num_train_optimization_steps_per_epoch:
                    logger.info("***** Save model *****")

                    model_to_save = student_model.module if hasattr(student_model,
                                                                    'module') else student_model
                    checkpoint_name = 'checkpoint-' + str(total_global_step)
                    output_model_dir = os.path.join(args.output_dir, str(fold),'train',args.outdir_name, checkpoint_name)
                    if not os.path.exists(output_model_dir):
                        os.makedirs(output_model_dir)
                    torch.save(model_to_save.state_dict(), os.path.join(output_model_dir, 'weights.pt'))






if __name__ == "__main__":
    main()
