import sys
import os
import logging
import glob
import torch
import numpy as np
import linecache
import random
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pathlib import Path
import pickle
import gc
from tqdm import tqdm
from torch.utils.data.sampler import Sampler
import sys
import random
logger = logging.getLogger(__name__)


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, set_type, max_seq_length, num_examples,
                 output_mode='classification', reduce_memory=True):
        logger.info('training_path: {}'.format(training_path))
        self.seq_len = max_seq_length
        self.output_mode = output_mode
        self.set_type = set_type
        self.num_samples = num_examples
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            with NamedTemporaryFile('w+b') as tempf_ids, \
                 NamedTemporaryFile('w+b') as tempf_mask, \
                 NamedTemporaryFile('w+b') as tempf_seg, \
                 NamedTemporaryFile('w+b') as tempf_label, \
                 NamedTemporaryFile('w+b') as tempf_qid:
                print('filename is:', tempf_ids.name, tempf_mask.name, tempf_seg.name, tempf_label.name, tempf_qid)
                # input_ids = np.memmap(filename='../code/cache/rob_input_ids8_{}_{}.memmap'.format(training_path.replace('/',''), str(rint)),
                input_ids = np.memmap(filename=tempf_ids.name,
                                      mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
                input_masks = np.memmap(filename=tempf_mask.name,
                                        shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
                segment_ids = np.memmap(filename=tempf_seg.name,
                                        shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
                label_ids = np.memmap(filename=tempf_label.name,
                                      shape=(self.num_samples, ), mode='w+', dtype=np.int32)
                label_ids[:] = -1
                query_ids = np.memmap(filename=tempf_qid.name,
                                  shape=(self.num_samples, ), mode='w+', dtype=np.int32)
                query_ids[:] = -1
                doc_ids = [None]*self.num_samples

        else:
            raise NotImplementedError

        logging.info("Loading training examples.")

        with open(training_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Training examples")):
                # if i == 0:
                #     continue
                # print(i)
                tokens = line.strip().split(',')
                # if i < 2:
                #     print(tokens)
                input_ids[i] = [int(id) for id in tokens[3].split()]
                input_masks[i] = [int(id) for id in tokens[4].split()]
                segment_ids[i] = [int(id) for id in tokens[5].split()]

                query_ids[i] = int(tokens[0])
                doc_ids[i] = tokens[1]
                guid = "%s-%s" % (self.set_type, tokens[0]+'-'+tokens[1]+'-'+tokens[2])
                if self.set_type != 'test' and self.set_type != 'dev':
                    if self.output_mode == "classification":
                        label_ids[i] = int(tokens[6])
                    elif self.output_mode == "regression":
                        label_ids[i] = float(tokens[6])
                    else:
                        raise NotImplementedError
                else:
                    label_ids[i] = 0

                if label_ids[i] != 0 and label_ids[i] != 1:
                    # print(i)
                    # print(line)
                    if label_ids[i] > 0:
                        label_ids[i] = 1
                    else:
                        label_ids[i] = 0

                if i < 2:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[i]]))
                    logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks[i]]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids[i]]))
                    logger.info("label: %s" % str(label_ids[i]))
                    logger.info("qid: %s" % str(query_ids[i]))
                    logger.info("docid: %s" % str(doc_ids[i]))

        logging.info("Loading complete!")
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.query_ids = query_ids
        self.doc_ids = doc_ids


    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.output_mode == "classification":
            label_id = torch.tensor(self.label_ids[item], dtype=torch.long)
        elif self.output_mode == "regression":
            label_id = torch.tensor(self.label_ids[item], dtype=torch.float)
        else:
            raise NotImplementedError

        return (torch.tensor(self.input_ids[item], dtype=torch.long),
                torch.tensor(self.input_masks[item], dtype=torch.long),
                torch.tensor(self.segment_ids[item], dtype=torch.long),
                label_id,
                self.query_ids[item],
                self.doc_ids[item])

class PregeneratedDataset_freeze(Dataset):
    def __init__(self, features_file, training_path, set_type, max_seq_length, num_examples,
                 output_mode='classification', reduce_memory=True):
        logger.info('training_path: {}'.format(training_path))
        self.seq_len = max_seq_length
        self.output_mode = output_mode
        self.set_type = set_type
        self.num_samples = num_examples
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            with NamedTemporaryFile('w+b') as tempf_ids, \
                    NamedTemporaryFile('w+b') as tempf_mask, \
                    NamedTemporaryFile('w+b') as tempf_seg, \
                    NamedTemporaryFile('w+b') as tempf_label, \
                    NamedTemporaryFile('w+b') as tempf_qid:
                print('filename is:', tempf_ids.name, tempf_mask.name, tempf_seg.name, tempf_label.name, tempf_qid)
                # input_ids = np.memmap(filename='../code/cache/rob_input_ids8_{}_{}.memmap'.format(training_path.replace('/',''), str(rint)),
                input_ids = np.memmap(filename=tempf_ids.name,
                                      mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
                input_masks = np.memmap(filename=tempf_mask.name,
                                        shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
                segment_ids = np.memmap(filename=tempf_seg.name,
                                        shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
                label_ids = np.memmap(filename=tempf_label.name,
                                      shape=(self.num_samples,), mode='w+', dtype=np.int32)
                label_ids[:] = -1
                query_ids = np.memmap(filename=tempf_qid.name,
                                      shape=(self.num_samples,), mode='w+', dtype=np.int32)
                query_ids[:] = -1
                doc_ids = [None] * self.num_samples

        else:
            raise NotImplementedError

        logging.info("Loading training examples.")

        with open(training_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Training examples")):
                # if i == 0:
                #     continue
                # print(i)
                tokens = line.strip().split(',')
                # if i < 2:
                #     print(tokens)
                input_ids[i] = [int(id) for id in tokens[3].split()]
                input_masks[i] = [int(id) for id in tokens[4].split()]
                segment_ids[i] = [int(id) for id in tokens[5].split()]

                query_ids[i] = int(tokens[0])
                doc_ids[i] = tokens[1]
                guid = "%s-%s" % (self.set_type, tokens[0]+'-'+tokens[1]+'-'+tokens[2])
                if self.set_type != 'test' and self.set_type != 'dev':
                    if self.output_mode == "classification":
                        label_ids[i] = int(tokens[6])
                    elif self.output_mode == "regression":
                        label_ids[i] = float(tokens[6])
                    else:
                        raise NotImplementedError
                else:
                    label_ids[i] = 0

                if label_ids[i] != 0 and label_ids[i] != 1:
                    # print(i)
                    # print(line)
                    if label_ids[i] > 0:
                        label_ids[i] = 1
                    else:
                        label_ids[i] = 0

                if i < 2:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[i]]))
                    logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks[i]]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids[i]]))
                    logger.info("label: %s" % str(label_ids[i]))
                    logger.info("qid: %s" % str(query_ids[i]))
                    logger.info("docid: %s" % str(doc_ids[i]))

        logging.info("Loading complete!")
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.query_ids = query_ids
        self.doc_ids = doc_ids
        self.features = torch.load(features_file)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.output_mode == "classification":
            label_id = torch.tensor(self.label_ids[item], dtype=torch.long)
        elif self.output_mode == "regression":
            label_id = torch.tensor(self.label_ids[item], dtype=torch.float)
        else:
            raise NotImplementedError
        return (torch.tensor(self.input_ids[item], dtype=torch.long),
                torch.tensor(self.input_masks[item], dtype=torch.long),
                torch.tensor(self.segment_ids[item], dtype=torch.long),
                label_id,
                self.query_ids[item],
                self.doc_ids[item],
                self.features[item])


output_modes = {
    "msmarco": "classification",
    "rob": "classification",
    "gov": "classification",
    "clue": "classification",
}


def get_labels(task_name):
    """See base class."""
    if task_name.lower() == "msmarco":
        return ["0", "1"]
    elif task_name.lower() == "rob":
        return ["0", "1"]
    elif task_name.lower() == "gov":
        return ["0", "1"]
    elif task_name.lower() == "clue":
        return ["0", "1"]
    else:
        raise NotImplementedError


def get_num_examples(fold_num, args):
    epoch = int(args.total_epoch)
    if args.task_name=='msmarco':
        file_dir = os.path.join(args.data_dir, 'tokens/', args.data_name)
    else:
        file_dir = os.path.join(args.data_dir, str(fold_num), 'tokens/train/', args.data_name)
    num_examples = int(len(linecache.getlines(file_dir)))
    return num_examples*epoch


def get_rank_task_dataloader(fold_num, task_name, set_name, args, sampler, batch_size=None):

    output_mode = output_modes[task_name]

    if task_name == 'msmarco':
        file_dir = os.path.join(args.data_dir, 'tokens/', args.data_name)
    else:
        if set_name.lower() == 'pre':
            file_dir = os.path.join(args.data_dir, args.data_name)
        else:
            file_dir = os.path.join(args.data_dir, str(fold_num), 'tokens/', set_name.lower(), args.data_name)

    num_examples = int(len(linecache.getlines(file_dir)))
    print('number of examples: ', str(num_examples))
    if args.model_name=='freeze':
        feature_dirs = os.path.join(args.data_dir, str(fold_num), 'tokens/{}/features'.format(set_name.lower()))
        dir_list = [item for item in os.listdir(feature_dirs) if item.startswith('pooled')]
        feature_dir = os.path.join(args.data_dir, str(fold_num), 'tokens/{}/features'.format(set_name.lower()),
                                   dir_list[0])
        logger.info(f'feature_path:{feature_dir}')
        dataset = PregeneratedDataset_freeze(feature_dir, file_dir, set_name.lower(), args.max_seq_length,
                                      num_examples, output_mode, reduce_memory=True)
    else:
        dataset = PregeneratedDataset(file_dir, set_name.lower(), args.max_seq_length,
                                  num_examples, output_mode, reduce_memory=True)

    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return num_examples, dataloader




