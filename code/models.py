import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, MaxPool1d, AvgPool1d
import transformers
from transformers import BertForTokenClassification, BertModel, BertConfig
from transformer.modeling import BertForSequenceClassification, BertEmbeddings
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class cobert(torch.nn.Module):
    def __init__(self, encoder_path, group_path, prf_path, num_labels, fold=None):
        super(cobert, self).__init__()
        if fold is not None:
            encoder_path = encoder_path.format(str(fold))
        self.QT_BERT = BertForSequenceClassification.from_pretrained(encoder_path, num_labels=num_labels)
        self.ATTN_BERT = transformers.BertForSequenceClassification.from_pretrained(prf_path, num_labels=num_labels)
        self.QB_BERT = BertForTokenClassification.from_pretrained(group_path, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ENCODER_input_ids, ENCODER_token_type_ids=None, ENCODER_attention_masks=None, logit_labels=None, args=None):
        batch_size = ENCODER_input_ids.size(0)
        # logger.info(f'size attn:{str(ENCODER_input_ids.size(0))}')
        top_num = int(args.top_num)
        overlap = int(args.overlap)
        ENCODER_logits, ENCODER_pooled_output = self.QT_BERT(ENCODER_input_ids, ENCODER_token_type_ids, ENCODER_attention_masks)

        device = ENCODER_input_ids.device
        QT_attn = ENCODER_pooled_output[top_num:].unsqueeze(1)
        PRF_input = ''
        for i in range(top_num):
            attn_start = ENCODER_pooled_output[i:i+1].unsqueeze(0)
            attn_start = attn_start.expand(batch_size-top_num, 1, -1)
            if i==0:
                PRF_input = torch.cat((attn_start, QT_attn), 1)
            else:
                now = torch.cat((attn_start, QT_attn), 1)
                PRF_input = torch.cat((PRF_input, now), 0)


        PRF_output = self.ATTN_BERT(inputs_embeds=PRF_input, output_hidden_states=True,
                                     return_dict=True)
        PRF_now = ''
        logits_for_soft = ENCODER_logits[:top_num, 1].expand(batch_size-top_num, top_num)
        probs = F.softmax(logits_for_soft, dim=-1)
        # logger.info(f'size attn:{str(ENCODER_logits[:4, 1].shape)}')
        for i in range(top_num):
            if i==0:
                PRF_now = PRF_output.hidden_states[-1][:batch_size-top_num, -1]*probs[:,i].unsqueeze(1)
            else:
                now = PRF_output.hidden_states[-1][(batch_size-top_num)*i:(batch_size-top_num)*(i+1), -1]*probs[:,i].unsqueeze(1)
                PRF_now = PRF_now+now

        PRF_rep = (PRF_now + ENCODER_pooled_output[top_num:]) / 2
        GROUP_input = PRF_rep.unsqueeze(0)
        GROUP_token_type_ids = torch.zeros(batch_size-top_num, dtype=torch.long, device=device)
        if overlap!=0:
            GROUP_token_type_ids[0:overlap] = 1
        if logit_labels is not None:
            logit_labels = logit_labels[top_num:].unsqueeze(0)
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids,
                                     return_dict=True, labels=logit_labels)
            return GROUP_output.loss, GROUP_output.logits.squeeze(0)
        else:
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids,
                                     return_dict=True)
            return GROUP_output.logits.squeeze(0)


class cobert_freeze(torch.nn.Module):
    def __init__(self, encoder_path, group_path, prf_path, num_labels, fold=None):
        super(cobert_freeze, self).__init__()

        if fold is not None:
            encoder_path = encoder_path.format(str(fold))
            if 'best4test' in encoder_path:
                dir_list = [item for item in os.listdir(encoder_path) if item.startswith('checkpoint')]
                encoder_path = os.path.join(encoder_path,dir_list[0])
        logger.info(f'freeze path:{encoder_path}')
        self.QT_BERT = BertForSequenceClassification.from_pretrained(encoder_path, num_labels=num_labels)

        for p in self.parameters():
            p.requires_grad = False
        self.ATTN_BERT = transformers.BertForSequenceClassification.from_pretrained(prf_path, num_labels=num_labels)

        self.QB_BERT = BertForTokenClassification.from_pretrained(group_path, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features, ENCODER_input_ids, ENCODER_token_type_ids=None, ENCODER_attention_masks=None, logit_labels=None, args=None):
        batch_size = ENCODER_input_ids.size(0)
        top_num = int(args.top_num)
        overlap = int(args.overlap)
        device = ENCODER_input_ids.device
        ENCODER_pooled_output = features
        GROUP_input = ENCODER_pooled_output.unsqueeze(0)

        GROUP_token_type_ids = torch.zeros(batch_size-top_num, dtype=torch.long, device=device)
        if overlap!=0:
            GROUP_token_type_ids[0:overlap] = 1
        if logit_labels is not None:
            logit_labels = logit_labels[top_num:].unsqueeze(0)
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids,
                                     return_dict=True, labels=logit_labels)
            return GROUP_output.loss, GROUP_output.logits.squeeze(0)
        else:
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids,
                                     return_dict=True)

            return GROUP_output.logits.squeeze(0)


class cobert_freeze2embedding(torch.nn.Module):
    def __init__(self, encoder_path, group_path, prf_path, num_labels, fold=None):
        super(cobert_freeze2embedding, self).__init__()

        if fold is not None:
            encoder_path = encoder_path.format(str(fold))
            if 'best4test' in encoder_path:
                dir_list = [item for item in os.listdir(encoder_path) if item.startswith('checkpoint')]
                encoder_path = os.path.join(encoder_path,dir_list[0])
        # self.QT_BERT.to(device)
        logger.info(f'freeze path:{encoder_path}')
        self.QT_BERT = BertForSequenceClassification.from_pretrained(encoder_path, num_labels=num_labels)

        for p in self.parameters():
            p.requires_grad = False
        self.ATTN_BERT = transformers.BertForSequenceClassification.from_pretrained(prf_path, num_labels=num_labels)
        self.QB_BERT = BertForTokenClassification.from_pretrained(group_path, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ENCODER_input_ids, ENCODER_token_type_ids=None, ENCODER_attention_masks=None, logit_labels=None, args=None):
        ENCODER_logits, ENCODER_pooled_output = self.QT_BERT(ENCODER_input_ids, ENCODER_token_type_ids, ENCODER_attention_masks)
        return ENCODER_logits, ENCODER_pooled_output


class cobert_no_resi(torch.nn.Module):
    def __init__(self, encoder_path, group_path, prf_path, num_labels, fold=None):
        super(cobert_no_resi, self).__init__()

        if fold is not None:
            encoder_path = encoder_path.format(str(fold))
        self.QT_BERT = BertForSequenceClassification.from_pretrained(encoder_path, num_labels=num_labels)
        self.ATTN_BERT = transformers.BertForSequenceClassification.from_pretrained(prf_path, num_labels=num_labels)

        self.QB_BERT = BertForTokenClassification.from_pretrained(group_path, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ENCODER_input_ids, ENCODER_token_type_ids=None, ENCODER_attention_masks=None, logit_labels=None, args=None):
        batch_size = ENCODER_input_ids.size(0)
        top_num = int(args.top_num)
        overlap = int(args.overlap)
        ENCODER_logits, ENCODER_pooled_output = self.QT_BERT(ENCODER_input_ids, ENCODER_token_type_ids, ENCODER_attention_masks)

        device = ENCODER_input_ids.device
        QT_attn = ENCODER_pooled_output[top_num:].unsqueeze(1)
        PRF_input = ''
        for i in range(top_num):
            attn_start = ENCODER_pooled_output[i:i+1].unsqueeze(0)
            attn_start = attn_start.expand(batch_size-top_num, 1, -1)
            if i==0:
                PRF_input = torch.cat((attn_start, QT_attn), 1)
            else:
                now = torch.cat((attn_start, QT_attn), 1)
                PRF_input = torch.cat((PRF_input, now), 0)

        PRF_output = self.ATTN_BERT(inputs_embeds=PRF_input, output_hidden_states=True,
                                     return_dict=True)
        PRF_now = ''
        logits_for_soft = ENCODER_logits[:top_num, 1].expand(batch_size-top_num, top_num)
        probs = F.softmax(logits_for_soft, dim=-1)
        # logger.info(f'size attn:{str(ENCODER_logits[:4, 1].shape)}')
        for i in range(top_num):
            if i==0:
                PRF_now = PRF_output.hidden_states[-1][:batch_size-top_num, -1]*probs[:,i].unsqueeze(1)
            else:
                now = PRF_output.hidden_states[-1][(batch_size-top_num)*i:(batch_size-top_num)*(i+1), -1]*probs[:,i].unsqueeze(1)
                PRF_now = PRF_now+now

        PRF_rep = PRF_now
        GROUP_input = PRF_rep.unsqueeze(0)

        GROUP_token_type_ids = torch.zeros(batch_size-top_num, dtype=torch.long, device=device)
        if overlap!=0:
            GROUP_token_type_ids[0:overlap] = 1
        if logit_labels is not None:
            logit_labels = logit_labels[top_num:].unsqueeze(0)
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids, return_dict=True, labels=logit_labels)
            return GROUP_output.loss, GROUP_output.logits.squeeze(0)
        else:
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids, return_dict=True)

            return GROUP_output.logits.squeeze(0)


class cobert_prf_only(torch.nn.Module):
    def __init__(self, encoder_path, group_path, prf_path, num_labels, fold=None):
        super(cobert_prf_only, self).__init__()

        if fold is not None:
            encoder_path = encoder_path.format(str(fold))
        self.QT_BERT = BertForSequenceClassification.from_pretrained(encoder_path, num_labels=num_labels)
        config = self.QT_BERT.config
        self.ATTN_BERT = transformers.BertForSequenceClassification.from_pretrained(prf_path, num_labels=num_labels)

        self.QB_BERT = BertForTokenClassification.from_pretrained(group_path, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, ENCODER_input_ids, ENCODER_token_type_ids=None, ENCODER_attention_masks=None, logit_labels=None, args=None):
        batch_size = ENCODER_input_ids.size(0)
        top_num = int(args.top_num)
        overlap = int(args.overlap)
        ENCODER_logits, ENCODER_pooled_output = self.QT_BERT(ENCODER_input_ids, ENCODER_token_type_ids, ENCODER_attention_masks)

        QT_attn = ENCODER_pooled_output[top_num:].unsqueeze(1)
        PRF_input = ''
        for i in range(top_num):
            attn_start = ENCODER_pooled_output[i:i+1].unsqueeze(0)
            attn_start = attn_start.expand(batch_size-top_num, 1, -1)
            if i==0:
                PRF_input = torch.cat((attn_start, QT_attn), 1)
            else:
                now = torch.cat((attn_start, QT_attn), 1)
                PRF_input = torch.cat((PRF_input, now), 0)

        PRF_output = self.ATTN_BERT(inputs_embeds=PRF_input, output_hidden_states=True,
                                     return_dict=True)
        PRF_now = ''
        logits_for_soft = ENCODER_logits[:top_num, 1].expand(batch_size-top_num, top_num)
        probs = F.softmax(logits_for_soft, dim=-1)
        for i in range(top_num):
            if i==0:
                PRF_now = PRF_output.hidden_states[-1][:batch_size-top_num, -1]*probs[:,i].unsqueeze(1)
            else:
                now = PRF_output.hidden_states[-1][(batch_size-top_num)*i:(batch_size-top_num)*(i+1), -1]*probs[:,i].unsqueeze(1)
                PRF_now = PRF_now+now

        PRF_rep = (PRF_now + ENCODER_pooled_output[top_num:]) / 2
        class_input = self.dropout(PRF_rep)
        output_logit = self.classifier(class_input)


        if logit_labels is not None:
            logit_labels = logit_labels[top_num:].unsqueeze(0)
            return logit_labels, output_logit
        else:
            return output_logit


class cobert_group_only(torch.nn.Module):
    def __init__(self, encoder_path, group_path, prf_path, num_labels, fold=None):
        super(cobert_group_only, self).__init__()

        if fold is not None:
            encoder_path = encoder_path.format(str(fold))
        self.QT_BERT = BertForSequenceClassification.from_pretrained(encoder_path, num_labels=num_labels)
        self.QB_BERT = BertForTokenClassification.from_pretrained(group_path, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ENCODER_input_ids, ENCODER_token_type_ids=None, ENCODER_attention_mask=None, logit_labels=None, args=None):
        batch_size = ENCODER_input_ids.size(0)
        top_num = int(args.top_num)
        overlap = int(args.overlap)
        ENCODER_logits, ENCODER_pooled_output = self.QT_BERT(ENCODER_input_ids, ENCODER_token_type_ids, ENCODER_attention_mask)

        device = ENCODER_input_ids.device

        PRF_rep = ENCODER_pooled_output[top_num:]
        GROUP_input = PRF_rep.unsqueeze(0)

        GROUP_token_type_ids = torch.zeros(batch_size - top_num, dtype=torch.long, device=device)
        if overlap != 0:
            GROUP_token_type_ids[0:overlap] = 1
        # GROUP_token_type_ids[12:batch_size] = 1
        if logit_labels is not None:
            logit_labels = logit_labels[top_num:].unsqueeze(0)
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids,
                                     return_dict=True, labels=logit_labels)
            return GROUP_output.loss, GROUP_output.logits.squeeze(0)
        else:
            GROUP_output = self.QB_BERT(inputs_embeds=GROUP_input, token_type_ids=GROUP_token_type_ids,
                                     return_dict=True)

            return GROUP_output.logits.squeeze(0)







