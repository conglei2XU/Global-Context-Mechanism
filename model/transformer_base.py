import logging
import pdb
from abc import ABC
from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, BertModel, RobertaModel, XLNetModel

from model.context_mechanism import GlobalContext, SAN
from model.crf import CRFPretrained
from utils.constants import PAD

TAGGER = {
    'LSTM': nn.LSTM,
    'GRU': nn.GRU
}
CONTEXT = {
    'global': GlobalContext,
    'self-attention': SAN,
    'cross-ner': '',
}
TRANSFORMERS = {'bert-base-cased': AutoModel,
                'roberta-base': RobertaModel,
                'xlnet-cased': XLNetModel}

logger = logging.getLogger('__main__')


def log_sum_exp_batch(log_Tensor, axis=-1):  # shape (batch_size,n,m)
    # torch.max return max_value and max_value_indices,
    # in there, get the transition label_idx with max score (batch_size, num_labels)
    return torch.max(log_Tensor, axis)[0] + torch.log(
        torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class BertForSeqTask(nn.Module):

    def __init__(self, model_name, bert_config):
        """
        根据bert_config 以及 model_name 初始化 模型结构
        """

        super(BertForSeqTask, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        tagger_config = bert_config.tagger_config
        self.use_tagger = bert_config.use_tagger
        self.use_context = bert_config.use_context
        self.use_crf = bert_config.use_crf
        self.label2idx = bert_config.label2idx
        self.num_labels = len(bert_config.label2idx)

        # self.drop_out = bert_config.hidden_dropout_prob
        self.num_layers = tagger_config['num_layers']
        # if bert_config.fix_bert:
        self.tagger_size = tagger_config['hidden_size']
        self.bert_config = bert_config
        if bert_config.fix_pretrained:
            for parameters in self.bert.parameters():
                parameters.requires_grad = False

        if bert_config.use_tagger:
            if tagger_config['bidirectional']:
                hidden_dim = tagger_config['hidden_size'] // 2
            else:
                hidden_dim = tagger_config['hidden_size']
            self.tagger_size = tagger_config['hidden_size']

            self.tagger = TAGGER[tagger_config['tagger_name']](tagger_config['input_size'],
                                                               hidden_size=int(hidden_dim),
                                                               bidirectional=True,
                                                               batch_first=True,
                                                               num_layers=self.num_layers)
            final_hidden_dim = tagger_config['hidden_size']
        else:
            final_hidden_dim = bert_config.hidden_size
            self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        if self.use_context:
            self.context_name = tagger_config['context_mechanism']

            input_dim = tagger_config['hidden_size'] if bert_config.use_tagger else bert_config.hidden_size
            if tagger_config['context_mechanism'] == 'global':
                self.context_mechanism = CONTEXT[tagger_config['context_mechanism']](input_dim,
                                                                                     tagger_config['dropout_rate'])
            if tagger_config['context_mechanism'] == 'self-attention':
                self.num_heads = 6
                self.context_mechanism = CONTEXT[tagger_config['context_mechanism']](input_dim, self.num_heads)
                # self.classifier = nn.Linear(input_dim * 2, self.num_labels)
        self.num_layers = 1
        # self.extra_linear = nn.Linear(final_hidden_dim, 4 * final_hidden_dim)
        # self.full_linear = nn.Linear(4 * final_hidden_dim, final_hidden_dim)
        if self.use_crf:
            self.fc = nn.Linear(final_hidden_dim, self.num_labels)
            self.CRF = CRFPretrained(
                label_size=self.num_labels,
                label2idx=self.label2idx)

        self.drop_out = nn.Dropout(0.1)
        # input_size = bert_config.hidden_size
        self.classifier = nn.Linear(final_hidden_dim, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                inference_masks=None,
                orig_to_tok_index=None,
                word_seq_len=None,
                wordpiece_len=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        # (loss[Optional], logit, hidden_states[Optional], output_attentions[Optional])
        sequence_output, gate_weight, outputs = self.get_features(input_ids,
                                                                  attention_mask=attention_mask,
                                                                  orig_to_token_index=orig_to_tok_index,
                                                                  wordpiece_len=wordpiece_len
                                                                  )
        # sequence_output = outputs[0]
        batch_size = input_ids.size(0)

        if self.use_crf:
            assert orig_to_tok_index is not None, "In CRF mode, we need to select word from word pieces"
            word_rep = self.fc(sequence_output)
            bestScores, logits = self.CRF.decode(word_rep, word_seq_len)

        else:
            # intermidate_state = self.extra_linear(sequence_output)
            # sequence_output = self.full_linear(intermidate_state)
            logits = self.classifier(sequence_output)

        # add extra linear parameters

        loss = None
        if labels is not None and not self.use_crf:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.label2idx[PAD])
            # logits, _ = torch.max(logits, dim=-1)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]

        return ((loss,) + output), gate_weight

    def loss(self,
             input_ids=None,
             attention_mask=None,
             token_type_ids=None,
             position_ids=None,
             head_mask=None,
             inputs_embeds=None,
             labels=None,
             inference_masks=None,
             orig_to_tok_index=None,
             word_seq_len=None,
             wordpiece_len=None,
             output_attentions=None,
             output_hidden_states=None,
             return_dict=None,
             ):
        """
        calculate the loss of the model using CRF
        """
        sequence_output, gate_weights, outputs = self.get_features(input_ids,
                                                                   attention_mask=attention_mask,
                                                                   orig_to_token_index=orig_to_tok_index,
                                                                   wordpiece_len=wordpiece_len
                                                                   )

        word_rep = self.fc(sequence_output)
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=word_rep.device).view(1, sent_len).expand(
            batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_len.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score = self.CRF(word_rep, word_seq_len, labels, mask)
        return unlabed_score - labeled_score

    def get_features(self,
                     input_ids,
                     attention_mask,
                     token_type_ids=None,
                     labels=None,
                     orig_to_token_index=None,
                     wordpiece_len=None
                     ):
        sequence_output = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    ).last_hidden_state
        # sequence_output = outputs[0]
        outputs = ()
        gate_weight = None
        batch_size = input_ids.size(0)
        # remove the [CLS] token and [SEP] token
        # max_seq_len = input_ids.size(1)
        # select_mask, wordpiece_len_new = [], []
        # for seq_len_tensor in wordpiece_len:
        #     seq_len = seq_len_tensor.item()
        #     mask_ = [0] + [1] * (seq_len-2) + [0] + [1] * (max_seq_len - seq_len)
        #     select_mask.append(mask_)
        #     wordpiece_len_new.append(seq_len - 2)
        # select_mask_tensor = torch.tensor(select_mask, device=sequence_output.device, dtype=torch.bool)
        # attention_mask_self = torch.masked_select(attention_mask, select_mask_tensor).contiguous().view(batch_size, -1)
        # wordpiece_len_new = torch.tensor(wordpiece_len_new)
        # sequence_output = torch.masked_select(sequence_output,
        #                                       select_mask_tensor.unsqueeze(-1),
        #                                       ).view(batch_size, -1, 768)
        if self.use_tagger:
            packed_rnn_input = pack_padded_sequence(sequence_output,
                                                    batch_first=True,
                                                    lengths=wordpiece_len.cpu(),
                                                    enforce_sorted=False)
            # for not use first token [CLS] and last token [SEP]
            # packed_rnn_input = pack_padded_sequence(sequence_output,
            #                                         batch_first=True,
            #                                         lengths=wordpiece_len_new,
            #                                         enforce_sorted=False)
            packed_rnn_output, (h_n, c_n) = self.tagger(packed_rnn_input)
            rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)
            # laptop14 for bert-base-cased
            # forward_global = rnn_output[torch.arange(batch_size), wordpiece_len_new - 1, :self.tagger_size//2]
            # forward_global = sequence_output[:, -1, :self.tagger_size // 2]
            # Conll2003
            # forward_global = h_n[0, :, :]
            # backward_global = rnn_output[:, 0, self.tagger_size // 2:]
            # intermidate_step = F.relu(self.extra_linear(sequence_output))
            # sequence_output = self.full_linear(intermidate_step)
            # cls_plachoder
            context_input = rnn_output
            hidden_size = self.tagger_size // 2
            sequence_output = rnn_output
        else:
            context_input = sequence_output
            hidden_size = self.bert_config.hidden_size // 2
        # remove the [CLS] token and [SEP] token for Globa Context Mechanism
        # max_seq_len = input_ids.size(1)
        # select_mask, wordpiece_len_new = [], []
        # for seq_len_tensor in wordpiece_len:
        #     seq_len = seq_len_tensor.item()
        #     mask_ = [0] + [1] * (seq_len-2) + [0] + [1] * (max_seq_len - seq_len)
        #     select_mask.append(mask_)
        #     wordpiece_len_new.append(seq_len - 2)
        # select_mask_tensor = torch.tensor(select_mask, device=sequence_output.device, dtype=torch.bool)
        # attention_mask_self = torch.masked_select(attention_mask, select_mask_tensor).contiguous().view(batch_size, -1)
        # wordpiece_len_new = torch.tensor(wordpiece_len_new)
        # context_input = torch.masked_select(context_input,
        #                                       select_mask_tensor.unsqueeze(-1),
        #                                       ).view(batch_size, -1, 600)
        if self.use_context:
            assert wordpiece_len is not None, 'context mechanism need wordpiece length'
            backward_global = sequence_output[:, 0, hidden_size // 2:]
            # for not use first token [CLS] and last token [SEP]
            # forward_global = sequence_output[torch.arange(batch_size), wordpiece_len_new - 1, :hidden_size//2]

            forward_global = sequence_output[torch.arange(batch_size), wordpiece_len - 1, :hidden_size // 2]

            if self.context_name == 'global':
                sequence_output, gate_weight = self.context_mechanism(context_input, forward_global, backward_global)
                # for context directions experiments
                # sequence_output, gate_weight = self.context_mechanism(sequence_output, backward_global,
                # forward_global)
            if self.context_name == 'self-attention':
                # sequence_outp
                # ut = torch.cat([sequence_output] * self.num_heads, dim=2)
                # for not use first token [CLS] and last token [SEP]
                # sequence_output, gate_weight = self.context_mechanism(context_input, src_mask=attention_mask_self)
                sequence_output, gate_weight = self.context_mechanism(context_input, src_mask=attention_mask)
        if self.use_crf:
            assert orig_to_token_index is not None, "In CRF mode, we need to select word from word pieces"
            batch_size, _, rep_size = sequence_output.size()
            _, max_sent_len = orig_to_token_index.size()
            # select the word index.
            sequence_output = torch.gather(sequence_output[:, :, :], 1,
                                           orig_to_token_index.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size))

        return sequence_output, gate_weight, outputs
