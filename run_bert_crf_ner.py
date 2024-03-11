import json
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import time
from seqeval.metrics import classification_report as conll_report
from sklearn.metrics import classification_report as word_report
from tqdm import tqdm as tqdm

from transformers import AdamW, BertModel, AutoTokenizer

cuda_yes = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_yes else "cpu")


def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0] + torch.log(torch.exp(log_M - torch.max(log_M, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_Tensor, axis=-1):  # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + torch.log(
        torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data.
    result of convert_examples_to_features(InputExample)
    """

    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids


class CoNLLDataProcessor:
    '''
    CoNLL-2003
    '''

    def __init__(self, out_lists, label_types, num_train):
        self.data = out_lists
        self.num_train = num_train
        self._label_types = label_types + ["[CLS]", "[SEP]"]
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i, label in enumerate(self._label_types)}

    def get_train_examples(self):
        return self._create_examples(self.data[:self.num_train])

    def get_test_examples(self):
        return self._create_examples(self.data[self.num_train:])

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]
            examples.append(InputExample(
                guid=guid, words=words, labels=labels))
        return examples


class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = self.example2feature(self.examples[idx], self.tokenizer,
                                    self.label_map, self.max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def pad(cls, batch):
        seqlen_list = [len(sample[0]) for sample in batch]
        if args.dynamic_padding:
            maxlen = np.array(seqlen_list).max()
        else:
            maxlen = args.max_seq_length

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

    @staticmethod
    def example2feature(example, tokenizer, label_map, max_seq_length):
        add_label = 'X'
        # tokenize_count = []
        tokens = ['[CLS]']
        predict_mask = [0]
        label_ids = [label_map['[CLS]']]
        for i, w in enumerate(example.words):
            # use bertTokenizer to split words
            # 1996-08-22 => 1996 - 08 - 22
            # sheepmeat => sheep ##me ##at
            sub_words = tokenizer.tokenize(w)
            if not sub_words:
                sub_words = ['[UNK]']
            # tokenize_count.append(len(sub_words))
            tokens.extend(sub_words)
            for j in range(len(sub_words)):
                if j == 0:
                    predict_mask.append(1)
                    label_ids.append(label_map[example.labels[i]])
                else:
                    # '##xxx' -> 'X' (see bert paper)
                    predict_mask.append(0)
                    label_ids.append(label_map[example.labels[i]])

        # truncate
        if len(tokens) > max_seq_length - 1:
            print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens),
                                                                                     max_seq_length))
            tokens = tokens[0:(max_seq_length - 1)]
            predict_mask = predict_mask[0:(max_seq_length - 1)]
            label_ids = label_ids[0:(max_seq_length - 1)]
        tokens.append('[SEP]')
        predict_mask.append(0)
        label_ids.append(label_map['[SEP]'])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        feat = InputFeatures(
            # guid=example.guid,
            # tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            predict_mask=predict_mask,
            label_ids=label_ids)

        return feat


class BERT_CRF_NER(nn.Module):
    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device

        # use pretrained BertModel
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        # additional layers
        self.sent_attn_layer = SentenceAttentionLayer(self.hidden_size, attention_dim=512)
        self.bilstm = nn.LSTM(self.hidden_size, self.hidden_size, 2, batch_first=True)
        # note: padding='same' guarantees output dim = self.hidden_size
        self.cnn = nn.Conv1d(self.max_seq_length, self.max_seq_length, kernel_size=5, padding='same')
        # Maps the output of the bert into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)
        # self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        '''
        sentences -> word embedding -> lstm -> MLP -> feats
        '''
        bert_seq_out = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]

        if args.use_attention:
            # Note: make sure that batch_size is even
            ori_indices = torch.tensor(range(0, bert_seq_out.shape[0], 2))
            ori_rep = torch.index_select(bert_seq_out, 0, ori_indices.to(device))
            add_indices = torch.tensor(range(1, bert_seq_out.shape[0], 2))
            add_rep = torch.index_select(bert_seq_out, 0, add_indices.to(device))
            bert_seq_out = self.sent_attn_layer(ori_rep.to(device), add_rep.to(device))

        if args.use_bilstm:
            bert_seq_out, _ = self.bilstm(bert_seq_out)

        if args.use_cnn:
            ori_seq_length = bert_seq_out.shape[1]
            if args.dynamic_padding:
                # pad to max_seq_length
                bert_seq_out = nn.functional.pad(bert_seq_out, (0, 0, 0, self.max_seq_length - ori_seq_length),
                                                 "constant", 0)
            bert_seq_out = self.cnn(bert_seq_out)
            if args.dynamic_padding:
                # change back to original sequence length
                bert_seq_out = bert_seq_out[:, :ori_seq_length, :]

        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, feats, label_ids):
        '''
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                    batch_transitions.gather(-1, (label_ids[:, t] * self.num_labels + label_ids[:, t - 1]).view(-1, 1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path chosen from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bert_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bert_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    # this forward is just for predict, not for train
    # don't confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)

        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids


class SentenceAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(SentenceAttentionLayer, self).__init__()
        self.normalize = nn.LayerNorm(hidden_dim)
        self.linear_proj = nn.Linear(hidden_dim, attention_dim)
        # Sentence-level context vector u_s
        self.sent_proj = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x, y):
        """
        perform sentence-level attention between original sentence and additional context
        :param x: embeddings of original sentence
        :param y: embeddings of additional context
        :return:
        """
        y_norm = self.normalize(y)
        ui = torch.tanh(self.linear_proj(y_norm))

        # Compute attention matrix
        ai = self.sent_proj(ui).squeeze(-1)
        ai = F.softmax(ai, dim=1)
        ai = ai.unsqueeze(-1)

        # Get weighted sentences input
        output = x * ai

        return output


def evaluate(model, predict_dataloader, batch_size, use_conlleval):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []

    inverted_map = {}

    for i in label_map:
        inverted_map[label_map[i]] = i

    cls_label_id = list(inverted_map.keys())[-2]
    sep_label_id = list(inverted_map.keys())[-1]

    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            if args.use_attention:
                ori_indices = torch.tensor(range(0, label_ids.shape[0], 2))
                predict_mask = torch.index_select(predict_mask, 0, ori_indices.to(device))
                label_ids = torch.index_select(label_ids, 0, ori_indices.to(device))
            score, label_seq_ids = model(input_ids, segment_ids, input_mask)

            predicted = np.array(label_seq_ids.cpu())
            mask = np.array(predict_mask.cpu())
            truth = np.array(label_ids.cpu())

            for M in range(len(mask)):
                cls_idx = np.where(truth[M] == cls_label_id)[0][0]
                sep_idx = np.where(truth[M] == sep_label_id)[0][0]
                if use_conlleval:
                    all_preds.append(
                        [inverted_map[i] for i in predicted[M][cls_idx + 1:sep_idx][mask[M][cls_idx + 1:sep_idx] == 1]])
                    all_labels.append(
                        [inverted_map[i] for i in truth[M][cls_idx + 1:sep_idx][mask[M][cls_idx + 1:sep_idx] == 1]])
                else:
                    all_preds.extend(
                        [inverted_map[i] if inverted_map[i] != "[CLS]" else "O" for i in predicted[M][mask[M]]])
                    all_labels.extend(
                        [inverted_map[i] if inverted_map[i] != "[CLS]" else "O" for i in truth[M][mask[M]]])

    if use_conlleval:
        print(conll_report(all_labels, all_preds, digits=4))
    else:
        print(word_report(all_labels, all_preds,
                          labels=[label for label in list(set(all_labels)) if label not in ["O", "X"]]))
    return all_preds


def get_results(mask_list, res_list, inverted_map):
    cls_idx = np.where(res_list == 3)[0][0]
    sep_idx = np.where(res_list == 4)[0][0]
    res_list = res_list[cls_idx + 1:sep_idx]
    mask_list = mask_list[cls_idx + 1:sep_idx]
    return [inverted_map[i] for i in res_list]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def is_good_label(label):
    # TODO: change valid labels here!
    valued_labels = ["Chemical", "Disease"]
    if args.additional_data:
        valued_labels.append("X")
    for I in valued_labels:
        if I in label:
            return True
    return label == 'O'


def data_processing(input_files):
    X, y = list(), list()
    error, _num_train = 0, 0

    for fn in input_files:
        file = open(os.path.join(args.data_path, f"{args.dataset}/{fn}"), 'r')
        rows = [line.strip() for line in file.readlines()]
        flag = True
        X.append([])
        y.append([])

        for row in rows:
            tokens = row.strip().split("\t")
            if len(row) == 0:
                X.append([])
                y.append([])
                flag = True
            else:
                if flag:
                    if len(tokens) == 1 or not is_good_label(tokens[-1]):
                        flag = False
                        error += 1
                        X.pop(-1)
                        y.pop(-1)
                    else:
                        X[-1].append(tokens[0])
                        y[-1].append(tokens[-1].split("-")[-1])

        assert len(X) == len(y)
        if "train" in fn:
            _num_train = len(X)

        for M in range(len(X)):
            assert len(X[M]) == len(y[M])

    _out_lists, _label_types = list(), list()

    for row in range(len(X)):
        _out_lists.append([X[row], y[row]])
        _label_types.extend(y[row])

    assert _num_train > 0, print("No train file provided?!")
    return _out_lists, list(set(_label_types)), _num_train


if __name__ == "__main__":
    batch_size = 8
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0

    # Argument parsings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="The path where the xlsx file is stored.",
    )
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--train_fn", default=None, type=str, required=True, help="Train file name.")
    parser.add_argument("--test_fn", default=None, type=str, required=True, help="Test file name.")
    parser.add_argument("--pretrained_model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                        type=str, help="Pretrained language model.")
    parser.add_argument("--num_epochs", default=5, type=int, help="Number of training epochs.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="Max sequence length.")
    parser.add_argument("--additional_data", default=False, type=bool, help="Additional data added or not.")
    parser.add_argument("--dynamic_padding", default=True, type=bool,
                        help="Whether or not to use dynamic batch padding.")
    parser.add_argument("--use_bilstm", default=False, type=bool, help="Whether or not to use BiLSTM layer.")
    parser.add_argument("--use_cnn", default=False, type=bool, help="Whether or not to use CNN layer.")
    parser.add_argument("--use_attention", default=False, type=bool, help="Whether or not to use attention layer.")
    parser.add_argument("--use_conlleval", default=True, type=bool, help="Whether or not to use conll evaluation.")
    args = parser.parse_args()

    # Data Processing
    out_lists, label_types, num_train = data_processing([args.train_fn, args.test_fn])
    print("DATA:", len(out_lists), label_types, num_train)

    conllProcessor = CoNLLDataProcessor(out_lists, label_types, num_train)
    label_list = conllProcessor.get_labels()
    label_map = conllProcessor.get_label_map()
    start_label_id = conllProcessor.get_start_label_id()
    stop_label_id = conllProcessor.get_stop_label_id()
    # Examples
    train_examples = conllProcessor.get_train_examples()
    test_examples = conllProcessor.get_test_examples()

    # Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    train_dataset = NerDataset(train_examples, tokenizer, label_map, args.max_seq_length)
    test_dataset = NerDataset(test_examples, tokenizer, label_map, args.max_seq_length)

    # Dataloaders
    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,  # set this to False if use train_v3
                                       num_workers=4,
                                       collate_fn=NerDataset.pad)

    test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      collate_fn=NerDataset.pad)

    bert_model = BertModel.from_pretrained(args.pretrained_model)
    model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id,
                         len(label_list), args.max_seq_length, batch_size, device)
    model.to(device)

    # Hyper-parameters for Optimizer
    param_optimizer = list(model.named_parameters())
    weight_decay_finetune = 1e-5  # 0.01
    weight_decay_crf_fc = 5e-6  # 0.005
    lr0_crf_fc = 8e-5
    learning_rate0 = 5e-5

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in new_param)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')],
         'lr': lr0_crf_fc, 'weight_decay': weight_decay_crf_fc},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'],
         'lr': lr0_crf_fc, 'weight_decay': 0.0}
    ]

    optimizer = AdamW(model.parameters(), lr=learning_rate0, correct_bias=False)

    total_train_epochs = args.num_epochs
    gradient_accumulation_steps = 1
    total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
    global_step_th = 0
    warmup_proportion = 0.1

    for epoch in range(total_train_epochs):
        tr_loss = 0
        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            if args.use_attention:
                ori_label_indices = torch.tensor(range(0, label_ids.shape[0], 2))
                label_ids = torch.index_select(label_ids, 0, ori_label_indices.to(device))

            neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

            if gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

            neg_log_likelihood.backward()

            tr_loss += neg_log_likelihood.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = learning_rate0 * warmup_linear(global_step_th / total_train_steps, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss,
                                                                                 (time.time() - train_start) / 60.0))
        pred_res = evaluate(model, test_dataloader, batch_size, args.use_conlleval)
        with open(os.path.join(args.data_path, f"{args.dataset}/results/pred_{epoch}"), 'w', encoding='utf-8') as pred_file:
            json.dump(pred_res, pred_file)
