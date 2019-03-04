# -*- coding: utf-8 -*-
#!/usr/bin/python3

import os
import pickle

from helper import import_tf, set_logger

__all__ = ['BertNer']

class BertNer(object):
    def __init__(self, **kwargs):
        self.tf = import_tf(kwargs['gpu_no'])
        self.logger = set_logger('BertNer', kwargs['log_dir'], kwargs['verbose'])
        self.model_dir = kwargs['ner_model']

        from bert.tokenization import FullTokenizer
        self.tokenizer = FullTokenizer(os.path.join(self.model_dir, 'vocab.txt'))
        
        self.ner_sq_len = 128
        self.input_ids = self.tf.placeholder(self.tf.int32, (None, self.ner_sq_len), 'input_ids')
        self.input_mask = self.tf.placeholder(self.tf.int32, (None, self.ner_sq_len), 'input_mask')

        # init graph
        self._init_graph()

        # init ner assist data
        self._init_predict_var()

        self.per_proun = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸', '子', '丑', '寅', '卯', '辰', '巳',
                          '午', '未', '申', '酉', '戌', '亥']

    def _init_graph(self):
        """
        init bert ner graph
        :return:
        """
        try:
            with self.tf.gfile.GFile(os.path.join(self.model_dir, 'ner_model.pb'), 'rb') as f:
                graph_def = self.tf.GraphDef()
                graph_def.ParseFromString(f.read())
                input_map = {"input_ids:0": self.input_ids,
                             'input_mask:0': self.input_mask}

                self.pred_ids = self.tf.import_graph_def(graph_def,
                                                         name='',
                                                         input_map=input_map,
                                                         return_elements=['pred_ids:0'])[0]
                graph = self.pred_ids.graph

                sess_config = self.tf.ConfigProto(allow_soft_placement=True)
                sess_config.gpu_options.allow_growth = True

                self.sess = self.tf.Session(graph=graph, config=sess_config)
                self.sess.run(self.tf.global_variables_initializer())
                self.tf.reset_default_graph()

        except Exception as e:
            self.logger.error(e)

    def _init_predict_var(self):
        """
        initialize assist of bert ner
        :return: labels num of ner, label to id dict, id to label dict
        """
        with open(os.path.join(self.model_dir, 'label2id.pkl'), 'rb') as rf:
            self.id2label = {value: key for key, value in pickle.load(rf).items()}

    def _convert_lst_to_features(self, lst_str, is_tokenized=True, mask_cls_sep=False):
        """
        Loads a data file into a list of `InputBatch`s.
        :param lst_str: list str
        :param is_tokenized: whether token unknown word
        :param mask_cls_sep: masking the embedding on [CLS] and [SEP] with zero.
        :return: input feature instance
        """
        from bert.extract_features import read_tokenized_examples, read_examples, InputFeatures

        examples = read_tokenized_examples(lst_str) if is_tokenized else read_examples(lst_str)

        _tokenize = lambda x: self.tokenizer.mark_unk_tokens(x) if is_tokenized else self.tokenizer.tokenize(x)

        for (ex_index, example) in enumerate(examples):
            tokens_a = _tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = _tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.ner_sq_len - 2:
                    tokens_a = tokens_a[0:(self.ner_sq_len - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ['[CLS]'] + tokens_a + ['[SEP]']
            input_type_ids = [0] * len(tokens)
            input_mask = [int(not mask_cls_sep)] + [1] * len(tokens_a) + [int(not mask_cls_sep)]

            if tokens_b:
                tokens += tokens_b + ['[SEP]']
                input_type_ids += [1] * (len(tokens_b) + 1)
                input_mask += [1] * len(tokens_b) + [int(not mask_cls_sep)]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # Zero-pad up to the sequence length. more pythonic
            pad_len = self.ner_sq_len - len(input_ids)
            input_ids += [0] * pad_len
            input_mask += [0] * pad_len
            input_type_ids += [0] * pad_len

            assert len(input_ids) == self.ner_sq_len
            assert len(input_mask) == self.ner_sq_len
            assert len(input_type_ids) == self.ner_sq_len

            yield InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

    def _truncate_seq_pair(self, tokens_a, tokens_b):
        """
        Truncates a sequence pair in place to the maximum length.
        :param tokens_a: text a
        :param tokens_b: text b
        """
        try:
            while True:
                total_length = len(tokens_a) + len(tokens_b)

                if total_length <= self.ner_sq_len - 3:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        except:
            self.logger.error()

    def _convert_id_to_label(self, pred_ids_result, batch_size):
        """
        turn id to label
        :param pred_ids_result: predict result
        :param batch_size: batch size of predict ids result
        :return: label list
        """
        result = []
        index_result = []
        for row in range(batch_size):
            curr_seq = []
            curr_idx = []
            ids = pred_ids_result[row]
            for idx, id in enumerate(ids):
                if id == 0:
                    break
                curr_label = self.id2label[id]
                if curr_label in ['[CLS]', '[SEP]']:
                    if id == 102 and (idx < len(ids) and ids[idx + 1] == 0):
                        break
                    continue
                # elif curr_label == '[SEP]':
                #     break
                curr_seq.append(curr_label)
                curr_idx.append(id)
            result.append(curr_seq)
            index_result.append(curr_idx)
        return result, index_result

    def predict(self, content_list):
        """
        bert ner predict
        :param content_list: content list
        :return: predict result
        """
        ner_result = None
        try:
            tmp_f = list(self._convert_lst_to_features(content_list))

            input_ids = [f.input_ids for f in tmp_f]
            input_masks = [f.input_mask for f in tmp_f]

            pred_result = self.sess.run(self.pred_ids, feed_dict={self.input_ids: input_ids,
                                                                     self.input_mask: input_masks})

            pred_result = self._convert_id_to_label(pred_result, len(pred_result))[0]

            # zip str predict id
            str_pred = []
            for w in zip(content_list, pred_result):
                sub_list = []
                for z in zip(list(w[0]), w[1]):
                    sub_list.append([z[0], z[1]])

                str_pred.append(sub_list)

            # get ner
            ner_result = [self._combine_ner(s) for s in str_pred]

        except Exception as e:
            self.logger.error(e)

        finally:
            return ner_result

    def _combine_ner(self, pred_result):
        """
        combine ner
        :param pred_result: model predict result and origin content words list
        :return: entity words and index
        """
        words_len = len(pred_result)
        i = 0
        tmp = ''
        _ner_list = []

        while i < words_len:
            word = pred_result[i]
            # add personal pronoun
            if word[0] in self.per_proun and word[1][0] == 'O':
                _ner_list.append([word[0], 'PER'])

            if word[1][0] == 'O' and tmp is not '':
                _ner_list.append([tmp, pred_result[i-1][1][2:]])
                tmp = ''

            elif word[1][0] == 'I':
                tmp = tmp + word[0]
                if i == words_len-1:
                    _ner_list.append([tmp, word[1][2:]])

            elif word[1][0] == 'B':
                if tmp is not '':
                    _ner_list.append([tmp, word[1][2:]])

                tmp = word[0]
                if i == words_len-1:
                    _ner_list.append([tmp, word[1][2:]])

            i += 1

        return _ner_list

if __name__ == '__main__':
    str1 = '1995年，湖北农民万其珍应下叔叔万述荣的临终嘱托，成为万家第三代义渡艄公。11年后的2016年，他的儿子万芳权接过父亲手中的船桨，将祖上传下来的“义渡”承诺再次传承。就这样，从万家爷爷，'
    str2 = '两年前，来自上海的“高龄产妇”周月（化名），在香港顺产生下了一个活泼可爱的女儿。这是一个试管婴儿，回想起从在香港检查出内膜移位，到取卵，再到孕育的过程，周月至今仍觉得不可思议。'
    bn = BertNer(gpu_no=0, log_dir='log/', verbose=True, ner_model=r'bert_ner_model\\')
    print(bn.predict(['小张']))
    # req_list = []
    # for i in range(20):
    #     req_list.append(str1)
    #     req_list.append(str2)
    # while True:
    #     print(bn.predict(req_list))