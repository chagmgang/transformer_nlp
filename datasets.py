import tokenization
import pandas
import random
import json
import torch
import copy

import numpy as np

class MaskedLMDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, vocab_size, token_length,
                 padding_word='[PAD]',
                 masked_word='[MASK]', mask_ratio=0.15,
                 max_pred=5):

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.token_length = token_length
        self.padding_word = padding_word
        self.masked_word = masked_word
        self.mask_ratio = mask_ratio
        self.max_pred = max_pred

        self.text = pandas.read_csv('data/emotion.csv')
        self.text = list(self.text['text'])
        self.text = [self.tokenizer.tokenize(text) for text in self.text]
        self.text = [self.tokenizer.convert_tokens_to_ids(text)[:self.token_length] for text in self.text]
        self.mask_idx = self.tokenizer.convert_tokens_to_ids([self.masked_word])[0]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids([self.padding_word])[0]

    def __getitem__(self, idx):
        input_ids = self.text[idx]
        segment_ids = [0] * len(input_ids)
        
        n_pred = min(self.max_pred, max(1, int(round(len(input_ids) * self.mask_ratio))))
        cand_masked_pos = [i for i, token in enumerate(input_ids)]
        np.random.shuffle(cand_masked_pos)
        masked_pos, masked_tokens = [], []
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if np.random.rand() < 0.8:
                input_ids[pos] = self.mask_idx
            elif np.random.rand() < 0.5:
                input_ids[pos] = np.random.choice(self.vocab_size)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([self.pad_idx] * n_pad)
            masked_pos.extend([self.pad_idx] * n_pad)

        n_pad = self.token_length - len(input_ids)
        input_ids.extend([self.pad_idx] * n_pad)
        segment_ids.extend([self.pad_idx] * n_pad)

        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        segment_ids = torch.as_tensor(segment_ids, dtype=torch.long)
        masked_pos = torch.as_tensor(masked_pos, dtype=torch.long)
        masked_tokens = torch.as_tensor(masked_tokens, dtype=torch.long)

        return input_ids, segment_ids, masked_pos, masked_tokens

    def __len__(self):
        return len(self.text)

class BertDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, text_file, vocab_size,
                 token_length, padding_word='[PAD]',
                 sep_word='[SEP]', cls_word='[CLS]',
                 masked_word='[MASK]', mask_ratio=0.15,
                 max_pred=5):

        self.tokenizer = tokenizer
        self.text_file = text_file
        self.vocab_size = vocab_size
        self.token_length = token_length
        self.padding_word = padding_word
        self.sep_word = sep_word
        self.cls_word = cls_word
        self.masked_word = masked_word
        self.mask_ratio = mask_ratio
        self.max_pred = max_pred

        f = open(text_file, 'r')
        self.text = []
        while True:
            line = f.readline()
            if not line: break
            line = line.replace('\n', '').lower()
            tokens = self.tokenizer.tokenize(line)
            self.text.append(tokens)
        f.close()

        self.text = [self.tokenizer.convert_tokens_to_ids(text) for text in self.text]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids([self.padding_word])[0]
        self.sep_idx = self.tokenizer.convert_tokens_to_ids([self.sep_word])[0]
        self.cls_idx = self.tokenizer.convert_tokens_to_ids([self.cls_word])[0]
        self.mask_idx = self.tokenizer.convert_tokens_to_ids([self.masked_word])[0]

        self.data_length = len(self.text)
        self.positive_idx = [(i, i+1) for i in range(self.data_length-1)]
        
        self.negative_idx = []
        while len(self.negative_idx) < len(self.positive_idx):
            first_idx = np.random.choice(len(self.text))
            second_idx = np.random.choice(len(self.text))
            if first_idx + 1 != second_idx and first_idx != second_idx:
                self.negative_idx.append((first_idx, second_idx))

    def item(self, idx, l):
        positive_idx = l[idx]

        ### make positive sample
        tokens_a, tokens_b = self.text[positive_idx[0]], self.text[positive_idx[1]]
        input_ids = [self.cls_idx] + tokens_a + [self.sep_idx] + tokens_b + [self.sep_idx]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        n_pred = min(self.max_pred, max(1, int(round(len(input_ids) * self.mask_ratio))))
        cand_masked_pos = [i for i, token in enumerate(input_ids)
                            if token != self.cls_idx and token != self.sep_idx]
        np.random.shuffle(cand_masked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if np.random.rand() < 0.8:
                input_ids[pos] = self.mask_idx
            elif np.random.rand() < 0.5:
                input_ids[pos] = np.random.choice(self.vocab_size)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([self.pad_idx] * n_pad)
            masked_pos.extend([self.pad_idx] * n_pad)
       
        n_pad = self.token_length - len(input_ids)
        input_ids.extend([self.pad_idx] * n_pad)
        segment_ids.extend([self.pad_idx] * n_pad)

        if positive_idx[0] + 1 == positive_idx[1]:
            isNext = True
        else:
            isNext = False

        return input_ids, segment_ids, masked_tokens, masked_pos, isNext

    def to_long_tensor(self, tensor):
        return torch.as_tensor(tensor, dtype=torch.long)

    def __getitem__(self, idx):
        pos_input_ids, pos_segment_ids, pos_masked_tokens, pos_masked_pos, pos_isNext = self.item(idx, self.positive_idx)
        neg_input_ids, neg_segment_ids, neg_masked_tokens, neg_masked_pos, neg_isNext = self.item(idx, self.negative_idx)

        pos_input_ids = self.to_long_tensor(pos_input_ids)
        pos_segment_ids = self.to_long_tensor(pos_segment_ids)
        pos_masked_tokens = self.to_long_tensor(pos_masked_tokens)
        pos_masked_pos = self.to_long_tensor(pos_masked_pos)
        pos_isNext = self.to_long_tensor(pos_isNext)

        neg_input_ids = self.to_long_tensor(neg_input_ids)
        neg_segment_ids = self.to_long_tensor(neg_segment_ids)
        neg_masked_tokens = self.to_long_tensor(neg_masked_tokens)
        neg_masked_pos = self.to_long_tensor(neg_masked_pos)
        neg_isNext = self.to_long_tensor(neg_isNext)

        return pos_input_ids, pos_segment_ids, pos_masked_tokens, pos_masked_pos, pos_isNext, \
                neg_input_ids, neg_segment_ids, neg_masked_tokens, neg_masked_pos, neg_isNext

    def __len__(self):
        return len(self.positive_idx)

class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, text_file,
                 token_length, start_word='[SOS]',
                 padding_word='[PAD]', end_word='[EOS]'):

        self.tokenizer = tokenizer
        self.text_file = text_file
        self.token_length = token_length
        self.start_word = start_word
        self.padding_word = padding_word
        self.end_word = end_word

        self.text = open(self.text_file)
        self.lines = []
        while True:
            line = self.text.readline()
            if not line: break
            self.lines.append(line.replace('\n', ''))
        self.text.close()

    def __getitem__(self, idx):
        text = self.lines[idx]
        tokens = [self.start_word]
        tokens.extend(self.tokenizer.tokenize(text)[:self.token_length])
        tokens.append(self.end_word)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        while len(input_tokens) < self.token_length:
            input_tokens.append(self.padding_word)
        while len(target_tokens) < self.token_length:
            target_tokens.append(self.padding_word)

        input_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        target_tokens = self.tokenizer.convert_tokens_to_ids(target_tokens)

        input_tokens = torch.as_tensor(input_tokens, dtype=torch.long)
        target_tokens = torch.as_tensor(target_tokens, dtype=torch.long)

        return input_tokens, target_tokens

    def __len__(self):
        return len(self.lines)

class EmotionDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, csv_file, length, padding_word='[PAD]'):

        self.tokenizer = tokenizer
        self.dataframe = pandas.read_csv(csv_file)
        self.length = length
        self.emotion_list = sorted(list(set(self.dataframe['emotions'])))

        self.padding_word = padding_word

    def __getitem__(self, idx):
        text = self.dataframe.loc[idx, ['text']].values[0]
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_text = tokenized_text[:self.length]
        while len(tokenized_text) < self.length:
            tokenized_text.append(self.padding_word)

        ids_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        emotion = self.dataframe.loc[idx, ['emotions']].values[0]
        emotion = self.emotion_list.index(emotion)

        return torch.as_tensor(ids_text, dtype=torch.long), torch.as_tensor(emotion, dtype=torch.long)

    def __len__(self):
        return self.dataframe.shape[0]

class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, src_tokenizer, tgt_tokenizer,
                 file_root, src_column, tgt_column,
                 src_length, tgt_length,
                 start_token='[SOS]', end_token='[EOS]',
                 padding_token='[PAD]'):

        self.dataframe = pandas.read_excel(file_root)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.src_length = src_length
        self.tgt_length = tgt_length

    def __getitem__(self, idx):
        src = self.dataframe.loc[idx, self.src_column]

        tokenized_src = self.src_tokenizer.tokenize(src)
        tokenized_src = tokenized_src[:self.src_length]
        while len(tokenized_src) < self.src_length:
            tokenized_src.append(self.padding_token)
        inp = self.src_tokenizer.convert_tokens_to_ids(tokenized_src)

        tgt_token = [self.start_token]
        tgt = self.dataframe.loc[idx, self.tgt_column]
        tgt = self.tgt_tokenizer.tokenize(tgt)
        tgt_token.extend(tgt)
        tgt_token.append(self.end_token)

        tgt_input = tgt_token[:-1]
        tgt_output = tgt_token[1:]

        tgt_input = tgt_input[:self.tgt_length]
        tgt_output = tgt_output[:self.tgt_length]
        
        while len(tgt_input) < self.tgt_length + 1:
            tgt_input.append(self.padding_token)
        while len(tgt_output) < self.tgt_length + 1:
            tgt_output.append(self.padding_token)

        tgt_input = self.tgt_tokenizer.convert_tokens_to_ids(tgt_input)
        tgt_output = self.tgt_tokenizer.convert_tokens_to_ids(tgt_output)

        inp = torch.as_tensor(inp, dtype=torch.long)
        tgt_input = torch.as_tensor(tgt_input, dtype=torch.long)
        tgt_output = torch.as_tensor(tgt_output, dtype=torch.long)

        return inp, tgt_input, tgt_output

    def __len__(self):
        return self.dataframe.shape[0]
