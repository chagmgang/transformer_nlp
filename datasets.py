import tokenization
import pandas
import torch
import copy

import numpy as np

class MaskedLMDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, text_file,
                 token_length, padding_word='[PAD]',
                 masked_word='[MASK]', mask_ratio=0.15):

        self.tokenizer = tokenizer
        self.text_file = text_file
        self.token_length = token_length
        self.padding_word = padding_word
        self.masked_word = masked_word
        self.mask_ratio = mask_ratio

        self.text = open(self.text_file)
        self.lines = []
        while True:
            line = self.text.readline()
            if not line: break
            self.lines.append(line.replace('\n', ''))
        self.text.close()

    def __getitem__(self, idx):
        text = self.lines[idx]
        
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.token_length]

        masked_idxs = np.arange(len(tokens))
        np.random.shuffle(masked_idxs)
        masked_idxs = masked_idxs[:int(self.mask_ratio * len(tokens))]
        
        original_tokens = copy.deepcopy(tokens)
        masked_tokens = copy.deepcopy(tokens)

        for masked_idx in masked_idxs:
            masked_tokens[masked_idx] = self.masked_word

        while len(original_tokens) < self.token_length:
            original_tokens.append(self.padding_word)
        while len(masked_tokens) < self.token_length:
            masked_tokens.append(self.padding_word)

        masked_pos = np.zeros(self.token_length)
        for masked_idx in masked_idxs:
            masked_pos[masked_idx] = 1
        
        original_tokens = self.tokenizer.convert_tokens_to_ids(original_tokens)
        masked_tokens = self.tokenizer.convert_tokens_to_ids(masked_tokens)

        original_tokens = torch.as_tensor(original_tokens, dtype=torch.long)
        masked_tokens = torch.as_tensor(masked_tokens, dtype=torch.long)
        masked_pos = torch.as_tensor(masked_pos, dtype=torch.long)

        return original_tokens, masked_tokens, masked_pos

    def __len__(self):
        return len(self.lines)

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
