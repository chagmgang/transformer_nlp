import json
import torch
import tokenization
import datasets

from transformer.model import BertModel
from torch.utils.tensorboard import SummaryWriter

def train():

    config = json.load(open('config.json'))

    writer = SummaryWriter('runs/bert')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/eng_vocab.txt', do_lower_case=False)

    dataset = datasets.BertDataset(
        tokenizer=tokenizer, text_file='data/bert_sample.txt',
        vocab_size=config['eng_vocab_length'], token_length=128,
        padding_word=config['pad_word'], sep_word=config['sep_word'],
        cls_word=config['cls_word'], masked_word=config['mask_word'],
        mask_ratio=0.15)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=8,
        num_workers=4, shuffle=True)

    pad_index = tokenizer.convert_tokens_to_ids([config['pad_word']])[0]

    bert = BertModel(
        vocab_size=config['eng_vocab_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'], d_k=config['d_k'],
        d_v=config['d_v'], n_heads=config['num_heads'],
        n_layers=config['num_layers'], pad_index=pad_index,
        device=device).to(device)

    optimizer = torch.optim.Adam(bert.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    bert_step = 0
    for epoch in range(400):

        total_loss = 0
        for step, batch in enumerate(data_loader):

            pos_input_ids, pos_segment_ids, pos_masked_tokens, pos_masked_pos, pos_isNext, \
                neg_input_ids, neg_segment_ids, neg_masked_tokens, neg_masked_pos, neg_isNext = batch
            
            pos_input_ids = pos_input_ids.to(device)
            pos_segment_ids = pos_segment_ids.to(device)
            pos_masked_tokens = pos_masked_tokens.to(device)
            pos_masked_pos = pos_masked_pos.to(device)
            pos_isNext = pos_isNext.to(device)

            neg_input_ids = neg_input_ids.to(device)
            neg_segment_ids = neg_segment_ids.to(device)
            neg_masked_tokens = neg_masked_tokens.to(device)
            neg_masked_pos = neg_masked_pos.to(device)
            neg_isNext = neg_isNext.to(device)
            
            optimizer.zero_grad()
            pos_logits_lm, pos_logits_clsf, _ = bert(pos_input_ids, pos_segment_ids, pos_masked_pos)
            neg_logits_lm, neg_logits_clsf, _ = bert(neg_input_ids, neg_segment_ids, neg_masked_pos)

            pos_loss_lm = criterion(pos_logits_lm.transpose(1, 2), pos_masked_tokens)
            neg_loss_lm = criterion(neg_logits_lm.transpose(1, 2), neg_masked_tokens)

            pos_loss_clsf = criterion(pos_logits_clsf, pos_isNext)
            neg_loss_clsf = criterion(neg_logits_clsf, neg_isNext)

            loss = pos_loss_lm + neg_loss_lm + pos_loss_clsf + neg_loss_clsf
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print(epoch, total_loss)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1,
        num_workers=4, shuffle=True)

    pos_input_ids, pos_segment_ids, pos_masked_tokens, pos_masked_pos, pos_isNext, \
        neg_input_ids, neg_segment_ids, neg_masked_tokens, neg_masked_pos, neg_isNext = next(iter(data_loader))
    
    pos_input_ids = pos_input_ids.to(device)
    pos_segment_ids = pos_segment_ids.to(device)
    pos_masked_tokens = pos_masked_tokens.to(device)
    pos_masked_pos = pos_masked_pos.to(device)
    pos_isNext = pos_isNext.to(device)

    neg_input_ids = neg_input_ids.to(device)
    neg_segment_ids = neg_segment_ids.to(device)
    neg_masked_tokens = neg_masked_tokens.to(device)
    neg_masked_pos = neg_masked_pos.to(device)
    neg_isNext = neg_isNext.to(device)

    pos_logits_lm, pos_logits_clsf, _ = bert(pos_input_ids, pos_segment_ids, pos_masked_pos)
    neg_logits_lm, neg_logits_clsf, _ = bert(neg_input_ids, neg_segment_ids, neg_masked_pos)

    print('--------')
    print('--------')
    print(torch.argmax(pos_logits_lm, axis=2))
    print(pos_masked_tokens)
    print('--------')

    print(torch.argmax(pos_logits_clsf, axis=1))
    print(pos_isNext)
    print('--------')
    print('--------')

    print(torch.argmax(neg_logits_lm, axis=2))
    print(neg_masked_tokens)
    print('--------')
    
    print(torch.argmax(neg_logits_clsf ,axis=1))
    print(neg_isNext)
    print('--------')
    print('--------')

if __name__ == '__main__':
    train()
