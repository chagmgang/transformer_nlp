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

    dataset = datasets.MaskedLMDataset(
        tokenizer=tokenizer, vocab_size=config['eng_vocab_length'],
        token_length=50)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32,
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
    for epoch in range(10000):

        total_loss = 0
        for step, batch in enumerate(data_loader):
            
            bert_step += 1
            input_ids, segment_ids, masked_pos, masked_tokens = batch
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            masked_pos = masked_pos.to(device)
            masked_tokens = masked_tokens.to(device)

            optimizer.zero_grad()
            logits_lm, _, _ = bert(input_ids, segment_ids, masked_pos)
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
            loss_lm.backward()
            optimizer.step()

            print(bert_step, loss_lm.item()) 
            writer.add_scalar('data/train_loss', loss_lm.item(), bert_step)

if __name__ == '__main__':
    train()
