import json
import torch
import tokenization
import datasets

from transformer.model import BertModel

def bert_masked_lm_loss(logits, original_tokens, masked_poses, criterion):
    masked_tokens = []
    masked_logits = []
    batch, seq_length = masked_poses.shape
    for i in range(batch):
        for j in range(seq_length):
            if masked_poses[i, j] == 1:
                masked_tokens.append(original_tokens[i, j])
                masked_logits.append(logits[i, j])
    masked_tokens = torch.stack(masked_tokens)
    masked_logits = torch.stack(masked_logits)

    loss = criterion(masked_logits, masked_tokens)
    return loss

def train():

    config = json.load(open('config.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/kor_vocab.txt', do_lower_case=False)
    dataset = datasets.MaskedLMDataset(
        tokenizer=tokenizer, text_file='data/sample.txt',
        token_length=60, padding_word=config['pad_word'],
        masked_word=config['mask_word'],
        mask_ratio=0.15)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=4,
        num_workers=1, shuffle=True)

    pad_index = tokenizer.convert_tokens_to_ids([config['pad_word']])[0]

    bert = BertModel(
        vocab_size=config['kor_vocab_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'], d_k=config['d_k'],
        d_v=config['d_v'], n_heads=config['num_heads'],
        n_layers=config['num_layers'], pad_index=pad_index,
        device=device).to(device)

    optimizer = torch.optim.Adam(bert.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.8)

    for epoch in range(100):

        for step, (original_tokens, masked_tokens, masked_pos) in enumerate(data_loader):
            original_tokens = original_tokens.to(device)
            masked_tokens = masked_tokens.to(device)
            masked_pos = masked_pos.to(device)

            optimizer.zero_grad()
            logits, _, _ = bert(masked_tokens)
            loss = bert_masked_lm_loss(logits, original_tokens, masked_pos, criterion)
            loss.backward()
            optimizer.step()

            print(epoch, step, loss.item(), scheduler.get_lr()[0])

if __name__ == '__main__':
    train()