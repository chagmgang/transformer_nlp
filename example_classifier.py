import json
import torch
import tokenization
import datasets

from transformer.model import Classifier

def train():

    config = json.load(open('config.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/eng_vocab.txt',
        do_lower_case=False)
    dataset = datasets.EmotionDataset(
        tokenizer=tokenizer,
        csv_file='data/emotion.csv',
        length=30,
        padding_word=config['pad_word'])

    train_length = int(len(dataset) * 0.8)
    valid_length = len(dataset) - train_length
    batch_size = 128
    num_worker = 4

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_length, valid_length])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        num_workers=num_worker, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size,
        num_workers=num_worker, shuffle=False)

    pad_index = tokenizer.convert_tokens_to_ids([config['pad_word']])[0]
    classifier = Classifier(
        vocab_size=config['eng_vocab_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'], d_k=config['d_k'],
        d_v=config['d_v'], n_heads=config['num_heads'],
        n_layers=config['num_layers'], pad_index=pad_index,
        device=device, num_classes=len(dataset.emotion_list)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

    train_step = 0

    for epoch in range(3):

        for step, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)

            train_step += 1
            optimizer.zero_grad()
            logits, _ = classifier(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            preds = preds.eq(labels).sum().item() / texts.shape[0]

            print('train')
            print(f'epoch : {epoch}')
            print(f'loss  : {loss.item()}')
            print(f'acc   : {preds}')
            print(f'lr    : {scheduler.get_lr()[0]}')
            print('------------')

        total_loss = 0
        total_acc = 0
        for step, (texts, labels) in enumerate(valid_loader):
            texts, labels = texts.to(device), labels.to(device)

            with torch.no_grad():
                logits, _ = classifier(texts)
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
                preds = preds.eq(labels).sum()

                total_loss += loss.item()
                total_acc += preds.item()

        print('valid')
        print(f'epoch : {epoch}')
        print(f'loss  : {total_loss / len(valid_loader)}')
        print(f'acc   : {total_acc / len(valid_dataset)}')
        print('------------')
        scheduler.step()

if __name__ == '__main__':
    train()
