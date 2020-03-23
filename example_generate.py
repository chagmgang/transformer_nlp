import json
import torch
import datasets
import tokenization

from transformer.model import GPTModel

def train():
    config = json.load(open('config.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/kor_vocab.txt', do_lower_case=False)

    dataset = datasets.GPTDataset(
        tokenizer=tokenizer,
        text_file='data/sample.txt',
        token_length=100,
        start_word=config['start_word'],
        end_word=config['end_word'],
        padding_word=config['pad_word'])

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=4, num_workers=4,
        shuffle=True)

    pad_index = tokenizer.convert_tokens_to_ids([config['pad_word']])[0]

    generator = GPTModel(
        vocab_size=config['kor_vocab_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        d_k=config['d_k'], d_v=config['d_v'],
        n_heads=config['num_heads'], n_layers=config['num_layers'],
        pad_index=pad_index, device=device).to(device)

    optimizer = torch.optim.Adam(generator.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    for epoch in range(200):

        total_loss = 0
        for step, (dec_inputs, dec_outputs) in enumerate(dataloader):
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)

            optimizer.zero_grad()
            dec_logits, attns = generator(dec_inputs)
            loss = criterion(
                dec_logits.view(-1, dec_logits.size(-1)),
                dec_outputs.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('-------------')
        print(f'epoch : {epoch}')
        print(f'loss  : {total_loss}')
        print(f'lr    : {scheduler.get_lr()[0]}')
        
        scheduler.step()

    f = open('data/sample.txt')
    test_sentences = []
    while True:
        line = f.readline()
        if not line: break
        test_sentences.append(line.replace('\n', ''))
    f.close()

    generator.eval()

    eos_flag = tokenizer.convert_tokens_to_ids([dataset.end_word])[0]
    for test_sentence in test_sentences:
        tokens = [dataset.start_word]
        tokens.extend(tokenizer.tokenize(test_sentence)[:10])
        test_sentence_dec = tokenizer.convert_tokens_to_ids(tokens)

        while test_sentence_dec[-1] != eos_flag:
            dec_inputs = torch.as_tensor([test_sentence_dec], dtype=torch.long).to(device)
            dec_logits, _ = generator(dec_inputs)
            predict = torch.argmax(dec_logits, axis=2)[:, -1].squeeze().detach().cpu().numpy()
            test_sentence_dec.append(int(predict))
        print('----------------------')
        print(f'input text : {tokens}')
        predict_text = ' '.join(tokenizer.convert_ids_to_tokens(test_sentence_dec))
        predict_text = predict_text.replace(" ##", "")
        predict_text = predict_text.replace("##", "")
        print(f'predict_text : {predict_text}')

if __name__ == '__main__':
    train()
