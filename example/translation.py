import six
import json
import torch
import tokenization
import datasets

from transformer.model import Translation

def train():

    config = json.load(open('config.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/kor_vocab.txt', do_lower_case=False)
    tgt_tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/eng_vocab.txt', do_lower_case=False)

    src_pad_index = src_tokenizer.convert_tokens_to_ids([config['pad_word']])[0]
    tgt_pad_index = tgt_tokenizer.convert_tokens_to_ids([config['pad_word']])[0]

    transformer = Translation(
        src_vocab_size=config['kor_vocab_length'],
        tgt_vocab_size=config['eng_vocab_length'],
        d_model=config['d_model'], d_ff=config['d_ff'],
        d_k=config['d_k'], d_v=config['d_v'], n_heads=config['num_heads'], 
        n_layers=config['num_layers'], src_pad_index=src_pad_index,
        tgt_pad_index=src_pad_index, device=device).to(device)

    dataset = datasets.TranslationDataset(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_length=40,
        tgt_length=40,
        file_root='data/translation.xlsx',
        src_column='원문',
        tgt_column='번역문')

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, num_workers=4,
        batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)

    for epoch in range(4):

        total_loss = 0
        for step, (enc_inputs, dec_inputs, target_batch) in enumerate(dataloader):
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(enc_inputs, dec_inputs)
            loss = criterion(
                dec_logits.view(-1, dec_logits.size(-1)),
                target_batch.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print('----------------------')
            print(f'epoch : {epoch}')
            print(f'step  : {step / len(dataloader)}')
            print(f'loss  : {loss.item()}')
            print(f'lr    : {scheduler.get_lr()[0]}')

        scheduler.step()
        print('----------------------')
        print(f'epoch : {epoch}')
        print(f'loss  : {total_loss / len(dataloader)}')
        print(f'lr    : {scheduler.get_lr()[0]}')

    test_sentences = [
        '6.5, 7, 8 사이즈가 몇 개나 더 재입고 될지 제게 알려주시면 감사하겠습니다.',
        '씨티은행에서 일하세요?',
        '푸리토의 베스트셀러는 해외에서 입소문만으로 4차 완판을 기록하였다.',
        '11장에서는 예수님이 이번엔 나사로를 무덤에서 불러내어 죽은 자 가운데서 살리셨습니다.',
        'F/W 겐조타이거 키즈와 그리고 이번에 주문한 키즈 중 부족한 수량에 대한 환불입니다.',
        '강아지들과 내 사진을 보낼게.']

    for test_sentence in test_sentences:
        orig_text = test_sentence
        test_sentence = src_tokenizer.tokenize(test_sentence)
        test_sentence_ids = src_tokenizer.convert_tokens_to_ids(test_sentence)
        enc_token = torch.as_tensor([test_sentence_ids], dtype=torch.long)

        test_sentence_dec = ['[SOS]']
        test_sentence_dec = tgt_tokenizer.convert_tokens_to_ids(test_sentence_dec)
        eos_flag = tgt_tokenizer.convert_tokens_to_ids(['[EOS]'])

        while test_sentence_dec[-1] != eos_flag[0]:
            dec_input = torch.as_tensor([test_sentence_dec], dtype=torch.long)
            enc_token, dec_input = enc_token.to(device), dec_input.to(device)
            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(enc_token, dec_input)
            predict = torch.argmax(dec_logits, axis=2)[:, -1].squeeze().detach().cpu().numpy()
            test_sentence_dec.append(int(predict))

        predict_text = ' '.join(tgt_tokenizer.convert_ids_to_tokens(test_sentence_dec))
        predict_text = predict_text.replace(" ##", "")
        predict_text = predict_text.replace("##", "")
        print(f'orig_text    : {orig_text}')
        print(f'predict_text : {predict_text}')
        print('-----------------')

if __name__ == '__main__':
    train()
