# Transformer related Model in pytorch

* Transformer Encoder Emotion Classifier
* Transformer Translation Model
* Transformer GPT2
* Transformer BERT

## Data

### Dataset From
* Translation Data from [AIHUB](http://www.aihub.or.kr/aidata/87/download)
* Emotion Classification Data from [Kaggle](https://www.kaggle.com/eray1yildiz/emotion-classification)
* GPT2 Data from Some News in [Naver News](https://news.naver.com/)

### Dataset Shape

* bert sample txt file in `data/bert_sample.txt`
* format
```
The rain had only ceased with the gray streaks of morning at Blazing Star, and the settlement awoke to a moral sense of cleanliness, and the finding of forgotten knives, tin cups, and smaller camp utensils, where the heavy showers had washed away the debris and dust heaps before the cabin doors.
Indeed, it was recorded in Blazing Star that a fortunate early riser had once picked up on the highway a solid chunk of gold quartz which the rain had freed from its incumbering soil, and washed into immediate and glittering popularity.
Possibly this may have been the reason why early risers in that locality, during the rainy season, adopted a thoughtful habit of body, and seldom lifted their eyes to the rifted or india-ink washed skies above them.
"Cass" Beard had risen early that morning, but not with a view to discovery.
```

* emotion csv file in `data/emotion.csv`
* format
```
,text,emotions
27383,i feel awful about it too because it s my job to get him in a position to succeed and it just didn t happen here,sadness
110083,im alone i feel awful,sadness
140764,ive probably mentioned this before but i really do feel proud of myself for actually keeping up with my new years resolution of monthly and weekly goals,joy
100071,i was feeling a little low few days back,sadness
```

* sample txt file in `data/sample.txt`
* format

```
대한항공은 인천-베이징 노선 운항과 관련해 이번 주 예정된 23일, 25일, 27일 항공편을 정상 운항하고 28일부터 4월 25일까지 잠정 중단할 예정이다.
이에 그치지 않고 최소 400이닝 이상 던진 투수 중 평균 게임 스코어가 63.0으로 샌디 쿠팩스(62.8)를 넘어 역대 1위다. 4번의 평균자책점 타이틀, 6번의 사이영상 파이널 후보, 그 중 3번의 수상, 현역 최다 15완봉승 기록은 커쇼를 역대 최고 투수로 주저하지 않게 만든다. 
```

* translation xlsx file in `data/translation.xlsx`
* format

| SID  |  원문 | 번역문  | 
|---|---|---|
|  1 | 'Bible Coloring'은 성경의 아름다운 이야기를 체험 할 수 있는 컬러링 앱입니다.  |  Bible Coloring' is a coloring application that allows you to experience beautiful stories in the Bible. | 
|  2 | 씨티은행에서 일하세요?  |  Do you work at a City bank? |
|  3 |  11장에서는 예수님이 이번엔 나사로를 무덤에서 불러내어 죽은 자 가운데서 살리셨습니다. |In Chapter 11 Jesus called Lazarus from the tomb and raised him from the dead.   |

## Tokenizer

* Tokenizer from [google bert](https://github.com/google-research/bert)
* Korean Vocabulary file `vocab/kor_vocab.txt`
* English Vocabulary file `vocab/eng_vocab.txt`

```python
import tokenization
kor_tokenizer = tokenization.FullTokenizer(
    vocab_file='vocab/kor_vocab.txt', do_lower_case=False)
eng_tokenizer = tokenization.FullTokenizer(
    vocab_file='vocab/eng_vocab.txt', do_lower_case=False)

kor_tokenizer.tokenize('대한항공은 인천-베이징 노선 운항과 관련해 이번 주 예정된 23일, 25일, 27일 항공편을 정상 운항하고 28일부터 4월 25일까지 잠정 중단할 예정이다.')

>>> ['대한항공', '##은', '인천', '-', '베이징', '노선', '운항', '##과', '관련', '##해', '이번', '주', '예정', '##된', '23', '##일', ',', '25', '##일', ',', '27', '##일', '항공', '##편을', '정상', '운항', '##하고', '28', '##일부', '##터', '4', '##월', '25', '##일', '##까지', '잠', '##정', '중단', '##할', '예정', '##이다', '.']

eng_tokenizer.tokenize('In Chapter 11 Jesus called Lazarus from the tomb and raised him from the dead.')

>>> ['In', 'Chapter', '11', 'Jesus', 'called', 'Lazarus', 'from', 'the', 'tomb', 'and', 'raised', 'him', 'from', 'the', 'dead', '.']
```

## How to Configure Model

* config file in `config.json`
```json
{
    "kor_vocab_length": 50000,
    "eng_vocab_length": 28998,
    "d_model": 768,
    "d_ff": 2048,
    "d_k": 64,
    "d_v": 64,
    "num_layers": 12,
    "num_heads": 8,
    "start_word": "[SOS]",
    "end_word": "[EOS]",
    "sep_word": "[SEP]",
    "cls_word": "[CLS]",
    "pad_word": "[PAD]",
    "mask_word": "[MASK]"
}
```

## Test Command

* Emotion Classification
```
python example/classifier.py
```

* Text Generation
```
python example/generate.py
```

* Translation
```
python example/translation.py
```

* BERT Encoding
```
python example/bert.py
```

## Reference

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [nlp-tutorial](https://github.com/graykode/nlp-tutorial)
* [google-bert](https://github.com/google-research/bert)
* [illustrated-bert](http://jalammar.github.io/illustrated-bert/)
* [illustrated-gpt2](http://jalammar.github.io/illustrated-gpt2/)
