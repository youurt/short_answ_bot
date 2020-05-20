# shortanswerbot

[A Telegram bot](https://t.me/short_answer_bot) which gives a simple answer for simple question.

![Chat](Bildschirmfoto 2020-05-20 um 14.30.45.png)

## How?

[Google Colab Version](https://colab.research.google.com/drive/19nBpPDxtibZ6DTKGPVBGH5DXIFhpHEnY#scrollTo=hQb3M4Fi8O9P)

With the help of the wikipedia api, telegram, a pretrained BERT model, transformers and spacy for detecting the user intent. User asks for "Who is Lebron James?" and the script detects this Person like so:

```bash
Lebron James [PRS]
```

## BERT

Then the script uses the summary of that wikipedia article and takes the first 512 Tokens of the text. With a pretrained&finetuned BERT model it generates the answer from the wikipedia passage.

```python
model = AutoModelForQuestionAnswering.from_pretrained(
    "mrm8488/bert-medium-finetuned-squadv2")
tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/bert-medium-finetuned-squadv2")
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
