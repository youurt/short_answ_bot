# shortanswerbot

[A Telegram bot](https://t.me/short_answer_bot) which gives a simple answer for simple question.

![Chat](https://github.com/youurt/short_answ_bot/blob/master/pic.png)

## How?

[Google Colab Version](https://colab.research.google.com/drive/19nBpPDxtibZ6DTKGPVBGH5DXIFhpHEnY#scrollTo=hQb3M4Fi8O9P)

With the help of the wikipedia api, telegram, a pretrained BERT model, transformers and spacy for detecting the user intent. Flask app hosted on heroku.

## BERT

With a pretrained & finetuned BERT model it generates the answer from the wikipedia passage.

```python
model = AutoModelForQuestionAnswering.from_pretrained(
    "mrm8488/bert-medium-finetuned-squadv2")
tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/bert-medium-finetuned-squadv2")
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
