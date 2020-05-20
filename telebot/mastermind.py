import spacy
import wikipedia
import torch
# from transformers import BertForQuestionAnswering
# from transformers import BertTokenizer
# model = BertForQuestionAnswering.from_pretrained('mrm8488/bert-tiny-finetuned-squadv2')
# tokenizer = BertTokenizer.from_pretrained('mrm8488/bert-tiny-finetuned-squadv2')


from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
model = AutoModelForQuestionAnswering.from_pretrained(
    "mrm8488/bert-medium-finetuned-squadv2")
tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/bert-medium-finetuned-squadv2")


def topic(question):
    """
    takes msg input and makes nlp tokenization. 
    then it generates the user intent which is either an entity or a Noun
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)

    for ent in doc.ents:
        return ent.text
    for token in doc:
        if token.pos_ == 'NOUN':
            return token.text
    return False


def wikipedia_search(question):
    """
    searches wikipedia for the intent and generates an answer text
    """

    phrase = topic(question)
    wiki_answer = wikipedia.summary(phrase, auto_suggest=False)

    return wiki_answer


def answer_question(question):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========

    answer_text = wikipedia_search(question)
    input_ids = tokenizer.encode(question, answer_text, max_length=512)

    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # ======== Set Segment IDs ========
    sep_index = input_ids.index(tokenizer.sep_token_id)


    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b


    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========

    start_scores, end_scores = model(torch.tensor(
        [input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # ======== Reconstruct Answer ========

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

  
    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):

       
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

     
        else:
            answer += ' ' + tokens[i]

    return answer

