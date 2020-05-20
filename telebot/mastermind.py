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
    # Apply the tokenizer to the input text, treating them as a text-pair.
    answer_text = wikipedia_search(question)
    input_ids = tokenizer.encode(question, answer_text, max_length=512)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor(
        [input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return(answer)

