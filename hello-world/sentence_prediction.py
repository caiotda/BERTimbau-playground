import torch

from transformers import pipeline
from transformers import BertTokenizer
from transformers import AutoModelForMaskedLM
# Import model with Language masking attached. Not sure if it was trained with
# Language masking.

# Note: BERTimbau was fine tuned for NER, Sentence entangling and sentence 
# similarity.


def get_top3_predicts(predict):
    top3 = predict[:3]
    s = ""
    for idx, prediction in enumerate(top3):
        seq = prediction['sequence']
        score = format(prediction['score']*100, '.2f')
        s += f'{idx+1} - {seq}; Probability: {score}%\n'
    return s

DEBUG = True

model = AutoModelForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

batch = ["Meu carro é [MASK].", "Hoje o dia esta [MASK].", "[MASK], nao e?",
                "[MASK] conversou com o presidente hoje pela manha.", "O presidente do Brasil é [MASK]."]
if not DEBUG:
    batch = []
    print("Digite [MASK] sempre que quiser prever algo. Ao terminar de digitar as\
        frases, aperte enter")
    phrase = input("Digite as frase que você quer prever: \n> ")
    while (phrase != ''):
        batch.append(phrase)
        phrase = input("> ")

input_tok = tokenizer.encode(batch, padding=True, return_tensors="pt")
pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

for phrase in batch:
    predict = pipe(phrase)
    out = get_top3_predicts(predict)
    print(f"Phrase: {phrase}")
    print("Prediction: \n",out, sep="")

