import torch

from transformers import pipeline
from transformers import BertTokenizer
from transformers import AutoModelForPreTraining
# Model was pretrained on Language Masking, so we can just import the automodel
# for pre training.
DEBUG = True

model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

batch = ["Meu carro e [MASK]", "Hoje o dia esta [MASK].", "[MASK], nao e?",
                "[MASK] conversou com o presidente hoje pela manha."]
if not DEBUG:
    batch = []
    print("Digite [MASK] sempre que quiser prever algo. Ao terminar de digitar as\
        frases, aperte enter")
    phrase = input("Digite as frase que você quer prever: \n> ")
    while (phrase != ''):
        batch.append(phrase)
        phrase = input("> ")
        
input_tok = tokenizer.encode(batch, padding=True, return_tensors="pt")
with torch.no_grad():
    outs = model(input_tok)
    encoded = outs[0][0, 1:-1]  # Ignore [CLS] and [SEP] special tokens

print(encoded)