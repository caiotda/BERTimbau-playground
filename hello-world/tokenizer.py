from transformers import AutoTokenizer  # Or BertTokenizer

model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)


# This outputs the input string as input ids, mask tokens and token 
# type ids
sequence = "Primeiro exemplo utilizando o BERTimbau :)"
token_simples = tokenizer(sequence)
print("Exemplo simples:", token_simples)

# Encoding and decoding process example


# The tokenizer essentially splits the input into encoded tokens
# and then transforms them into token id's. Lets look the tokens

# input to tokens
tokens = tokenizer.tokenize(sequence)
print(tokens)

teste = "Testando com outras palavras mais derivativas: carro, carrão, carrinho, cachorro, cachorros."
tokens_longo = tokenizer.tokenize(teste)
print("---------\n", tokens_longo)


# To ids

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)


# Decoding

decoded = tokenizer.decode(ids)
print(decoded)
