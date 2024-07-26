import tiktoken


# Obtain the tokenizer specific for the GPT-4 model
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Text to tokenize
text = "Aquest és un text en català que vull tokenitzar."

# Tokenize the text
tokens = tokenizer.encode(text)

# Decode the tokens to text
decoded_text = tokenizer.decode(tokens)

print(f"Tokens: {tokens}")
print(f"Text decodificat: {decoded_text}")
print(f"Llargada del text: {len(decoded_text)}")