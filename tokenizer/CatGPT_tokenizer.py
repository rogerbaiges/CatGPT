import os
from tokenizers import ByteLevelBPETokenizer

# Initialize the ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the limited dataset
tokenizer.train(files=['data/small_catalan_oscar.txt'], vocab_size=32768, min_frequency=2, special_tokens=[
    "<s>",  # Start of sequence token
    "<pad>",  # Padding token
    "</s>",  # End of sequence token
    "<unk>",  # Unknown token
    "<mask>",  # Mask token for masked language modeling
])

# Save the trained tokenizer model
tokenizer.save_model('tokenizer')

print("Tokenizer successfully trained and saved.")


# Try the tokenizer on a sample text

# Load the tokenizer
tokenizer = ByteLevelBPETokenizer(
    './tokenizer/vocab.json',
    './tokenizer/merges.txt'
)   

# Encode a sample text
sample_text = "Aquesta és una frase de prova."
encoded = tokenizer.encode(sample_text)