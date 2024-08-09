import streamlit as st
import torch
from tokenizers import ByteLevelBPETokenizer
import torch.nn.functional as F
from CatGPT_model import GPT, GPTConfig

# In order to run this code just exectue the CatGPT_run.py file


def load_model(model, model_path, device='cpu'):
    """
    Load the model state dictionary from the specified path and load it into the model.

    Parameters:
    model (torch.nn.Module): The model into which the state dictionary will be loaded.
    model_path (str): The path from where the model will be loaded.
    device (torch.device): The device on which the model will be loaded.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {model_path}")

@st.cache_resource
def get_model():
    model = GPT(GPTConfig(vocab_size=32768))
    model = torch.compile(model)
    load_model(model, model_path="models/CatGPT.pth")
    return model

def generate_text(model, input_text='La intel·ligència artificial tindrà la capacitat de', num_return_sequences=1, max_length=100, device='cpu'):
    enc = ByteLevelBPETokenizer(
        'tokenizer/vocab.json',
        'tokenizer/merges.txt'
    )

    # Encode the input text
    tokens = enc.encode(input_text).ids
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)

    if len(tokens) > max_length:
        max_length = len(tokens) + 50
        print(f"Max length set to {max_length} as input text is longer than the previous max length")
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (B, 8)
    x = tokens.to(device)

    # Generate sequences
    with torch.no_grad():
        for _ in range(max_length - x.size(1)):
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
            x = torch.cat((x, next_token), dim=1)

    # Decode and print the generated sequences
    generated_texts = []
    for i in range(num_return_sequences):
        tokens = x[i].tolist()
        decoded = enc.decode(tokens)
        generated_texts.append(decoded)
    return generated_texts

# Streamlit interface
st.title('CatGPT Model')
st.write('Generate text using your CatGPT model')

input_text = st.text_area('Input Text', 'La intel·ligència artificial tindrà la capacitat de')
num_return_sequences = st.number_input('Number of Sequences', min_value=1, max_value=5, value=1)
max_length = st.slider('Max Length', min_value=35, max_value=300, value=50)

model = get_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if st.button('Generate'):
    with st.spinner('Generating...'):
        generated_texts = generate_text(model, input_text, num_return_sequences, max_length, device)
    for i, text in enumerate(generated_texts):
        st.write(f'Sample {i+1}:')
        st.write(text)