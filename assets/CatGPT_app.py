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


def generate_text(model, input_text='La intel·ligència artificial tindrà la capacitat de', num_return_sequences=1, max_length=100, device='cpu', temperature=1.0, top_k=10, repetition_penalty=1.2):
    enc = ByteLevelBPETokenizer(
        'tokenizer/vocab.json',
        'tokenizer/merges.txt'
    )

    # Encode the input text
    tokens = enc.encode(input_text).ids
    tokens = torch.tensor(tokens, dtype=torch.long)

    if len(tokens) > max_length:
        max_length = len(tokens) + 50
        print(f"Max length set to {max_length} as input text is longer than the previous max length")

    generated_texts = []
    for _ in range(num_return_sequences):
        x = tokens.unsqueeze(0).to(device)

        # Generate sequence
        with torch.no_grad():
            for _ in range(max_length - x.size(1)):
                logits, _ = model(x)
                logits = logits[:, -1, :]

                # Apply temperature scaling
                logits = logits / temperature

                # Apply repetition penalty
                if x.size(1) > 1:
                    for token in set(x[0].tolist()):
                        logits[0, token] /= repetition_penalty

                # Apply Top-k sampling
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    probs = F.softmax(top_k_probs, dim=-1)
                    next_token = top_k_indices.gather(dim=-1, index=torch.multinomial(probs, 1))
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                x = torch.cat((x, next_token), dim=1)

        # Decode and store the generated sequence
        decoded = enc.decode(x[0].tolist())
        generated_texts.append(decoded)

    return generated_texts


# Streamlit interface
st.title('CatGPT Model')
st.write('Generate text using your CatGPT model')

input_text = st.text_area('Input Text', 'La intel·ligència artificial tindrà la capacitat de')
num_return_sequences = st.number_input('Number of Sequences', min_value=1, max_value=5, value=1)

# Arrange sliders horizontally
col1, col2, col3, col4 = st.columns(4)

with col1:
    max_length = st.slider('Max Length', min_value=35, max_value=300, value=50)

with col2:
    temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=1.0, step=0.1)

with col3:
    top_k = st.slider('Top-k', min_value=0, max_value=50, value=1)

with col4:
    repetition_penalty = st.slider('Repetition Penalty', min_value=1.0, max_value=2.0, value=1.2, step=0.1)

model = get_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if st.button('Generate'):
    with st.spinner('Generating...'):
        generated_texts = generate_text(model, input_text, num_return_sequences, max_length, device, temperature, top_k, repetition_penalty)
    for i, text in enumerate(generated_texts):
        st.write(f'Sample {i+1}:')
        st.write(text)