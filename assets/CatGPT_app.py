import streamlit as st
import torch
from tokenizers import ByteLevelBPETokenizer
import torch.nn.functional as F
from CatGPT_model import GPT, GPTConfig
import random
import time

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

st.set_page_config(page_title="CatGPT")

# Use columns to place the logo next to the title
col1, col2 = st.columns([3, 1])

with col1:
    st.title('CatGPT Model')
    st.write('Generate text using your CatGPT model')

with col2:
    logo_path = "logo/CatGPT_round.png"
    st.image(logo_path, width=150)



random_inputs = ["La intel·ligència artificial tindrà la capacitat de",
                   "El 23 d'abril, dia de Sant Jordi, els carrers de Catalunya s'omplen de",
                   "Durant la Guerra Civil Espanyola, Catalunya va ser un bastió de resistència republicana perquè",
                   "Els Bitcoin i altres criptomonedes s'han convertit en temes importants a Catalunya, especialment després de",
                   "La meva casa és un lloc molt acollidor",
                   "El clima mediterrani permet gaudir de llargues jornades assolellades a la vora del mar.",
                   "El sol es ponia en l'horitzó de milers de colors però els pirates no podien"]


# Check if an input_text is already selected in session_state
if 'input_text' not in st.session_state:
    st.session_state.input_text = random.choice(random_inputs)

input_text = st.text_area('Input Text', st.session_state.input_text)

num_return_sequences = st.number_input('Number of Sequences', min_value=1, max_value=5, value=1)

# Arrange sliders horizontally
col1, col2, col3, col4 = st.columns(4)

with col1:
    max_length = st.slider('Max Length', min_value=35, max_value=200, value=75)

with col2:
    temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=1.0, step=0.1)

with col3:
    top_k = st.slider('Top-k', min_value=0, max_value=5, value=1)

with col4:
    repetition_penalty = st.slider('Repetition Penalty', min_value=1.0, max_value=2.0, value=1.2, step=0.1)

model = get_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Tips for generating text

tips = [
    "Tip: Use temperature to control the randomness of predictions. Lower values make the model more deterministic.",
    "Tip: Adjust top-k to limit the sampling pool to the top-k probable next tokens.",
    "Tip: Use repetition penalty to avoid generating repetitive sequences.",
    "Tip: Experiment with different max_length values to control the length of the output.",
    "Tip: In order to generate more creative text, try different input prompts.",
    "Tip: As the max_length increases, there will be a higher chance of generating repetitive text or hallucinations. Adjust repetition penalty accordingly.",
    "Tip: The model may generate text that is not coherent or relevant to the input prompt. Try different prompts to get better results.",
    "Tip: Adjust top-k to 0 to allow sampling from the entire vocabulary distribution.",
    "Tip: Use a higher temperature to generate more diverse and creative text but remember to set a top_k different to 1.",
    "Tip: Providing a specific and well-defined prompt can improve the relevance of the generated text."
]


if st.button('Generate'):
    with st.spinner('Generating...'):
        # Select a random tip to display
        selected_tip = random.choice(tips)
        tip_placeholder = st.info(selected_tip)  # Show the tip
        generated_texts = generate_text(model, input_text, num_return_sequences, max_length=max_length, device=device, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty)
        tip_placeholder.empty()  # Delete the tip after generating the text
    for i, text in enumerate(generated_texts):
        st.write(f'Sample {i+1}:')
        st.write(text)