import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

# In order to run this code just exectue the CatGPT_run.py file

@st.cache_resource
def get_model():
    """
    Function to load the model and tokenizer directly from the Hugging Face model hub.
    """
    tokenizer = AutoTokenizer.from_pretrained("baiges/CatGPT")
    model = AutoModelForCausalLM.from_pretrained("baiges/CatGPT")
    return model, tokenizer


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
                   "Durant la Guerra Civil Espanyola, Catalunya",
                   "Els Bitcoin i altres criptomonedes s'han convertit en temes importants a Catalunya, especialment després de",
                   "La meva casa és un lloc molt acollidor",
                   "El clima mediterrani permet gaudir de llargues jornades assolellades a la vora del mar.",
                   "El sol es ponia en l'horitzó de milers de colors però els pirates no podien",
                   "L'oli d'oliva verge extra"]


# Check if an input_text is already selected in session_state
if 'input_text' not in st.session_state:
    st.session_state.input_text = random.choice(random_inputs)

input_text = st.text_area('Input Text', st.session_state.input_text)

num_return_sequences = st.number_input('Number of Sequences', min_value=1, max_value=5, value=1)

# Arrange sliders horizontally
col1, col2, col3, col4 = st.columns(4)

with col1:
    max_length = st.slider('Max Length', min_value=35, max_value=500, value=75)

with col2:
    temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=1.0, step=0.1)

with col3:
    top_k = st.slider('Top-k', min_value=0, max_value=5, value=1)

with col4:
    repetition_penalty = st.slider('Repetition Penalty', min_value=1.0, max_value=2.0, value=1.2, step=0.1)


# Load the model

model, HF_tokenizer = get_model()

# Tips for generating text

tips = [
    "Tip: Use temperature to control the randomness of predictions. Lower values make the model more deterministic.",
    "Tip: Adjust top-k to limit the sampling pool to the top-k probable next tokens.",
    "Tip: Use repetition penalty to avoid generating repetitive sequences.",
    "Tip: In order to generate more creative text, try different input prompts.",
    "Tip: As the max_length increases, there will be a higher chance of generating repetitive text or hallucinations. Adjust repetition penalty accordingly.",
    "Tip: The model may generate text that is not coherent or relevant to the input prompt. Try different prompts to get better results.",
    "Tip: Adjust top-k to 0 to allow sampling from the entire vocabulary distribution.",
    "Tip: Use a higher temperature to generate more diverse and creative text but remember to set a top_k different to 1.",
    "Tip: Providing a specific and well-defined prompt can improve the relevance of the generated text.",
]


if st.button('Generate'):
    with st.spinner('Generating...'):
        # Select a random tip to display
        selected_tip = random.choice(tips)
        tip_placeholder = st.info(selected_tip)  # Show the tip
        if HF_tokenizer:
            inputs = HF_tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=num_return_sequences, repetition_penalty=repetition_penalty, temperature=temperature, top_k=top_k, do_sample=True)
            generated_texts = []
            for output in outputs:
                generate_text = HF_tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(generate_text)
        tip_placeholder.empty()  # Delete the tip after generating the text
    for i, text in enumerate(generated_texts):
        st.write(f'Sample {i+1}:')
        st.write(text)