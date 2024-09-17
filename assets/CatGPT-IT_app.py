import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import random
import logging

# Load the model and tokenizer

model = AutoModelForCausalLM.from_pretrained("baiges/CatGPT-IT")
logging.getLogger("transformers").setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained("baiges/CatGPT-IT")

# Ensure model is on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configuration for response generation
genconf = GenerationConfig(
    max_length=500,           # Maximum length of the generated response
    repetition_penalty=1.2,   # To avoid repetition of phrases in responses
    temperature=0.6,          # Controls randomness in response (lower is more deterministic)
    top_k=2,                  # Use top-k sampling
    do_sample=True,           # Enable sampling
)

# Function to generate a response
def generate_response(prompt):
    # Create input with special tokens for user and assistant
    input_text = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant'
    
    # Tokenize the input prompt
    tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Calculate the length of the prompt (we'll use this to slice the output)
    prompt_length = tokens.shape[1]

    # Generate the response, setting the attention mask
    attention_mask = torch.ones(tokens.shape, dtype=torch.long, device=device)
    output_tokens = model.generate(
        tokens,
        attention_mask=attention_mask,
        generation_config=genconf
    )

    # Decode the response from tokens, keeping only new tokens after the prompt
    result = tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)

    return result.strip()

# Streamlit UI design
st.set_page_config(page_title="CatGPT Chatbot", layout="centered")

# Use columns to place the logo next to the title
col1, col2 = st.columns([3, 1])

with col1:
    st.title('CatGPT Chatbot')
    st.write('Chat with CatGPT, your Catalan AI assistant')

with col2:
    logo_path = "logo/CatGPT_round.png"
    st.image(logo_path, width=150)

st.markdown("""
    <style>
        body {
            background-color: #2c2f33;
        }
        .stTextInput input {
            color: white;
            background-color: #565a67;
            border: 1px solid #b1b1b1;
        }
        .stButton>button {
            background-color: #7289da;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            margin-top: 10px;
        }
        .stButton>button:hover {
            background-color: #5b6eae;
        }
        .message-container {
            display: flex;
            margin-bottom: 10px;
        }
        .message-container.user {
            justify-content: flex-end;
        }
        .message-container.bot {
            justify-content: flex-start;
        }
        .user-message, .bot-message {
            background-color: #7289da;
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: left;  /* Align text to the left within the bubble */
            font-weight: normal;
            max-width: 70%;    /* Limit the maximum width of the message bubble */
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .bot-message {
            background-color: #99aab5;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 10px;
            padding-left: 10px;
            border: 1px solid #b1b1b1;
            border-radius: 10px;
            background-color: #2c2f33;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize or retrieve session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add pre-defined initial messages
    initial_messages = [
        "Hola! Sóc CatGPT, el teu assistent virtual en català. Com puc ajudar-te avui?",
        "Benvingut! Estic aquí per ajudar-te amb qualsevol consulta o tasca que necessitis.",
        "Hola! Sóc una intel·ligència artificial en català. Què puc fer per tu?",
        "Salutacions! Aquí CatGPT, preparat per ajudar-te en el que necessitis.",
        "Hola! Com puc ser-te útil avui?"
    ]
    selected_message = random.choice(initial_messages)
    st.session_state.messages.append({"role": "bot", "text": selected_message})

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'''
                <div class="message-container user">
                    <div class="user-message">{message["text"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'''
                <div class="message-container bot">
                    <div class="bot-message">{message["text"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# Input text box
def send_message():
    user_input = st.session_state.input_text
    if user_input:
        st.session_state.messages.append({"role": "user", "text": user_input})
        response = generate_response(user_input)
        st.session_state.messages.append({"role": "bot", "text": response})
        st.session_state.input_text = ""  # Clear the input box

def clear_chat():
    st.session_state.messages = []
    # Add a new initial message
    initial_messages = [
        "Hola! Sóc CatGPT, el teu assistent virtual en català. Com puc ajudar-te avui?",
        "Benvingut! Estic aquí per ajudar-te amb qualsevol consulta o tasca que necessitis.",
        "Hola! Sóc una intel·ligència artificial en català. Què puc fer per tu?",
        "Salutacions! Aquí CatGPT, preparat per ajudar-te en el que necessitis.",
        "Hola! Com puc ser-te útil avui?"
    ]
    selected_message = random.choice(initial_messages)
    st.session_state.messages.append({"role": "bot", "text": selected_message})

# Clear chat button
if st.button("Clear Chat"):
    clear_chat()

# Input container at the bottom
st.text_input(
    "Type your message here...",
    value="",
    key="input_text",
    on_change=send_message,
    placeholder="Type your message here..."
)