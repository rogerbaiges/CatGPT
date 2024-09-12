import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("baiges/CatGPT-IT")
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
    input_text = f'<|im_start|>user \n{prompt}<|im_end|>\n<|im_start|>assistant'
    
    # Tokenize the input prompt
    tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Calculate the length of the prompt (we'll use this to slice the output)
    prompt_length = tokens.shape[1]  
    
    # Generate the response, setting the attention mask
    attention_mask = torch.ones(tokens.shape, dtype=torch.long, device=device)
    output_tokens = model.generate(tokens, attention_mask=attention_mask, generation_config=genconf)
    
    # Decode the response from tokens, keeping only new tokens after the prompt
    result = tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)
    
    return result.strip()

# Streamlit UI design
st.set_page_config(page_title="Beta Chatbot Interface", layout="centered")

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
            background-color: #565a67;
            color: white;
            border-radius: 10px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #92a4dc;
        }
        .user-message {
            background-color: #1E90FF;
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: right;
            font-weight: bold;
            margin-bottom: 10px;
            margin-left: 25%;
        }
        .bot-message {
            background-color: #3CB371;
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: left;
            font-weight: bold;
            margin-bottom: 10px;
            margin-right: 25%;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 10px;
            padding-left: 10px;
            border: 1px solid #b1b1b1;
            border-radius: 10px;
            background-color: #444654;
            margin-bottom: 10px;
        }
        .input-container {
            position: fixed;
            bottom: 10px;
            width: 60%;
            left: 20%;
            display: flex;
            align-items: center;
            padding: 10px;
        }
        .input-container input {
            flex-grow: 1;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #b1b1b1;
            margin-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Beta Chatbot Interface")

# Initialize or retrieve session state
if "context" not in st.session_state:
    st.session_state.context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input text box
def send_message():
    user_input = st.session_state.input_text
    if user_input:
        response = generate_response(user_input)
        st.session_state.messages.append({"role": "user", "text": user_input})
        st.session_state.messages.append({"role": "bot", "text": response})
        st.session_state.input_text = ""  # Clear the input box

def clear_chat():
    st.session_state.context = ""

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Clear chat button
if st.button("Clear Chat"):
    clear_chat()

# Input container at the bottom
st.text_input("You:", value="", key="input_text", on_change=send_message, placeholder="Type your message here...")
