import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("models/final_model")
tokenizer = AutoTokenizer.from_pretrained("baiges/CatGPT")

# Ensure model is on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate a response
def generate_response(input_text, context=None):
    input_text = f"<s> {input_text} </s> <s>"
    if context:
        input_text = context + " " + input_text

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=min(1024, len(input_ids) + 250),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            do_sample=True,
            top_k=2,
            repetition_penalty=1.4
        )

    generated_ids = output_ids[0][input_ids.size(-1):]
    decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if "</s>" in decoded_output:
        decoded_output = decoded_output.split("</s>")[0].strip()

    return decoded_output

def update_context(context, user_input, response, max_length=750):
    new_interaction = f"<s> {user_input} </s> <s> {response} </s> <s>"
    context += new_interaction
    context_ids = tokenizer.encode(context)

    if len(context_ids) > max_length:
        context_ids = context_ids[-max_length:]

    context = tokenizer.decode(context_ids, skip_special_tokens=True)
    return context

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
        response = generate_response(user_input, st.session_state.context)
        st.session_state.context = update_context(st.session_state.context, user_input, response)
        st.session_state.messages.append({"role": "user", "text": user_input})
        st.session_state.messages.append({"role": "bot", "text": response})
        st.session_state.input_text = ""  # Clear the input box

def clear_chat():
    st.session_state.context = ""
    st.session_state.messages = []

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
