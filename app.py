import streamlit as st
import os
import pdfplumber
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel

st.set_page_config(page_title="Model Comparison Chat", layout="wide")

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #111;
        color: #fff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .css-1v0mbdj, .css-1d391kg, .st-bw, .st-cg {
        background: #111 !important;
        color: #fff !important;
    }
    .stTextInput > div > div > input {
        background: #222 !important;
        color: #fff !important;
        border: 1px solid #444 !important;
    }
    .stButton > button {
        background: #222 !important;
        color: #fff !important;
        border: 1px solid #444 !important;
    }
    .stRadio > div, .stSelectbox > div {
        background: #111 !important;
        color: #fff !important;
    }
    .chat-message {
        background: #222;
        color: #fff;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PDF_PATH = "trainingdata/TrainingPDF_SolarSystem.pdf"
CHUNK_SIZE = 500

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_choice):
    if model_choice == "Trained Model":
        base_model_path = "jykh01/tinyllama-base"
        lora_path = "jykh01/tinyllama-lora"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        base_model_path = "jykh01/tinyllama-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
    return model, tokenizer

@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=True)
def read_and_chunk_pdf(pdf_path, chunk_size=CHUNK_SIZE):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks, text

@st.cache_data(show_spinner=True)
def get_pdf_token_count(_tokenizer, text):
    tokens = _tokenizer.encode(text)
    return len(tokens)

with st.sidebar:
    st.markdown("## Model")
    model_choice = st.radio("Select Model", ["Base Model", "Trained Model"])

st.markdown("# Solar System Knowledge Base Chatbot")

with st.spinner("Loading model and PDF..."):
    model, tokenizer = load_model_and_tokenizer(model_choice)
    embedder = load_embedder()
    pdf_chunks, pdf_text = read_and_chunk_pdf(PDF_PATH)
    pdf_token_count = get_pdf_token_count(tokenizer, pdf_text)
    chunk_embeddings = embedder.encode(pdf_chunks, convert_to_tensor=True)

def get_relevant_chunks(query, embedder, pdf_chunks, chunk_embeddings, top_k=2):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, chunk_embeddings, top_k=top_k)[0]
    return [pdf_chunks[hit['corpus_id']] for hit in hits]

def answer_with_pdf_context(query, model, tokenizer, embedder, pdf_chunks, chunk_embeddings):
    relevant_chunks = get_relevant_chunks(query, embedder, pdf_chunks, chunk_embeddings)
    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean answer
    if "Answer:" in answer:
        answer = answer.split("Answer:", 1)[-1].strip()

    # Remove anything after next "Question:" or similar markers
    for stop_token in ["\nQuestion:", "\nQ:", "\nUser:", "\nYou:", "\nPrompt:"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0].strip()

    return answer

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question...", key="input")
    submit = st.form_submit_button("Send")

if submit and user_input.strip():
    st.session_state.chat_history.append(f"You: {user_input}")
    with st.spinner(f"Generating answer from {model_choice.lower()}..."):
        answer = answer_with_pdf_context(user_input, model, tokenizer, embedder, pdf_chunks, chunk_embeddings)
    st.session_state.chat_history.append(f"{model_choice}: {answer}")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        st.markdown(f'<div class="chat-message">{msg}</div>', unsafe_allow_html=True)
