import streamlit as st
import pdfplumber
import docx
import torch
import faiss
import numpy as np
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ---- CONFIG ----
st.set_page_config(page_title="RAG Q&A Assistant", layout="wide")
MODEL_ID = "google/gemma-2b-it"
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ENCODING = tiktoken.get_encoding("cl100k_base")

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, model, embedder

tokenizer, model, embedder = load_models()

# ---- FILE PARSING ----
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return None

# ---- CHUNKING ----
def split_text(text, max_tokens=300, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk, tokens = [], 0
        for word in words[i:]:
            token_len = len(ENCODING.encode(word))
            if tokens + token_len > max_tokens:
                break
            chunk.append(word)
            tokens += token_len
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks

# ---- EMBEDDING & FAISS ----
def index_chunks(chunks):
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, {i: chunk for i, chunk in enumerate(chunks)}

def retrieve_chunks(query, index, mapping, k=3):
    q_vec = embedder.encode([query])
    _, idxs = index.search(q_vec, k)
    return [mapping[i] for i in idxs[0]]

# ---- PROMPTING & GENERATION ----
def build_prompt(context, question):
    return f"""
<start_of_turn>user
Answer the questions using only the context below
Context:
{context}

Question:
{question}
<end_of_turn>
<start_of_turn>model
""".strip()

def generate_answer(context, question, max_tokens=300):
    prompt = build_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        ).to(DEVICE)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("<start_of_turn>model")[-1].strip()

# ---- STREAMLIT UI ----
st.markdown("# 🤖 RAG Q&A Chatbot")
st.markdown("Ask questions about any **PDF or Word (.docx)** file using a local open-source LLM (Gemma-2b-it).")

uploaded_file = st.file_uploader("📤 Upload your document", type=["pdf", "docx"])

if uploaded_file:
    text = extract_text(uploaded_file)
    if not text or text.strip() == "":
        st.error("❌ Could not extract text from this file.")
    else:
        chunks = split_text(text)
        index, mapping = index_chunks(chunks)
        st.success(f"✅ Document loaded and split into {len(chunks)} chunks.")

        st.markdown("### 💬 Ask a question below:")

        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input("🔎 Your question")
        with col2:
            ask = st.button("Ask")

        if ask and question:
            retrieved = retrieve_chunks(question, index, mapping)
            context = "\n".join(retrieved)
            with st.spinner("Thinking... 🤔"):
                answer = generate_answer(context, question)

            st.markdown("### 🧠 Answer:")
            st.markdown(f"""
<div style="background-color:#f0f2f6;padding:15px;border-radius:10px;">
{answer.split('Question:')[-1][1:]}
</div>
""", unsafe_allow_html=True)

            with st.expander("📄 Retrieved Chunks"):
                for i, chunk in enumerate(retrieved, 1):
                    st.markdown(f"**Chunk {i}:**\n```\n{chunk}\n```")