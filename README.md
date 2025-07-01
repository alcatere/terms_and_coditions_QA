# 📄 RAG-Based Q&A Chatbot for Terms & Conditions

This project implements a **Retrieval-Augmented Generation (RAG)** system using an open-source LLM to answer natural language questions based on uploaded **Terms & Conditions** or **contract documents** (PDF or Word). It's designed for ease of use through a **Streamlit interface**, ideal for legal teams, customer service, or compliance use cases.

---

## 🔍 Features

- ✅ Upload and parse `.pdf` or `.docx` documents
- 🧠 Chunk documents and store them in a vector database (FAISS)
- 📎 Embed text with `sentence-transformers`
- 🤖 Use **`google/gemma-2b-it`**, a lightweight instruct-tuned LLM
- 💬 Ask questions in natural language about the uploaded document
- 🎯 Answers are grounded in the retrieved document context
- 🧪 Runs locally or deploys easily to Streamlit Cloud

---

## 🚀 Live Demo

> Will appear here once deployed:  
`https://<your-streamlit-app-url>.streamlit.app`

---

## 📦 Installation (Local)

```bash
git clone https://github.com/alcatere/terms_and_coditions_QA.git
cd terms_and_coditions_QA
pip install -r requirements.txt
streamlit run app.py