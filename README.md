# MedPrompt 🩺 — Clinical Text Structuring with AI

**MedPrompt** is a modular NLP pipeline that converts unstructured clinical notes into structured summaries using biomedical NER and local LLMs.

## Features
- Extracts key entities (diseases, drugs, symptoms) using Clinical-AI-Apollo NER model
- Filters irrelevant info and adds context with local corpus + Wikipedia
- Uses Ollama LLM (Mistral) to generate:
  -  JSON summaries (Concise & Detailed)
  -  Short human-readable summary
- Interactive Streamlit UI with visualizations

## 🛠 Tech Stack
**Python**, HuggingFace Transformers, BioBERT, Ollama, Streamlit, pandas, Wikipedia API

##  Run Locally
```bash
git clone https://github.com/PagadalaNikitha572/MedPrompt.git
cd MedPrompt
pip install -r requirements.txt
ollama pull mistral
streamlit run app.py
