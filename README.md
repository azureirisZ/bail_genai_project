\# ⚖️ AI Bail Reckoner

> \*\*An AI-powered Legal Assistant that predicts bail outcomes and cites real case laws.\*\*



\## 🚀 Features

\- \*\*Smart Verdict Generation:\*\* Uses a custom LSTM + Attention model trained on legal cases.

\- \*\*RAG Engine:\*\* Retrieves real legal precedents (Section 439, 302 IPC) using a custom Word2Vec Search System.

\- \*\*Secure Dashboard:\*\* Encrypted chat history and case management.

\- \*\*Privacy First:\*\* 100% Offline-capable architecture (No OpenAI/Gemini APIs).



\## 📂 Project Structure

\- `src/` - The core AI logic (RAG engine, Model definitions).

\- `scripts/` - Utilities for data processing and uploading to Hugging Face.

\- `app.py` - The main Web Application (Flask).

\- `database/` - (Local only) The vector search index.

\- `weights/` - (Local only) The trained model weights.



\## 🔗 Resources

\- \*\*Model Weights:\*\* \[Hugging Face Link](https://huggingface.co/anabaena/bail-reckoner-models)

\- \*\*Legal Database:\*\* \[Hugging Face Link](https://huggingface.co/anabaena/bail-reckoner-data)

