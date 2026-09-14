## 🚀 Live Demo

👉 [Open YouTube RAG Assistant]((https://youtube-rag-assistant-anu.streamlit.app/))

# 🎥 YouTube RAG Assistant

### 🎯 Stop Watching. Start Asking.

> **What if you could talk to a YouTube video instead of watching the entire thing?**

YouTube RAG Assistant turns long YouTube videos into an interactive question-answering experience.

🔗 **Live Demo:** YOUR_STREAMLIT_LINK_HERE

---

## 🚀 The Problem

We've all been there.

You open a **1-hour YouTube video** to learn something...

You watch for 20 minutes.

You forget where the information was.

You rewind.

You search again.

You watch another 10 minutes.

😵‍💫 **There has to be a better way.**

That's where **YouTube RAG Assistant** comes in.

Instead of manually searching through a long video, simply provide the YouTube link and ask a question.

The system finds the most relevant parts of the video's transcript and generates an answer using those pieces of information.

---

# 🧠 How It Works

The application follows a **Retrieval-Augmented Generation (RAG)** pipeline.

```text
                🎥 YouTube Video
                       │
                       ▼
              📝 Get Transcript
                       │
                       ▼
              ✂️ Split into Chunks
                       │
                       ▼
             🧠 Generate Embeddings
                       │
                       ▼
              📦 Store in FAISS
                       │
                       ▼
                 ❓ User Question
                       │
                       ▼
              🔍 Retrieve Relevant
                    Chunks
                       │
                       ▼
                🧩 Build Context
                       │
                       ▼
                  🤖 Groq LLM
                       │
                       ▼
                 💬 Final Answer
