import os
import re

import streamlit as st

from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #777777;
    margin-bottom: 35px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🎥 YouTube RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about any YouTube video</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# YOUTUBE VIDEO INPUT
# ============================================================

st.subheader("🔗 YouTube Video")

youtube_url = st.text_input(
    "Enter YouTube video link",
    placeholder="https://www.youtube.com/watch?v=..."
)


# ============================================================
# PROCESS VIDEO
# ============================================================

if st.button("Process Video"):

    if not youtube_url:

        st.warning("Please enter a YouTube video link.")

    else:

        match = re.search(
            r"(?:v=|youtu\.be/|youtube\.com/shorts/)([^&?/]+)",
            youtube_url
        )

        if not match:

            st.error("Please enter a valid YouTube URL.")

        else:

            video_id = match.group(1)

            try:

                with st.spinner("Processing video..."):

                    # ====================================================
                    # STEP 1a - DOCUMENT INGESTION
                    # ====================================================

                    ytt_api = YouTubeTranscriptApi()

                    transcript_data = ytt_api.fetch(
                        video_id,
                        languages=["en"]
                    )

                    transcript = " ".join(
                        snippet.text
                        for snippet in transcript_data
                    )


                    # ====================================================
                    # STEP 1b - TEXT SPLITTING
                    # ====================================================

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.create_documents(
                        [transcript]
                    )


                    # ====================================================
                    # STEP 1c - EMBEDDINGS
                    # ====================================================

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )


                    # ====================================================
                    # STEP 1d - FAISS VECTOR STORE
                    # ====================================================

                    vector_store = FAISS.from_documents(
                        chunks,
                        embeddings
                    )


                    # ====================================================
                    # STEP 2 - RETRIEVAL
                    # ====================================================

                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )


                    # ====================================================
                    # STEP 3 - PROMPT
                    # ====================================================

                    prompt = PromptTemplate(
                        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
                        input_variables=[
                            "context",
                            "question"
                        ]
                    )


                    # ====================================================
                    # STEP 4 - GROQ
                    # ====================================================

                    if not groq_api_key:

                        st.error(
                            "GROQ_API_KEY was not found in your .env file."
                        )

                        st.stop()


                    llm = ChatGroq(
                        model="openai/gpt-oss-20b",
                        temperature=0,
                        api_key=groq_api_key
                    )


                    # ====================================================
                    # BUILDING CHAIN
                    # ====================================================

                    def format_docs(retrieved_docs):

                        context_text = "\n\n".join(
                            doc.page_content
                            for doc in retrieved_docs
                        )

                        return context_text


                    parallel_chain = RunnableParallel({
                        "context":
                            retriever |
                            RunnableLambda(format_docs),

                        "question":
                            RunnablePassthrough()
                    })


                    parser = StrOutputParser()


                    main_chain = (
                        parallel_chain |
                        prompt |
                        llm |
                        parser
                    )


                    # ====================================================
                    # SAVE CHAIN
                    # ====================================================

                    st.session_state.main_chain = main_chain

                    st.session_state.video_processed = True

                    st.session_state.video_id = video_id

                    st.session_state.chunk_count = len(chunks)


                st.success(
                    f"Video processed successfully! "
                    f"{len(chunks)} chunks created."
                )


            except Exception as e:

                st.error("Could not process this video.")

                st.caption(str(e))


# ============================================================
# QUESTION SECTION
# ============================================================

if st.session_state.get("video_processed", False):

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="What is this video about?"
    )


    # ========================================================
    # ASK QUESTION
    # ========================================================

    if st.button("Ask Question"):

        if not question:

            st.warning("Please enter a question.")

        else:

            try:

                with st.spinner("Thinking..."):

                    answer = st.session_state.main_chain.invoke(
                        question
                    )


                # ====================================================
                # ANSWER
                # ====================================================

                st.subheader("📝 Answer")

                st.info(answer)


            except Exception as e:

                st.error(
                    "Something went wrong while generating the answer."
                )

                st.caption(str(e))