import os
import re

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
# SETUP
# ============================================================

load_dotenv()


# ============================================================
# GET YOUTUBE LINK FROM USER
# ============================================================

youtube_url = input("Enter YouTube video link: ").strip()


# ============================================================
# EXTRACT VIDEO ID
# ============================================================

match = re.search(
    r"(?:v=|youtu\.be/|youtube\.com/shorts/)([^&?/]+)",
    youtube_url
)

if not match:
    print("Invalid YouTube URL.")
    exit()

video_id = match.group(1)


# ============================================================
# STEP 1a - INDEXING (DOCUMENT INGESTION)
# ============================================================

try:

    ytt_api = YouTubeTranscriptApi()

    transcript_data = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        snippet.text for snippet in transcript_data
    )

    print("\nTranscript loaded successfully.")

except Exception as e:

    print("\nCould not get the transcript.")
    print("Make sure the video has an English transcript.")
    print("Error:", e)

    exit()


# ============================================================
# STEP 1b - INDEXING (TEXT SPLITTING)
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print("Number of chunks:", len(chunks))


# ============================================================
# STEP 1c - INDEXING (EMBEDDING GENERATION)
# ============================================================

try:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

except Exception as e:

    print("\nCould not load the embedding model.")
    print("Error:", e)

    exit()


# ============================================================
# STEP 1d - INDEXING (STORING IN VECTOR STORE)
# ============================================================

try:

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("FAISS vector store created.")

except Exception as e:

    print("\nCould not create FAISS vector store.")
    print("Error:", e)

    exit()


# ============================================================
# STEP 2 - RETRIEVAL
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ============================================================
# STEP 3 - AUGMENTATION
# ============================================================

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
    input_variables=["context", "question"]
)


# ============================================================
# STEP 4 - GENERATION
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:

    print("\nGROQ_API_KEY was not found.")
    print("Please check your .env file.")

    exit()


try:

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=groq_api_key
    )

except Exception as e:

    print("\nCould not connect to Groq.")
    print("Error:", e)

    exit()


# ============================================================
# BUILDING A CHAIN
# ============================================================

def format_docs(retrieved_docs):

    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    return context_text


parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser


# ============================================================
# ASK QUESTION
# ============================================================

question = input("\nAsk your question about the video: ").strip()


if not question:

    print("Please enter a question.")

    exit()


try:

    answer = main_chain.invoke(question)

    print("\nAnswer:\n")
    print(answer)

except Exception as e:

    print("\nSomething went wrong while generating the answer.")
    print("Error:", e)