import warnings
import gradio as gr

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# --------------------------------------------------
# Suppress Warnings
# --------------------------------------------------

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Replace with your own Groq API Key
GROQ_API_KEY = "gsk_****************************"

# Recommended model
MODEL_NAME = "llama-3.3-70b-versatile"

# Local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------
# Initialize LLM
# --------------------------------------------------

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0.5,
)

# --------------------------------------------------
# Load PDF Document
# --------------------------------------------------


def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# --------------------------------------------------
# Split Documents into Chunks
# --------------------------------------------------


def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    return chunks


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------


def embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return embeddings


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------


def vector_database(chunks):

    print("Creating Embeddings...")

    embeddings = embedding_model()

    print("Creating FAISS Vector Database...")

    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)

    print("Vector Database Created")

    return vectordb


# --------------------------------------------------
# Create Retriever
# --------------------------------------------------


def retriever(file_path):

    print("Loading PDF...")

    documents = document_loader(file_path)

    print("Splitting PDF...")

    chunks = text_splitter(documents)

    print(f"Chunks Created: {len(chunks)}")

    vectordb = vector_database(chunks)

    print("Retriever Created")

    return vectordb.as_retriever(search_kwargs={"k": 3})


# --------------------------------------------------
# Create RetrievalQA Chain
# --------------------------------------------------


def qa_bot(file_path):

    retriever_obj = retriever(file_path)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True,
    )

    return qa


# --------------------------------------------------
# Question Answering Pipeline
# --------------------------------------------------


def answer_question(file_path, query):

    print("Creating RetrievalQA Chain...")

    qa = qa_bot(file_path)

    print("Calling LLM...")

    result = qa.invoke({"query": query})

    print("Answer Generated")

    return result["result"]


# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

rag_application = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath",
        ),
        gr.Textbox(
            label="Input Query", lines=2, placeholder="Type your question here..."
        ),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering Bot",
    description="Upload a PDF document and ask any question. The chatbot will answer using the document.",
)

# --------------------------------------------------
# Launch Application
# --------------------------------------------------

if __name__ == "__main__":
    rag_application.launch()
