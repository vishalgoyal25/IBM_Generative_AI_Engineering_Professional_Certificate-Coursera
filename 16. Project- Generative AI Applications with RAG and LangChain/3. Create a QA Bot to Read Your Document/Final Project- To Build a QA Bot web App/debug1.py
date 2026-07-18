from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Hello World",
    "Artificial Intelligence",
    "Large Language Models"
]

db = Chroma.from_texts(
    texts=texts,
    embedding=embedding
)

print("SUCCESS")