
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma




embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_rag")

vectorstore.reset_collection()