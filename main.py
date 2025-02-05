from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_rag")

docs = PyPDFLoader("documents/Titan.pdf").load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)
print(len(doc_splits))
vectorstore.add_documents(doc_splits)
print(vectorstore.get())