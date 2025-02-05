from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from utils import config

class DocumentProcessor:
    """
    Handles document loading and splitting.
    """

    @staticmethod
    def load_pdf():
        pass

    @staticmethod
    def load_documents():
        """ """
        print("---LOADING DOCUMENTS---")
        docs = [PyPDFLoader(os.path.join(config.documents_path, doc)).load() for doc in os.listdir(config.documents_path)]
        docs_list = [item for sublist in docs for item in sublist]
        print("---SPLITTING DOCUMENTS---")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits

    @staticmethod
    def load_web():
        pass
class IndexBuilder:

    def __init__(self):