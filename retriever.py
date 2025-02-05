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
        try:
            docs = [PyPDFLoader(os.path.join(config.documents_path, doc)).load() for doc in os.listdir(config.documents_path)]
        except Exception as e:
            raise RuntimeError(f"Error processing document: {e}")

        docs_list = [item for sublist in docs for item in sublist]
        print("---SPLITTING DOCUMENTS---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits

    @staticmethod
    def load_web():
        pass


class IndexBuilder:

    def __init__(self):
        self.vectorstore = None
        self.embeddings = OllamaEmbeddings(model = config.embedding_model)

    def build_vectorstore(self):
        """
        Initializes the Chroma vectorstore with the provided embeddings.
        """
        try:
            print("---BUILDING VECTORSTORE---")
            self.vectorstore = Chroma(collection_name=config.collection_name,
                                      embedding_function=self.embeddings,
                                      persist_directory=config.persist_directory)
        except Exception as e:
            raise RuntimeError(f"Error building vectorstore: {e}")

    def pull_documents(self, docs: list):
        try:
            print("---PULL DOCUMENTS IN VECTORSTORE---")
            self.vectorstore.add_documents(docs)
        except Exception as e:
            raise RuntimeError(f"Error pulling documents: {e}")

    def build_retriever(self):

        try:
            print("---BUILDING RETRIEVER---")
            retriever = self.vectorstore.as_retriever(k=config.top_k)
        except Exception as e:
            raise RuntimeError(f"Error pulling documents: {e}")

        return retriever

if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build_vectorstore()

    if len(builder.vectorstore.get()["ids"]) == 0:
        docs_list = DocumentProcessor.load_documents()
        builder.pull_documents(docs_list)

    retriever = builder.build_retriever()
    print(builder.vectorstore.get())













