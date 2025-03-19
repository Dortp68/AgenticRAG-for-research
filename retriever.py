from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

import asyncio
import os

from utils import config
from utils.websearch import web_search_text


class DocumentProcessor:
    """
    Handles document loading and splitting.
    """
    @staticmethod
    def load_pdf(path) -> list[Document]:
        try:
            docs = PyPDFLoader(path).load()
        except Exception as e:
            raise RuntimeError(f"Error processing document: {e}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        doc_splits = text_splitter.split_documents(docs)
        return doc_splits

    @staticmethod
    def load_documents() -> list[Document]:
        """
        Loading PDF documents from directory and preprocess them for vectorstore.

        Returns:
            list[Document]: list of preprocessed documents.
        """
        print("---LOADING DOCUMENTS---")
        try:
            docs = [PyPDFLoader(os.path.join(config.documents_path, doc)).load() for doc in os.listdir(os.path.join(os.getcwd(), config.documents_path))]
        except Exception as e:
            raise RuntimeError(f"Error processing document: {e}")

        docs_list = [item for sublist in docs for item in sublist]
        print("---SPLITTING DOCUMENTS---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits

    @staticmethod
    def load_web(urls: list[str]) -> list[str]:
        """
        Loading end preprocess web pages content from given urls.

        Returns:
            list[str]: list pages content.
        """
        async def load_(url: str) -> list[str]:
            loader = UnstructuredLoader(web_url=url)
            setup_docs = []
            async for doc in loader.alazy_load():
                if doc.metadata["category"] == "NarrativeText" or doc.metadata["category"] == "ListItem":
                    setup_docs.append(doc.page_content)
            return setup_docs

        print("---LOADING WEB PAGES---")
        web_content = []
        for url in urls:
            page_setup_docs = asyncio.run(load_(url))
            web_content.extend(page_setup_docs)
            if len(web_content) == 3: break
        return web_content


@tool
def web_search_tool(query: str) -> str:
    """Performs a web search and returns the top 5 result snippet."""
    urls = web_search_text(query)
    docs = DocumentProcessor.load_web(urls)
    result = "\n".join(docs)
    return result



class IndexBuilder:

    def __init__(self):
        self.vectorstore = None
        self.embeddings = OllamaEmbeddings(model=config.embedding_model)
        self.model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

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
        """
        Pulling list of document in vectorstore.
        """
        try:
            print("---PULL DOCUMENTS IN VECTORSTORE---")
            self.vectorstore.add_documents(docs)
        except Exception as e:
            raise RuntimeError(f"Error pulling documents: {e}")

    def build_retriever(self):
        """
        Builds BM25 and vector-based retrievers and combines them into an ensemble retriever, default or with reranking.

        Returns:
            ContextualCompressionRetriever: retriever with reranking.
        """
        try:
            print("---BUILDING BM25 RETRIEVER---")
            ids = self.vectorstore.get()["ids"]
            bm25_retriever = BM25Retriever.from_documents(self.vectorstore.get_by_ids(ids), search_kwargs={"k": 10})

        except Exception as e:
            raise RuntimeError(f"Error building BM25 retriever: {e}")

        try:
            print("---BUILDING MMR RETRIEVER---")
            mmr_retriever = self.vectorstore.as_retriever(search_type="mmr", k=10)
            print("---BUILDING SIMILARITY RETRIEVER---")
            sim_retriever = self.vectorstore.as_retriever(search_type="similarity", k=10)
            print("---COMBINING RETRIEVERS---")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[sim_retriever, mmr_retriever, bm25_retriever],
                weights=[0.3, 0.3, 0.4],
            )
            if config.reranking:
                print("---BUILDING RETRIEVER WITH RERANKING---")
                compressor = CrossEncoderReranker(model=self.model, top_n=config.top_k)
                ensemble_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=ensemble_retriever
                )
        except Exception as e:
            raise RuntimeError(f"Error pulling documents: {e}")

        return ensemble_retriever

















