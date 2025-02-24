from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool

import asyncio
import os

from utils import config
from utils.prompts import RETRIEVER_TOOL_PROMPT
from utils.utils import web_search_text


class DocumentProcessor:
    """
    Handles document loading and splitting.
    """

    @staticmethod
    def load_pdf():
        pass

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
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
            print(page_setup_docs)
            web_content.extend(page_setup_docs)
        return web_content


@tool
def web_search_tool(query: str) -> str:
    """Performs a web search and returns the top 5 result snippet."""
    urls = web_search_text(query)
    docs = DocumentProcessor.load_web(urls)
    result = "\n".join(docs)
    return result

@tool
def general_query_tool(query: str) -> str:
    """Response on general type user`s query"""
    return query


class IndexBuilder:

    def __init__(self):
        self.vectorstore = None
        self.embeddings = OllamaEmbeddings(model=config.embedding_model)

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
        Builds vector-based retrievers default or with reranking.

        Returns:
            ContextualCompressionRetriever: retriever with reranking.
        """
        try:
            if config.reranking:
                print("---BUILDING RETRIEVER WITH RERANKING---")
                base_retriever = self.vectorstore.as_retriever(k=10)
                model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                compressor = CrossEncoderReranker(model=model, top_n=config.top_k)
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=base_retriever
                )
            else:
                print("---BUILDING DEFAULT RETRIEVER---")
                retriever = self.vectorstore.as_retriever(k=config.top_k)
        except Exception as e:
            raise RuntimeError(f"Error pulling documents: {e}")

        return retriever

def get_retriever_tool():

    builder = IndexBuilder()
    builder.build_vectorstore()

    if len(builder.vectorstore.get()["ids"]) == 0:
        docs_list = DocumentProcessor.load_documents()
        builder.pull_documents(docs_list)

    retriever = builder.build_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_research_papers",
        RETRIEVER_TOOL_PROMPT,
        response_format = "content_and_artifact"
    )
    return  retriever_tool

















