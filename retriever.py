from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool

import asyncio
import os

from utils import config
from utils.prompts import RETRIEVER_TOOL_PROMPT

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
            docs = [PyPDFLoader(os.path.join(config.documents_path, doc)).load() for doc in os.listdir(os.path.join(os.getcwd(), config.documents_path))]
        except Exception as e:
            raise RuntimeError(f"Error processing document: {e}")

        docs_list = [item for sublist in docs for item in sublist]
        print("---SPLITTING DOCUMENTS---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits

    @staticmethod
    def load_web(urls: list[str]) -> list[str]:

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

        return web_content




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

from utils.utils import web_search_text

@tool
def web_search_tool(query: str) -> str:
    """Performs a web search and returns the top 5 result snippet."""
    urls = web_search_text(query)
    docs = DocumentProcessor.load_web(urls)
    result = "\n".join(docs)
    return result

















