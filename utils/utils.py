import gradio as gr
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool

from utils.prompts import RETRIEVER_TOOL_PROMPT
from retriever import web_search_tool, DocumentProcessor, IndexBuilder
from utils import config, llm, memory
from agents.main_graph import Supervisor


class GradioHandler:
    def __init__(self):

        self.builder, self.retriever_tool = self.build_retriever()
        self.tools = [self.retriever_tool, web_search_tool]
        self.config = {"configurable": {"thread_id": "1"}}
        self.maingraph = Supervisor(llm, self.tools, memory, self.config)
        self.agent = self.maingraph.graph

    @staticmethod
    def build_retriever() -> tuple[IndexBuilder, Tool]:

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
            response_format="content_and_artifact"
        )
        return builder, retriever_tool


    def respond(self, chatbot: list, user_input, input_audio_block=None):
        message = user_input['text']
        response = self.agent.invoke({"messages": [message]}, self.config)
        chatbot.append((message, response['messages'][-1].content))
        yield chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), ""

    def process_uploaded_files(self, files_dir: list, chatbot: list) -> tuple:
        """
        Process uploaded files to prepare a VectorDB.

        Parameters:
            files_dir (List): List of paths to the uploaded files.
            chatbot (List): An instance of the chatbot for communication. A list of tuples containing the chat history.

        Returns:
            Tuple: A tuple containing an empty string and the updated chatbot instance.
        """
        docs = []
        for f in files_dir:
            docs.extend(DocumentProcessor.load_pdf(f))
        self.builder.pull_documents(docs)
        chatbot.append(
            (None, "Uploaded files are ready. Please ask your question"))
        return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])

    def process_selected_options(self, options: list, top_k: int, chatbot: list) -> tuple:

        #Change config arguments
        config.top_k = top_k
        if "reranking" in options:
            config.reranking = True
        else:
            config.reranking = False

        if "check hallucinations" in options:
            config.hallucinations = True
        else:
            config.hallucinations = False

        #rebuild retriever and refresh graph
        try:
            retriever = self.builder.build_retriever()

            self.retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_research_papers",
                RETRIEVER_TOOL_PROMPT,
                response_format = "content_and_artifact"
            )
            self.tools=[self.retriever_tool, web_search_tool]
        except Exception as e:
            raise RuntimeError(f"Error refreshing rag: {e}")
        self.maingraph.refresh(llm, self.tools, memory, self.config)

        chatbot.append(
            (None, "RAG refreshed. Please ask your question"))
        return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])
