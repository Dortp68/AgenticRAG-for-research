from duckduckgo_search import DDGS
from typing import Optional


def web_search_text(query: str, max_results: Optional[int] = 3) -> list[str]:
    """
    Search for text on duckduckgo.com.

    Args:
        query (str): The text to search for.
        max_results (Optional[int]): The maximum number of search results to retrieve (default 10).

    Returns:
        List of search results as strings.
    """
    print("---USING WEB SEARCH---")
    with DDGS() as ddgs:
        results = results = [r['href'] for r in ddgs.text(query, max_results=max_results)]
    return results


from retriever import get_retriever_tool, web_search_tool, DocumentProcessor
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from utils import config

llm = ChatOllama(model=config.llm, temperature=0)
memory = MemorySaver()
builder, retriever_tool = get_retriever_tool()
tools = [retriever_tool, web_search_tool]

from agents.main_graph import Supervisor

config = {"configurable": {"thread_id": "1"}}
agent = Supervisor(llm, tools, memory, config).graph

import gradio as gr


def respond(chatbot: list, user_input, input_audio_block=None):
    message = user_input['text']
    response = agent.invoke({"messages": [message]}, config)
    chatbot.append((message, response['messages'][-1].content))
    yield chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), ""


def process_uploaded_files(files_dir: list, chatbot: list) -> tuple:
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
    builder.pull_documents(docs)
    chatbot.append(
        (None, "Uploaded files are ready. Please ask your question"))
    return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])


