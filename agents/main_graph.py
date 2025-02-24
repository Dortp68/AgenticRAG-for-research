from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool, BaseTool, InjectedToolCallId

from pydantic import BaseModel, Field

from langgraph.graph import MessagesState
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.types import Command

from typing import Literal, TypedDict

from utils.prompts import DOC_GRADER_PROMPT, RAG_PROMPT, ROUTER_PROMPT
from utils.utils import web_search_text
from retriever import DocumentProcessor

from agents.sub_graph import ChatAgent, AgenticRAG
from langgraph.prebuilt import create_react_agent

def create_supervisor(llm, tools):
    chat_agent = ChatAgent(llm).graph
    rag_agent = AgenticRAG(llm, tools).graph

    @tool
    def chat_agent_tool(query: str):
        """Answer on general user query"""
        response = chat_agent.invoke({"messages": [query]})
        return response['messages'][-1]

    @tool
    def rag_agent_tool(query: str):
        """Performs websearch and search information in vectorstore from research papers on neural network architectures, large language models, and new developments in this area."""
        response = rag_agent.invoke({"messages": [query]})
        return response['messages'][-1]

    agent = create_react_agent(
        model=llm,
        tools=[chat_agent_tool, rag_agent_tool],
        # prompt="You are a team supervisor managing a research expert and a math expert. "
        # "For current events, use research_agent. "
        # "For math problems, use math_agent."
    )
    return agent

class Supervisor:
    def __init__(self, llm, tools, memory=None, system=""):
        self.llm = llm
        self.system = system
        self.chat_agent = ChatAgent(self.llm).graph
        self.rag_agent = AgenticRAG(self.llm, tools).graph

        self.tools = [self.chat_agent_tool, self.rag_agent_tool]


    @tool
    def chat_agent_tool(self, query: str):
        """Answer on general user query"""
        response = self.chat_agent.invoke({"messages": [query]})
        return response['messages'][-1]

    @tool
    def rag_agent_tool(self, query: str):
        """Performs websearch and search information in vectorstore from research papers on neural network architectures, large language models, and new developments in this area."""
        response = self.rag_agent.invoke({"messages": [query]})
        return response['messages'][-1]

    def agent(self, state: MessagesState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)

        return {"messages": [response]}















