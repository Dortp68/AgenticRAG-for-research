
from retriever import get_retriever_tool, web_search_tool
from langchain_ollama import ChatOllama

from utils import config
llm = ChatOllama(model=config.llm, temperature=0)
retriever_tool = get_retriever_tool()


tools = [retriever_tool, web_search_tool]

from agents.main_graph import AgenticRAG
agent = AgenticRAG(llm, tools).graph
query = "Whats the difference between titan architecture and transformers?"
query2 = "Give me 5 reasons to visit Canada"
response = agent.invoke({"messages": [query]})
print(response["messages"])


