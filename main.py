from retriever import get_retriever_tool, web_search_tool, general_query_tool
from langchain_ollama import ChatOllama

from utils import config
llm = ChatOllama(model=config.llm, temperature=0)


retriever_tool = get_retriever_tool()
from agents.sub_graph import EssayWriter
d = EssayWriter(llm, retriever_tool).graph
response = d.invoke({"task": "Whats the difference between titan architecture and transformers?"})
print(response["draft"])
exit()


tools = [retriever_tool, web_search_tool]

from agents.main_graph import AgenticRAG
agent = AgenticRAG(llm, tools).graph
query = "Whats the difference between titan architecture and transformers?"
query1 = "Give me 5 reasons to visit Canada"
gquery = "Hello"
response = agent.invoke({"messages": [query]})
print(response["messages"])

