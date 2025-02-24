from retriever import get_retriever_tool, web_search_tool, general_query_tool
from langchain_ollama import ChatOllama

from utils import config
llm = ChatOllama(model=config.llm, temperature=0)


retriever_tool = get_retriever_tool()



tools = [retriever_tool, web_search_tool]

from agents.main_graph import AgenticRAG
agent = AgenticRAG(llm, tools).graph

from agents.sub_graph import EssayWriter
d = EssayWriter(llm, agent).graph
response = d.invoke({"task": "How Does the Attention Mechanism Work and Why Is It So Important for LLMs?"})
print(response["draft"])
exit()

query = "Whats the difference between titan architecture and transformers?"
query1 = "Give me 5 reasons to visit Canada"
gquery = "Hello"
response = agent.invoke({"messages": [query]})
print(response["messages"][-1].content)

