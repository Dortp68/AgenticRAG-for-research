
from retriever import get_retriever_tool, web_search_tool
from langchain_ollama import ChatOllama

from utils import config
llm = ChatOllama(model=config.llm, temperature=0)

retriever_tool = get_retriever_tool()
tools = [retriever_tool, web_search_tool]

from agents.main_graph import Supervisor
config = {"configurable": {"thread_id": "1"}}
s = Supervisor(llm, tools).graph
response = s.invoke({"messages": ["Hello, My name is Misha"]}, config)
print(response)
print(response["messages"][-1])

response = s.invoke({"messages": ["Whats the difference between titan architecture and transformers?"]}, config)
print(response)
print(response["messages"][-1])

response = s.invoke({"messages": ["What is my name?"]}, config)
print(response)
print(response["messages"][-1])
exit()

from agents.main_graph import create_supervisor
s = create_supervisor(llm, tools)
response = s.invoke({"messages": ["Write an essay on this topic: Advantages and Limitations of Transformers Compared to RNNs and CNNs"]})
print(response)

print(response["messages"][-1])
print(response["messages"][-2])
exit()

from agents.sub_graph import ChatAgent
g = ChatAgent(llm).graph


from agents.sub_graph import AgenticRAG
agent = AgenticRAG(llm, tools).graph


response = agent.invoke({"messages": ["Whats the difference between titan architecture and transformers?"]})
print(response["messages"][-1])
exit()

from agents.sub_graph import EssayWriter
d = EssayWriter(llm, agent).graph
response = d.invoke({"task": "Advantages and Limitations of Transformers Compared to RNNs and CNNs"})
print(response["draft"])
exit()

query = "Whats the difference between titan architecture and transformers?"
query1 = "Give me 5 reasons to visit Canada"
gquery = "How Does the Attention Mechanism Work and Why Is It So Important for LLMs?"
response = agent.invoke({"messages": [query1]})
print(response["messages"][-1].content)

