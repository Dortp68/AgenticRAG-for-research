from langgraph.graph import MessagesState
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from agents.sub_graph import ChatAgent, AgenticRAG, EssayWriter


# def create_supervisor(llm, tools):
#     chat_agent = ChatAgent(llm).graph
#     rag_agent = AgenticRAG(llm, tools).graph
#     essay_writer_agent = EssayWriter(llm, rag_agent).graph
#
#     @tool
#     def chat_agent_tool(query: str):
#         """Answer on general user query"""
#         response = chat_agent.invoke({"messages": [query]})
#         return response['messages'][-1]
#
#     @tool
#     def rag_agent_tool(query: str):
#         """Performs websearch and search information in vectorstore from research papers on neural network architectures, large language models, and new developments in this area."""
#         response = rag_agent.invoke({"messages": [query]})
#         return response['messages'][-1]
#
#     @tool
#     def essay_writer_tool(query: str):
#         """This tool writes a detailed answer to a question or essay on a user-defined topic"""
#         response = essay_writer_agent.invoke({"task": query})
#         return response["draft"]
#
#     agent = create_react_agent(
#         model=llm,
#         tools=[chat_agent_tool, rag_agent_tool, essay_writer_tool]
#         # prompt="You are a team supervisor managing a research expert and a math expert. "
#         # "For current events, use research_agent. "
#         # "For math problems, use math_agent."
#     )
#     return agent


from langgraph.checkpoint.memory import MemorySaver

class Supervisor:
    def __init__(self, llm, tools, memory, config, system=""):
        print("---COMPILING MAIN GRAPH---")

        chat_agent = ChatAgent(llm, memory).graph
        rag_agent = AgenticRAG(llm, tools).graph
        essay_writer_agent = EssayWriter(llm, rag_agent).graph

        @tool
        def chat_agent_tool(query: str) -> str:
            """Answer on general user query. If you call this tool do NOT change user query, pass the whole query"""
            response = chat_agent.invoke({"messages": [query]}, config)
            return response['messages'][-1].content

        @tool
        def rag_agent_tool(query: str) -> str:
            """Performs websearch and search information in vectorstore from research papers on neural network architectures, large language models, and new developments in this area."""
            response = rag_agent.invoke({"messages": [query]})
            return response['messages'][-1].content

        @tool
        def essay_writer_tool(query: str) -> str:
            """This tool writes a detailed answer to a question or essay on a user-defined topic"""
            response = essay_writer_agent.invoke({"task": query})
            return response["draft"]

        self.llm = llm
        self.system=system
        self.tools = [chat_agent_tool, rag_agent_tool, essay_writer_tool]

        builder = StateGraph(MessagesState)
        builder.add_node("agent", self.agent)
        builder.add_node("tools", ToolNode(self.tools))


        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            ["tools", END],
        )
        builder.add_edge("tools", END)
        self.graph = builder.compile(checkpointer=memory)



    def agent(self, state: MessagesState):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---CALL SUPERVISOR---")
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        return {"messages": [response]}














