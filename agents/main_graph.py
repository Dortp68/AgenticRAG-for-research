from langgraph.graph import MessagesState
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from agents.sub_graph import ChatAgent, AgenticRAG, EssayWriter

from pydantic import BaseModel
class ToolInput(BaseModel):
    query: str


class Supervisor:
    def __init__(self, llm, tools, memory, config, system=""):
        print("---COMPILING MAIN GRAPH---")

        chat_agent = ChatAgent(llm, memory).graph
        rag_agent = AgenticRAG(llm, tools).graph
        essay_writer_agent = EssayWriter(llm, rag_agent).graph

        @tool(args_schema=ToolInput)
        def chat(query: str) -> str:
            """Answer on general user query. If you call this tool do NOT change user query, pass the whole query."""
            print(query)
            response = chat_agent.invoke({"messages": [query]}, config)
            return response['messages'][-1].content

        @tool(args_schema=ToolInput)
        def research_assistant(query: str) -> str:
            """
            A research assistant tool that searches for information in a database and the internet.
            If you call this tool do NOT change user query, pass the whole query.
            """
            # Performs websearch and search information in vectorstore from research papers on neural network architectures, large language models, and new developments in this area.
            print(query)
            response = rag_agent.invoke({"messages": [query]})
            return response['messages'][-1].content

        @tool(args_schema=ToolInput)
        def essay_writer(query: str) -> str:
            """This tool writes a detailed answer to a question or essay on a user-defined topic"""
            print(query)
            response = essay_writer_agent.invoke({"task": query})
            return response["draft"]

        self.llm = llm
        self.system=system
        self.tools = [chat, research_assistant, essay_writer]

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
        model = self.llm.bind_tools(self.tools, tool_choice="auto")
        response = model.invoke(messages)
        print(response)
        return {"messages": [response]}

    def refresh(self, llm, tools, memory, config, system=""):
        self.__init__(llm, tools, memory, config, system)











