from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from pydantic import BaseModel, Field

from langgraph.graph import MessagesState
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.types import Command

from typing import Literal

from utils.prompts import DOC_GRADER_PROMPT, RAG_PROMPT
from utils.utils import web_search_text
from retriever import DocumentProcessor

class DocGradeScore(BaseModel):
    """Binary score that expresses the relevance of the document to the user's question"""

    binary_score: str = Field(None, description="Relevance score 'yes' or 'no'")

class GraphState(MessagesState):
    context: str
    query: str

class AgenticRAG:
    def __init__(self, llm, tools, memory=None, system=""):
        self.llm = llm
        self.system = system
        self.tools = tools

        builder = StateGraph(MessagesState)
        builder.add_node("agent", self.agent)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("generate", self.generate_answer)
        builder.add_node("grader", self.grade_documents)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            ["tools", END],
        )
        builder.add_conditional_edges("tools", self.edge_condition)
        builder.add_edge("generate", END)
        self.graph = builder.compile()

    def agent(self, state: MessagesState):
        """
            Invokes the agent model to generate a response based on the current state. Given
            the question, it will decide to retrieve using the retriever tool, or simply end.

            Args:
                state (messages): The current state

            Returns:
                dict: The updated state with the agent response appended to messages
            """
        print("---CALL AGENT---")
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)

        return {"messages": [response]}

    def grade_documents(self, state: MessagesState) -> Command[Literal["generate", "tools"]]:
        """
            Determines whether the retrieved documents are relevant to the question.

            Args:
                state (messages): The current state

            Returns:
                str: A decision for whether the documents are relevant or not
            """

        print("---CHECK RELEVANCE---")
        # LLM with tool and validation
        llm_with_tool = self.llm.with_structured_output(DocGradeScore, method="json_schema")

        question = state["messages"][0].content
        docs = state["messages"][-1].content


        prompt = DOC_GRADER_PROMPT.format(context=docs, question=question)
        score = llm_with_tool.invoke(prompt).binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return Command(goto="generate", update={"messages": docs})

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            msg = AIMessage(content="", tool_calls=[
                            {'name': 'web_search_tool', 'args': {'query': question},
                            'id': '41d01da6-534d-4aae-824c-b4014ec87e10', 'type': 'tool_call'}])

            return Command(goto="tools", update={"messages": msg})

    def edge_condition(self, state: MessagesState):

        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            last_tool = last_message.name
        else:
            raise RuntimeError("edge_conditions: tool call error")
        if last_tool == "retrieve_research_papers":
            return "grader"
        elif last_tool == "web_search_tool":
            return "generate"
        else:
            raise RuntimeError("edge_conditions")

    def generate_answer(self, state: MessagesState):
        print("---GENERATE---")
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = RAG_PROMPT.format(context=context, question=question)
        print(prompt)
        response = self.llm.invoke(prompt)
        return{"messages": [response]}











