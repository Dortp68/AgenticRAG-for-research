from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_core.messages import (AnyMessage,
                                     SystemMessage,
                                     HumanMessage,
                                     AIMessage,
                                     ChatMessage,
                                     ToolMessage)

from utils.prompts import (PLAN_PROMPT,
                           WRITER_PROMPT,
                           REFLECTION_PROMPT,
                           RESEARCH_PLAN_PROMPT,
                           RESEARCH_CRITIQUE_PROMPT,
                           DOC_GRADER_PROMPT,
                           RAG_PROMPT)


class DocGradeScore(BaseModel):
    """Binary score that expresses the relevance of the document to the user's question"""
    binary_score: str = Field(None, description="Relevance score 'yes' or 'no'")


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
        print("---CALL RAG AGENT---")
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
        response = self.llm.invoke(prompt)
        return{"messages": [response]}


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]

class EssayWriter:

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)

        builder.add_edge(START,"planner")
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("research_plan", "generate")
        self.graph = builder.compile()

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        response = self.llm.invoke(messages)
        return {"plan": response.content}

    def research_plan_node(self, state: AgentState):
        queries = self.llm.with_structured_output(Queries, method="json_schema").invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])

        content = state.get('content') or []
        for q in queries.queries:
            print("//////////////")
            print(q)
            print("//////////////")
            response = self.retriever.invoke({"messages": [q]})
            r = response["messages"][-1].content
            print(r)
            content.append(r)
        return {"content": content}

    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [
            SystemMessage(
                content=WRITER_PROMPT.format(content=content)
            ),
            user_message
        ]
        response = self.llm.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1
        }


class ChatAgent:
    def __init__(self, llm, system=""):
        self.llm = llm
        self.system = system

        builder = StateGraph(MessagesState)

        builder.add_edge(START, "agent")
        builder.add_node("agent", self.call_llm)
        builder.add_edge("agent", END)
        self.graph = builder.compile()

    def call_llm(self, state: MessagesState):
        print("---CALL CHAT AGENT---")
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        response = self.llm.invoke(messages)
        return {"messages": [response]}