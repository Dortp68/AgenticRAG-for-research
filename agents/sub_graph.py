from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_core.messages import (SystemMessage,
                                     HumanMessage,
                                     AIMessage,
                                     ToolMessage,
                                     filter_messages)

from utils.prompts import (PLAN_PROMPT,
                           WRITER_PROMPT,
                           CHECK_HALLUCINATIONS,
                           RESEARCH_PLAN_PROMPT,
                           DOC_GRADER_PROMPT,
                           RAG_PROMPT)

from utils import config

class DocGradeScore(BaseModel):
    """Binary score that expresses the relevance of the document to the user's question"""
    binary_score: str = Field(None, description="Relevance score 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class RagState(MessagesState):
    question: str
    context: str
    last_tool: str

class AgenticRAG:
    def __init__(self, llm, tools, memory=None, system=""):
        self.llm = llm
        self.system = system
        self.tools = tools

        builder = StateGraph(RagState)
        builder.add_node("agent", self.agent)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("generate", self.generate_answer)
        builder.add_node("grader", self.grade_documents)
        builder.add_node("tool_condition", self.edge_condition)
        builder.add_node("hallucinations", self.check_hallucinations)


        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            ["tools", END],
        )
        builder.add_edge("tools", "tool_condition")
        self.graph = builder.compile()

    def agent(self, state: RagState):
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
        question = messages[0].content
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        return {"messages": [response], "question": question}

    def grade_documents(self, state: RagState) -> Command[Literal["generate", "tools"]]:
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

        question = state["question"]
        docs = state["messages"][-1].content

        prompt = DOC_GRADER_PROMPT.format(context=docs, question=question)
        score = llm_with_tool.invoke(prompt).binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return Command(goto="generate", update={"messages": docs})

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            msg = AIMessage(content="", tool_calls=[
                            {'name': 'web_search_tool', 'args': {'query': question},
                            'id': '41d01da6-534d-4aae-824c-b4014ec87e10', 'type': 'tool_call'}])

            return Command(goto="tools", update={"messages": msg})

    def edge_condition(self, state: RagState):
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            last_tool = last_message.name
        else:
            raise RuntimeError("edge_conditions: tool call error")
        if last_tool == "retrieve_research_papers":
            return Command(update={"last_tool": last_tool}, goto="grader")
        elif last_tool == "web_search_tool":
            return Command(update={"last_tool": last_tool}, goto="generate")
        else:
            raise RuntimeError("edge_conditions")

    def generate_answer(self, state: RagState):
        print("---GENERATE---")
        question = state["question"]
        context = state["messages"][-1].content
        prompt = RAG_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        if state["last_tool"] == "web_search_tool":
            return Command(update={"messages": [response]}, goto=END)
        else:
            return Command(update={"messages": [response], "context": context}, goto="hallucinations")

    def check_hallucinations(self, state: RagState):

        if config.hallucinations == False:
            return Command(goto=END)

        print("---CHECK HALLUCINATIONS---")
        system_prompt = CHECK_HALLUCINATIONS.format(
            documents=state["context"],
            query=state["question"],
            generation=state["messages"][-1]
        )
        response = self.llm.with_structured_output(GradeHallucinations, method="json_schema").invoke(system_prompt)
        response = response.binary_score
        if response == "yes":
            print("---NO HALLUCINATIONS---")
            return Command(goto=END)
        else:
            print("---HALLUCINATIONS TRUE---")
            print(state["question"])
            msg = AIMessage(content="", tool_calls=[
                {'name': 'web_search_tool', 'args': {'query': state["question"]},
                 'id': '41d01da6-534d-4aae-824c-b4014ec87333', 'type': 'tool_call'}])

            return Command(goto="tools", update={"messages": msg})


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
        print("---CALL ESSAY WRITER---")
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
            response = self.retriever.invoke({"messages": [q]})
            r = response["messages"][-1].content
            content.append(r)
        return {"content": content}

    def generation_node(self, state: AgentState):
        print("---GENERATE ESSAY---")
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
    def __init__(self, llm, memory, system="You are helpful assistant"):
        self.llm = llm
        self.system = system
        builder = StateGraph(MessagesState)
        builder.add_edge(START, "agent")
        builder.add_node("agent", self.call_llm)
        builder.add_edge("agent", END)
        self.graph = builder.compile(checkpointer=memory)

    def call_llm(self, state: MessagesState):
        print("---CALL CHAT AGENT---")
        messages = state["messages"]
        messages = filter_messages(messages, include_types=[HumanMessage, ToolMessage, AIMessage])
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        response = self.llm.invoke(messages)
        return {"messages": [response]}