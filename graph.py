from typing import Annotated, List

from IPython.display import Image, display
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Importing necessary components from other modules
from chain import grade_prompt, llm, rag_chain
from retriever import retriever


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The question being asked.
        generation: The generated answer from the LLM.
        search: Indicates whether to add search functionality.
        documents: List of documents retrieved.
        steps: List of actions taken in the process, such as 'retrieve_documents' and 'generate_answer'.
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


# LLM with tool call
structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader


def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}


def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    return {"documents": documents, "question": question, "generation": generation, "steps": steps}


def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "documents": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "search": search, "steps": steps}


def web_search(state):
    web_search_tool = TavilySearchResults()
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    web_results = web_search_tool.invoke({"query": question})
    documents.extend([Document(page_content=d["content"], metadata={"url": d["url"]}) for d in web_results])
    return {"documents": documents, "question": question, "steps": steps}


def decide_to_generate(state):
    search = state["search"]
    return "search" if search == "Yes" else "generate"


# Graph setup
def create_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    graph = workflow.compile()

    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

    return graph


custom_graph = create_graph()
