from langgraph.graph import START, END, StateGraph

from graph_state import retrieve, generate, is_question_related, exit_with_excuse

from typing import List
from typing_extensions import TypedDict

import os

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    #history = List[dict]


def main():
    workflow = StateGraph(GraphState)

    # Nodelar
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("exit_with_excuse", exit_with_excuse)

    # Edgeler
    workflow.add_conditional_edges(START, is_question_related, {"retrieve": "retrieve", "exit_message": "exit_with_excuse"})
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("exit_with_excuse", END)

    app = workflow.compile()
    
    return app

    question = "Öğretmen atamaları kim tarafından yapılır?"
    input = {"question": question}

    output = app.invoke(input)

    print(output["generation"])


if __name__ == "__main__":
    main()