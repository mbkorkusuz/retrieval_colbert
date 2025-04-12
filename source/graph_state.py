from data_models_and_functions import retrieve_best_match, generate_answer, decide_question_relevance, unrelated_question_answer, rerank

## Nodes
def retrieve(state):
    print("---RETRIEVING---")

    question = state["question"]
    
    documents = retrieve_best_match(query=question)
    
    top_documents = rerank(query= question, documents=documents)

    state["documents"] = top_documents
    
    return state


def generate(state):
    print("---GENERATING ANSWER---")

    question = state["question"]
    documents = state["documents"]

    document = "\n".join(documents)


    generation = generate_answer(question=question, document=document)
    state["generation"] = generation

    return state



def exit_with_excuse(state):
    print("---EXITING WITH EXCUSE---")

    question = state["question"]

    # history iptal.
    #history = state.get("history", [])

    #chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    #chat_history = f"{chat_context}\n\n"

    generation = unrelated_question_answer(question=question)

    state["generation"] = generation

    return state


## Edges

def is_question_related(state):
    print("---CHECKING FOR QUESTION RELEVANCE---")

    question = state["question"]

    answer = decide_question_relevance(question=question)

    if "evet" in answer:
        return "retrieve"
    else:
        return "exit_message"
    