from langchain_core.prompts import PromptTemplate
from milvus_client import MilvusClient
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder
import numpy as np

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2") ## context window 512

MODEL_BASE_URL = "https://9jb4dhts2y4rur-8000.proxy.runpod.net"
MODEL_NAME = "c4ai-command-r-plus-GPTQ"

llm = BaseChatOpenAI(
    openai_api_base = f"{MODEL_BASE_URL}/v1",
    model_name = MODEL_NAME,
    streaming=True,
    api_key = "xxx",
    top_p = 0.75,
    temperature=1.3,
    max_tokens=200
)

llm_for_yes_no = BaseChatOpenAI(
    openai_api_base = f"{MODEL_BASE_URL}/v1",
    model_name = MODEL_NAME,
    streaming = True,
    api_key = "xxx",
    top_p = 0.2,
    temperature= 0.1,
    max_tokens= 5
)

system_message_for_generate_answer = SystemMessage(content="""Sen Milli Eğitim Bakanlığı (MEB) yönetmeliği ile alakalı soruları yanıtlayan bir AI asistanısın.\n Cevabın 200 tokeni aşmasın.""")
#system_message_for_generate_answer = SystemMessage(content="""You are an AI assistant that answers questions related to the regulations of the Ministry of National Education in Turkey.\n Your response should not exceed 200 tokens.""")

system_message_for_relevance_check = SystemMessage(content="Sen sorulan sorunun Milli Eğitim Bakanlığı (MEB) yönetmeliği ile alakalı olup olmadığını tespit eden bir AI asistanısın.")
#system_message_for_relevance_check = SystemMessage(content="You are an AI assistant that determines whether the given question is related to the regulations of the Ministry of National Education in Turkey.")

system_message_for_unrelated_question = SystemMessage(content= """Sen Milli Eğitim Bakanlığı (MEB) yönetmeliği ile alakalı soruları yanıtlayan bir AI asistanısın.\nEğer soru MEB yönetmeliğiyle alakasız ise kullanıcıya sorunun alakasız olduğunu belirt.\nEğer soru ile beraber bir döküman örneği verilmemişse sorunun senin alanında olmadığını kullanıcıya belirt.""")
#system_message_for_unrelated_question = SystemMessage(content="""You are an AI assistant that answers questions related to the regulations of the Ministry of National Education in Turkey.\nIf the question is unrelated to the regulations, inform the user that the question is not relevant.\nIf no document is provided with the question, inform the user that the question is outside your area of expertise.\nIf the question refers to previous conversation, respond using the information gathered from that context.""")

client = MilvusClient() # client initialized.
generate_prompt_template = """
Aşağıda bir soru ve bir döküman göreceksin. Soruyu dökümandan aldığın bilgilerle cevapla.\n

Soru:\n
{user_question}

Döküman:\n
{document_text}
\n
Cevap:
"""

question_relevant_prompt_template = """
Aşağıda bir soru göreceksin. Eğer bu soru MEB yönetmeliği ile alakalıysa 'evet', MEB yönetmeliği ile alakalı değilse 'hayır' cevabını ver. Açıklama yapma.\n

Soru:\n
{user_question}
\n
Cevap:
"""


unrelated_question_prompt_template = """
Aşağıdaki sorunun MEB yönetmeliği ile alakası yoksa kullanıcıya bunu belirten bir mesaj yaz.\n

Soru: {user_question}
"""

generate_prompt = PromptTemplate.from_template(generate_prompt_template)
question_relevant_prompt = PromptTemplate.from_template(question_relevant_prompt_template)
unrelated_question_prompt = PromptTemplate.from_template(unrelated_question_prompt_template)


def decide_question_relevance(question): ## sorulan soru MEble alakalı mı değil mi diye bakıyor
    formatted_prompt = question_relevant_prompt.format(user_question = question)
    messages = [system_message_for_relevance_check, formatted_prompt]

    output = llm_for_yes_no.invoke(messages).content.lower()
    

    return output


def unrelated_question_answer(question): # Sorulan soru MEBle alakalı değil ise bunu kullanıcıya belirten bir cevap döndürülüyor
    formatted_prompt = unrelated_question_prompt.format(user_question=question)
    messages = [system_message_for_unrelated_question, formatted_prompt]
    
    output = llm.invoke(messages).content

    return output



def retrieve_best_match(query): ## milvus clientdan alakalı chunkları çekiyor
    print(f"Kullanıcı sorusu için dökümanlar aranıyor: {query}")
    
    results = client.search(query=query, top_k=10)
    
    return results


def rerank(query, documents): # retrieve edilen chunkları en alakalısından en az alakalısına doğru sıralayıp en yüksek 3 puana sahip olanı döndürüyor.
    # ms marco l12 v2
    # context size 512 token.
    
    cross_scores = cross_encoder.predict([(query, doc) for doc in documents], show_progress_bar=True)

    cross_scores = np.array(cross_scores)

    sorted_score_indices = np.argsort(cross_scores)[::-1]

    top_documents = [documents[i] for i in sorted_score_indices[0:2]] # en yüksek skora sahip 2 chunkı döndürüyor. LLM in context size nı aşmamak için

    print(top_documents)

    return top_documents


def generate_answer(question, document): # Seçilen chunklar ve kullanıcı sorusu llm'e input olarak verilip cevap oluşturuluyor
    formatted_prompt = generate_prompt.format(user_question = question, document_text = document)
    messages = [system_message_for_generate_answer, formatted_prompt]
    
    output = llm.invoke(messages).content

    return output




