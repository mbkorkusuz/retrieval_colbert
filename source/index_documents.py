from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from jina import Executor
import torch
import os
from transformers import AutoModel, AutoTokenizer
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title


os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = "../data"


class ColBERTEmbedder(Executor):
    def __init__(self, model_name = "jinaai/jina-colbert-v2", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code = True)
        
    def get_token_embeddings(self, texts, batch_size=8):
        all_token_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            for i, text in enumerate(batch):
                token_embs = outputs.last_hidden_state[i].cpu().numpy()  # [seq_len, hidden_size]
                attention = inputs["attention_mask"][i].cpu().numpy()
                valid_token_embs = token_embs[attention == 1]
                all_token_embeddings.append(valid_token_embs)
        return all_token_embeddings


def connect_to_milvus():
    print("Milvusa bağlanılıyor...")
    connections.connect("default", host = "localhost", port = "19530")

def create_collection(collection_name):
    print("Koleksiyon oluşturuluyor...")

    if utility.has_collection(collection_name):## Eğer daha öncesinden collection_name adında bir collection varsa direkt onu kullan
        print(f"Var olan '{collection_name}' koleksiyonu yükleniyor...")
        collection = Collection(name=collection_name)
    else: ## Eğer yoksa schemayı belirleyip collection oluştur
        fields = [
                    FieldSchema(name="uid", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
                    FieldSchema(name="token_index", dtype=DataType.INT64),
                    FieldSchema(name="token_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=8192),
                ]


        schema = CollectionSchema(fields, description="Token-level embeddings for ColBERT")
        
        collection = Collection(name = collection_name, schema=schema) # Collection oluşturuldu. Columnlar tablo haline geldi gibi düşünülebilir.

        print(f"Yeni koleksiyon oluşturuldu: {collection_name}")

    if not collection.has_index():
        print("Embedding için indeksler oluşturuluyor...") ## bu ne demek????
        collection.create_index(field_name="token_embedding", index_params={"metric_type": "IP", "index_type": "FLAT"})
        

    collection.load() # tablo yüklendi



def chunk_and_process(file_name, embedder):
    print(f"{file_name} işleniyor...")
    
    file_path = os.path.join(DATA_DIR, file_name)
    base_doc_id = file_name

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        languages=["tur"],
        infer_table_structure = True,
        skip_infer_table_structure = False,
    )

    chunks = chunk_by_title(elements=elements, max_characters=500, overlap=0) ## en iyi 500 karakterken çalışıyor.
    data = [element.text for element in chunks]

    print(f"{file_name} chunklara ayrıldı. Şimdi token embeddingler hesaplanıyor ve Milvus'a gönderiliyor...")

    
    doc_ids = [f"{base_doc_id}_chunk{i}" for i in range(len(data))]

    insert_token_embeddings_to_milvus(embedder, data, doc_ids)
    


def insert_token_embeddings_to_milvus(embedder, documents, doc_ids):
    all_token_embeddings = embedder.get_token_embeddings(documents)

    milvus_data = {
        "doc_id": [],
        "token_index": [],
        "token_embedding": [],
        "chunk_text": [],
    }

    for doc_id, chunk_text, token_embs in zip(doc_ids, documents, all_token_embeddings):
        for idx, emb in enumerate(token_embs):
            milvus_data["doc_id"].append(doc_id)
            milvus_data["token_index"].append(idx)
            milvus_data["token_embedding"].append(emb.tolist())
            milvus_data["chunk_text"].append(chunk_text)

    collection = Collection("colbert_token_embeddings")
    collection.insert([
        milvus_data["doc_id"],
        milvus_data["token_index"],
        milvus_data["token_embedding"],
        milvus_data["chunk_text"],
    ])
    collection.flush()




def main():
    collection_name = "colbert_token_embeddings"
    embedder = ColBERTEmbedder()
    connect_to_milvus()

    create_collection(collection_name)

    for file_name in os.listdir(DATA_DIR):
        if not file_name.endswith(".pdf"):
            continue
        chunk_and_process(file_name = file_name, embedder = embedder)
    



if __name__ == "__main__":
    main()