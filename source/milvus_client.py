from pymilvus import Collection, connections
from transformers import AutoModel, AutoTokenizer
import torch


class MilvusClient:
    def __init__(self, collection_name="colbert_token_embeddings", model_name="jinaai/jina-colbert-v2"):
        self.collection_name = collection_name
        connections.connect("default", host="localhost", port="19530")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.collection = Collection(self.collection_name)
        self.collection.load()

    def embed_query_tokens(self, query):
        inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]
        attention_mask = inputs["attention_mask"][0]
        valid_embeddings = token_embeddings[attention_mask == 1]
        return valid_embeddings

    def compute_scores(self, query_embs, doc_data):
        scores = []
        for doc_id, info in doc_data.items():
            doc_embs = torch.tensor(info["embeddings"])
            sim_matrix = torch.matmul(query_embs, doc_embs.T)
            max_sim = torch.max(sim_matrix, dim=1).values
            score = torch.sum(max_sim).item()
            scores.append((info["chunk_text"], score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores
    
    def first_step_search(self, query, top_k=1000):
        # Normal bir şekilde arama yapıyor. 
        # Aramanın birinci adımı. En yakın chunklara ait doc_idleri buluyor sadece
        
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].numpy().tolist()


        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 1000}
        }

        results = self.collection.search(
            data=query_embedding,
            anns_field="token_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id"]
        )

        doc_ids = set()
        for hits in results:
            for hit in hits:
                doc_ids.add(hit.entity.get("doc_id"))

        return list(doc_ids)
    
    def fetch_token_embeddings_for_doc_ids(self, doc_id_list):
        ## doc_idleri çekiyor.
        
        self.collection.load()
        doc_id_list_str = "[" + ", ".join(f'"{doc_id}"' for doc_id in doc_id_list) + "]"
        expr = f"doc_id in {doc_id_list_str}"

        results = self.collection.query(
            expr=expr,
            output_fields=["doc_id", "token_index", "token_embedding", "chunk_text"]
        )

        grouped_data = {}
        for r in results:
            doc_id = r["doc_id"]
            if doc_id not in grouped_data:
                grouped_data[doc_id] = {
                    "embeddings": [],
                    "chunk_text": r["chunk_text"],
                }
            grouped_data[doc_id]["embeddings"].append((r["token_index"], r["token_embedding"]))

        for doc_id in grouped_data:
            grouped_data[doc_id]["embeddings"] = [
                emb for _, emb in sorted(grouped_data[doc_id]["embeddings"])
            ]

        return grouped_data

    
    def search(self, query, top_k=10):
        top_doc_ids = self.first_step_search(query, top_k=1000)
        doc_data = self.fetch_token_embeddings_for_doc_ids(top_doc_ids)
        query_embs = self.embed_query_tokens(query)
        scored_chunks = self.compute_scores(query_embs, doc_data)

        chunk_texts = [element for element, _ in scored_chunks[:top_k]] # top_k dökümanı return et
        
        return chunk_texts
        #return scored_chunks[:top_k]


if __name__ == "__main__":
    client = MilvusClient()
    results = client.full_search("Öğretmen atamalarını kim yapar?")
    for i in results:
        print(i)
        print("\n\n-------------\n\n")

    
