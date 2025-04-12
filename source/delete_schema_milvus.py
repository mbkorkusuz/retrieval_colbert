from pymilvus import connections, Collection

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="colbert_token_embeddings")
collection.drop()
print("Milvus koleksiyonu silindi!")
