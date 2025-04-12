def report_collection_stats():
    from pymilvus import Collection, connections

    connections.connect("default", host="localhost", port="19530")
    collection = Collection("colbert_token_embeddings")
    collection.load()

    token_count = collection.num_entities

    results = collection.query(
        expr="",
        output_fields=["doc_id"],
        limit=10000
    )
    doc_ids = set(r["doc_id"] for r in results)

    print(f"Toplam token embedding sayısı: {token_count}")
    print(f"Toplam farklı chunk (doc_id) sayısı: {len(doc_ids)}")

report_collection_stats()
