import urllib3
import storeOpenSearch

# Disable SSL warnings (use cautiously in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Search similar documents using k-NN
def search_similar_documents(query, top_k=4):
    query_embedding = generate_openai_embedding(query).tolist()
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        }
    }
    response = client.search(index=OPENSEARCH_INDEX, body=search_query)
    results = [(hit["_source"]["url"], hit["_score"]) for hit in response["hits"]["hits"]]
    return results

# Simple chatbot loop
def chatbot():
    print("\nHello! Ask me a question (type 'exit' to quit):")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        results = search_similar_documents(query)
        print("\nTop relevant sites:")
        for url, score in results:
            print(f"- {url} (score: {score:.2f})")
        print("\nAny other questions?")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":

    print("Removing Index")
    storeOpenSearch.delIndex()
    print("Making Index")
    storeOpenSearch.makeIndex()                # Create index if it doesn't exist
    print("Storing Embeddings")
    storeOpenSearch.store_embeddings_in_opensearch()           # Store all embeddings
    print("Starting Chatbot")
    chatbot()                                  # Launch chatbot