from openai import OpenAI
import numpy as np
from opensearchpy import OpenSearch
import mariadb
import os
import urllib3

# Disable SSL warnings (use cautiously in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# OpenAI Configuration
# ----------------------------
OPENAI_API_KEY = "sk-proj-..."  # Store securely in environment variables or secret managers
openAI_client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# OpenSearch Configuration
# ----------------------------
OPENSEARCH_HOST = "172.31.30.137"
OPENSEARCH_AUTH = ("admin", "H@RTn311_ROCKS")
OPENSEARCH_INDEX = "knn_vector_index"

client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=OPENSEARCH_AUTH,
    use_ssl=True,
    verify_certs=False  # Set to True with proper certs in production
)

# Verify OpenSearch connection
try:
    response = client.info()
    print("Connected to OpenSearch:", response)
except Exception as e:
    print("Error connecting to OpenSearch:", e)

# ----------------------------
# Create OpenSearch k-NN Index
# ----------------------------
def makeIndex(indexName):
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "url": {"type": "text"},
                "content": {"type": "text"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                }
            }
        }
    }

    if not client.indices.exists(index=indexName):
        client.indices.create(index=indexName, body=index_body)
        print(f"Index '{indexName}' created successfully.")
    else:
        print(f"Index '{indexName}' already exists.")

# ----------------------------
# MariaDB Configuration
# ----------------------------
mariadb_config = {
    "host": "172.31.30.137",
    "user": "root",
    "password": "H*W7]nD-C(4:#EfsV?MA5G$bQ",
    "port": 3306,
    "database": "hartnell_scraped_data"
}

# Fetch scraped data from MariaDB
def fetch_scraped_data():
    try:
        conn = mariadb.connect(**mariadb_config)
        cursor = conn.cursor()
        cursor.execute("SELECT id, url, content FROM scraped_data")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except mariadb.Error as e:
        print(f"Error fetching data: {e}")
        return []

# Generate OpenAI embedding
def generate_openai_embedding(text):
    response = openAI_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# Store single document embedding
def store_embedding(doc_id, text, url):
    embedding = generate_openai_embedding(text)
    document = {
        "url": url,
        "content": text,
        "vector": embedding.tolist()
    }
    response = client.index(index=OPENSEARCH_INDEX, id=doc_id, body=document)
    print(f"Stored document {doc_id}: {response['result']}")

# Store multiple embeddings from database to OpenSearch
def store_embeddings_in_opensearch():
    data = fetch_scraped_data()
    if not data:
        print("No data fetched from MariaDB.")
        return

    for doc_id, url, text in data:
        embedding = generate_openai_embedding(text)
        document = {
            "url": url,
            "content": text,
            "vector": embedding.tolist()
        }
        response = client.index(index=OPENSEARCH_INDEX, id=str(doc_id), body=document)
    print(f"Number of embeddings stored: {len(data)}")

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

# Delete a single document by ID
def delDocId(index, id):
    client.delete(index=index, id=id)

# Delete entire index
def delIndex(indexName):
    client.indices.delete(index=indexName)

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    makeIndex(OPENSEARCH_INDEX)                # Create index if it doesn't exist
    store_embeddings_in_opensearch()           # Store all embeddings
    chatbot()                                  # Launch chatbot