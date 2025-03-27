import faiss
import cohere
import mariadb
import numpy as np

# FAISS index path
FAISS_INDEX_PATH = "faiss_index.idx"

# MariaDB Connection
mariadb_config = {
    "host": "172.31.30.137",
    "user": "root",
    "password": "H*W7]nD-C(4:#EfsV?MA5G$bQ",
    "port": 3306,
    "database": "hartnell_scraped_data"
}

# Cohere API Key
COHERE_API_KEY = "ekac8v5OlcEJ1TGGOFC30hzY73zwSqcQPfnv6hJN"

def generate_query_embedding(query):
    """Generate an embedding for the query using Cohere."""
    co = cohere.Client(COHERE_API_KEY)
    response = co.embed(
        texts=[query], 
        model="embed-english-v3.0", 
        input_type="search_query"
    )
    return np.array(response.embeddings, dtype=np.float32)

def search_faiss(query_embedding, top_k=4):
    """Search FAISS index for the most relevant documents."""
    index = faiss.read_index(FAISS_INDEX_PATH)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

def fetch_results(indices):
    """Retrieve URLs from MariaDB based on FAISS results."""
    try:
        conn = mariadb.connect(**mariadb_config)
        cursor = conn.cursor()
        results = []
        for idx in indices:
            cursor.execute("SELECT url FROM scraped_data WHERE id = ?", (idx,))
            row = cursor.fetchone()
            if row:
                results.append(row[0])
        cursor.close()
        conn.close()
        return results
    except mariadb.Error as e:
        print(f"Error fetching results: {e}")
        return []

def chatbot():
    print("Hello! Ask me a question: ")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        query_embedding = generate_query_embedding(query).reshape(1, -1)
        indices = search_faiss(query_embedding)
        results = fetch_results(indices)
        
        print("\nTop relevant sites to your query:")
        for url in results:
            print(f"- {url}")
        print()
        print("Any Other questions?")

if __name__ == "__main__":
    chatbot()
