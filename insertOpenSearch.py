import mariadb
import numpy as np
import cohere
import faiss
import os

#RUN THIS ONLY ONCE TO CREATE LOCAL VECTORDB

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

# FAISS index path
FAISS_INDEX_PATH = "faiss_index.idx"

def fetch_scraped_data():
    """Fetch data from MariaDB."""
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

def generate_embeddings(texts):
    """Generate embeddings using Cohere."""
    co = cohere.Client(COHERE_API_KEY)
    response = co.embed(
        texts=texts, 
        model="embed-english-v3.0", 
        input_type="search_document"
    )
    return np.array(response.embeddings, dtype=np.float32)

def insert_into_faiss(data):
    """Insert vectorized data into FAISS."""
    ids, urls, contents, vectors = zip(*data)
    vectors = np.array(vectors, dtype=np.float32)

    # Initialize or load FAISS index
    dimension = vectors.shape[1]
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(dimension)

    index.add(vectors)  # Add embeddings to FAISS
    faiss.write_index(index, FAISS_INDEX_PATH)  # Save index

    print(f"Inserted {len(data)} documents into FAISS.")
    print(f"Total records in FAISS: {index.ntotal}")

def main():
    rows = fetch_scraped_data()
    if not rows:
        print("No data to process.")
        return

    ids, urls, texts = zip(*rows)
    vectors = generate_embeddings(texts)
    print(f"Created {len(vectors)} vectors")

    data_to_insert = list(zip(ids, urls, texts, vectors))
    insert_into_faiss(data_to_insert)

if __name__ == "__main__":
    main()
