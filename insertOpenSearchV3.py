import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import numpy as np
import mariadb
import openai
import logging
import time

# Set OpenAI key
openai.api_key = ""

# Connect to ChromaDB running on port 9200
client = chromadb.HttpClient(host="172.31.30.137", port=9200)
collection = client.get_or_create_collection("hartnell")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MariaDB config (unchanged)
mariadb_config = {
    "host": "172.31.30.137",
    "user": "root",
    "password": "H*W7]nD-C(4:#EfsV?MA5G$bQ",
    "port": 3306,
    "database": "hartnell_scraped_data"
}

def fetch_scraped_data():
    try:
        conn = mariadb.connect(**mariadb_config)
        cursor = conn.cursor()
        cursor.execute("SELECT id, url, content FROM big_scraped_data")
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except mariadb.Error as e:
        logger.error(f"Error fetching data: {e}")
        return []
    finally:
        if conn:
            conn.close()

def generate_openai_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def bulk_store_embeddings_in_chroma():
    data = fetch_scraped_data()
    if not data:
        logger.error("No data fetched from MariaDB.")
        return

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for doc_id, url, text in data:
        try:
            emb = generate_openai_embedding(text)
            if emb is None or len(emb) != 1536:
                logger.warning(f"Skipping doc {doc_id}: invalid embedding.")
                continue

            ids.append(str(doc_id))
            embeddings.append(emb.tolist())
            documents.append(text)
            metadatas.append({"url": url})
        except Exception as e:
            logger.error(f"Error generating embedding for {doc_id}: {e}")

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(ids)} documents in Chroma.")
    else:
        logger.warning("No valid documents to store.")

# Run the flow
if __name__ == "__main__":
    logger.info("Storing embeddings into ChromaDB...")
    bulk_store_embeddings_in_chroma()
    logger.info("success!")
