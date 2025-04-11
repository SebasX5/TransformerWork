from openai import OpenAI
import numpy as np
from opensearchpy import OpenSearch
import mariadb
import urllib3
from opensearchpy.connection import RequestsHttpConnection
import logging
from elasticsearch.helpers import bulk


# Disable SSL warnings (use cautiously in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# OpenAI Configuration
# ----------------------------
OPENAI_API_KEY = "sk-proj-"  # Store securely in environment variables or secret managers
openAI_client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# OpenSearch Configuration
# ----------------------------
OPENSEARCH_HOST = "172.31.30.137"
OPENSEARCH_AUTH = ("admin", "H@RTn311_ROCKS")
# OPENSEARCH_INDEX = "knn_vector_index"
OPENSEARCH_INDEX = "mock_knn_vector_index"

client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=OPENSEARCH_AUTH,
    http_compress=True,
    use_ssl=True,
    verify_certs=False,  # Set to True with proper certs in production
    connection_class=RequestsHttpConnection,
    timeout=60,  # Set timeout to 30 seconds or longer
    retries=10,  # Increase retries
    max_retries=10  # Allow multiple retries
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Create OpenSearch k-NN Index
# ----------------------------
def makeIndexNamed(indexName):
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


# Function to fetch specific records by IDs 
def fetch_scraped_data_by_ids(ids):
    try:
        conn = mariadb.connect(**mariadb_config)
        cursor = conn.cursor()
        format_strings = ','.join(['%s'] * len(ids))
        cursor.execute(f"SELECT id, url, content FROM scraped_data WHERE id IN ({format_strings})", tuple(ids))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except mariadb.Error as e:
        logger.error(f"Error fetching specific data: {e}")
        return []


# Retry mechanism for fetching data from MariaDB
# def fetch_scraped_data_with_retry(max_retries=3):
#     retries = 0
#     while retries < max_retries:
#         try:
#             return fetch_scraped_data()
#         except mariadb.Error as e:
#             retries += 1
#             logger.error(f"Error fetching data (attempt {retries}): {e}")
#             time.sleep(2 ** retries)  # Exponential backoff
#     logger.error("Failed to fetch data after multiple retries.")
#     return []


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
    # data = fetch_scraped_data()
    data =fetch_mock_data()
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
        print(response)
    print(f"Number of embeddings stored: {len(data)}")


# Store documents in OpenSearch with retry and error handling
def bulk_store_embeddings_in_opensearch():
    # data = fetch_scraped_data_with_retry()
    # data = fetch_scraped_data()
    data=fetch_mock_data()
    if not data:
        logger.error("No data fetched from MariaDB.")
        return

    actions = []
    failed_docs = []

    # Prepare documents for bulk indexing
    for doc_id, url, text in data:
        embedding = generate_openai_embedding(text)
        document = {
            "_op_type": "index",
            "_index": OPENSEARCH_INDEX,
            "_id": str(doc_id),
            "_source": {
                "url": url,
                "content": text,
                "vector": embedding.tolist()
            }
        }
        actions.append(document)

    # Perform bulk indexing
    try:
        success, failed = bulk(client, actions)
        logger.info(f"Successfully indexed {success} documents.")
        
        if failed > 0:
            logger.error(f"Failed to index {failed} documents. Storing failed IDs.")
            # Track failed documents
            failed_docs= [error['index']['_id'] for error in failed if 'index' in error]

    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Error during bulk indexing: {e}")
    
    # Retry indexing for failed documents
    if failed_docs:
        logger.info(f"Retrying failed documents: {failed_docs}")
        store_failed_documents(failed_docs)


# Retry failed documents by fetching them and re-indexing
def store_failed_documents(failed_ids, max_retries=1):
    retry_count = 0
    failed_docs = []
    
    while retry_count < max_retries:
        try:
            # Fetch specific failed documents from the database
            data = fetch_scraped_data_by_ids(failed_ids)
            
            if not data:
                logger.error("No failed documents to retry.")
                return
            
            actions = []
            
            for doc_id, url, text in data:
                embedding = generate_openai_embedding(text)
                document = {
                    "_op_type": "index",
                    "_index": OPENSEARCH_INDEX,
                    "_id": str(doc_id),
                    "_source": {
                        "url": url,
                        "content": text,
                        "vector": embedding.tolist()
                    }
                }
                actions.append(document)

            # Perform bulk indexing for failed documents
            success, failed = bulk(client, actions)
            logger.info(f"Successfully re-indexed {success} documents.")
            
            if failed > 0:
                logger.error(f"Failed to index {failed} documents. Storing failed IDs.")
                # Track failed documents
                failed_docs= [error['index']['_id'] for error in failed if 'index' in error]
            else:
                # If no failures, break the loop
                logger.info("All documents successfully re-indexed.")
                return
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Error during bulk indexing: {e}")
        
        # Increment the retry count
        retry_count += 1
        logger.info(f"Retry attempt {retry_count}/{max_retries}")

        # If there are still failed documents, we retry
        if failed_docs:
            logger.info(f"Retrying failed documents: {failed_docs}")
            # Call the function recursively for retry
            store_failed_documents(failed_docs, max_retries)
        else:
            break  # Exit if no failed documents remain after retry
    if retry_count == max_retries and failed_docs:
        logger.error(f"Failed to re-index {len(failed_docs)} documents after {max_retries} retries.")
        print(failed_docs)


# Delete a single document by ID
def delDocId(index, id):
    client.delete(index=index, id=id)

# Delete entire index
def delIndexNamed(indexName):
    if client.indices.exists(index=indexName):
        client.indices.delete(index=indexName)
        print(f"Index '{indexName}' deleted.")
    else:
        print(f"Index '{indexName}' not found. Nothing to delete.")

def delIndex():
    delIndexNamed(OPENSEARCH_INDEX)

def makeIndex():
    makeIndexNamed(OPENSEARCH_INDEX)


def fetch_mock_data():
    return [
        (1, "https://example.com/1", "Hartnell College is a public community college in Salinas, California."),
        (2, "https://example.com/2", "The college offers associate degrees and certificates across a variety of fields."),
        (3, "https://example.com/3", "Students benefit from smaller class sizes and personalized attention."),
        (4, "https://example.com/4", "Hartnell has a strong transfer program to California State Universities."),
        (5, "https://example.com/5", "Online and hybrid classes are available to meet student needs.")
    ]

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    try:
        response = client.info()
        logger.info(f"Connected to OpenSearch: {response}")
    except Exception as e:
        logger.error(f"Error connecting to OpenSearch: {str(e)}")


    print("Removing Index")
    delIndexNamed(OPENSEARCH_INDEX)
    print("Making Index")
    makeIndexNamed(OPENSEARCH_INDEX)                # Create index if it doesn't exist
    print("Storing Embeddings")
    # bulk_store_embeddings_in_opensearch()           # Store all embeddings
    store_embeddings_in_opensearch()