from openai import OpenAI
import numpy as np
from opensearchpy import OpenSearch
import mariadb
import urllib3
from opensearchpy.connection import RequestsHttpConnection
import logging
# from elasticsearch.helpers import bulk
from opensearchpy.helpers import bulk
import time
from itertools import islice


# Disable SSL warnings (use cautiously in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
USE_MOCK_DATA = True
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
    use_ssl=False,
    verify_certs=False,  # Set to True with proper certs in production
    connection_class=RequestsHttpConnection,
    timeout=30,  # Set timeout to 30 seconds or longer
    retries=10,  # Increase retries
    max_retries=10  # Allow multiple retries
)

# Set up logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("opensearch_errors.log"),  # Logs go to this file
        logging.StreamHandler()  # Also logs to console
    ]
)
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
                "number_of_replicas": 0
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
    data = fetch_mock_data() if USE_MOCK_DATA else fetch_scraped_data()
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
        time.sleep(1)
    print(f"Number of embeddings stored: {len(data)}")


# Store documents in OpenSearch with retry and error handling
def bulk_store_embeddings_in_opensearch(batch_size=25, throttle_seconds=5):
    data = fetch_mock_data() if USE_MOCK_DATA else fetch_scraped_data()
    if not data:
        logger.error("No data fetched from MariaDB.")
        return

    total_index=0
    failed_docs = []

    for batch_num, batch in enumerate(batch_data(data, batch_size), start=1):

        actions = []
        # Prepare documents for bulk indexinga
        for doc_id, url, text in batch:
            try:
                # Attempt to generate the embedding
                try:
                    embedding = generate_openai_embedding(text)
                except Exception as e:
                    logger.error(f"Embedding generation failed for doc {doc_id} with error: {e}")
                    failed_docs.append(str(doc_id))
                    continue  # Skip this document and move to the next
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
            except Exception as e:
                logger.error(f"Unexpected error while processing doc {doc_id}: {e}")
                failed_docs.append(str(doc_id))
                continue

        # Perform bulk indexing
        try:
            success, failed = bulk(client, actions)
            total_index += success
            logger.info(f"Batch {batch_num}: Indexed {success} docs, Failed: {len(failed)}")
            # logger.info(f"Successfully indexed {success} documents.")
            
            if len(failed) > 0:
                logger.error(f"Failed to index {failed} documents. Storing failed IDs.")
                # Track failed documents
                batch_fail_ids= [error['index']['_id'] for error in failed if 'index' in error]
                failed_docs.extend(batch_fail_ids)

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Error during bulk indexing: {e}")
        time.sleep(throttle_seconds)

        logger.info(f"Total documents successfully indexed: {total_index}")

        
        # Retry indexing for failed documents
        # if failed_docs:
        #     logger.info(f"Retrying failed documents: {failed_docs}")
        #     store_failed_documents(failed_docs)

# Helper: Chunking generator
def batch_data(data, batch_size):
    it = iter(data)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk


# Retry failed documents by fetching them and re-indexing
"""
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
"""

def store_failed_documents(failed_ids, max_retries=3, throttle_seconds=1):
    retry_count = 0
    remaining_failed_ids = failed_ids

    while retry_count < max_retries and remaining_failed_ids:
        logger.info(f"Retry attempt {retry_count + 1}/{max_retries} for {len(remaining_failed_ids)} documents.")

        try:
            data = fetch_scraped_data_by_ids(remaining_failed_ids)
            if not data:
                logger.warning("No documents found for given failed IDs.")
                return

            actions = []
            embedding_failed_ids = []

            for doc_id, url, text in data:
                try:
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
                except Exception as e:
                    logger.error(f"Embedding generation failed for doc {doc_id}: {e}")
                    embedding_failed_ids.append(doc_id)

            success, failed = bulk(client, actions)
            logger.info(f"Re-indexed {success} documents successfully.")

            failed_bulk_ids = [error['index']['_id'] for error in failed if 'index' in error]
            remaining_failed_ids = embedding_failed_ids + failed_bulk_ids

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error during retry: {e}")

        retry_count += 1
        if remaining_failed_ids:
            logger.warning(f"{len(remaining_failed_ids)} documents still failed. Retrying in {throttle_seconds}s...")
            time.sleep(throttle_seconds)

    # if remaining_failed_ids:
    #     logger.error(f"Failed to re-index {len(remaining_failed_ids)} documents after {max_retries} retries.")
    #     print("Final failed IDs:", remaining_failed_ids)


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
    return[
        (1, "https://example.com/1", "Hartnell College is a public community college in Salinas, California."),
        (2, "https://example.com/2", "The college offers associate degrees and certificates across a variety of fields."),
        (3, "https://example.com/3", "Students benefit from smaller class sizes and personalized attention."),
        (4, "https://example.com/4", "Hartnell has a strong transfer program to California State Universities."),
        (5, "https://example.com/5", "Online and hybrid classes are available to meet student needs."),
        (6, "https://example.com/6", "Hartnell College has multiple campuses to serve Monterey County."),
        (7, "https://example.com/7", "Financial aid and scholarships help many students afford tuition."),
        (8, "https://example.com/8", "The college provides career counseling and job placement assistance."),
        (9, "https://example.com/9", "STEM programs at Hartnell are supported by dedicated labs and faculty."),
        (10, "https://example.com/10", "Hartnell participates in community outreach and adult education."),
        (11, "https://example.com/11", "Students have access to tutoring, mentoring, and academic support."),
        (12, "https://example.com/12", "The college supports student clubs, athletics, and leadership programs."),
        (13, "https://example.com/13", "Hartnell's nursing program is well-regarded in the region."),
        (14, "https://example.com/14", "Many classes are designed to accommodate working adults."),
        (15, "https://example.com/15", "The Salinas campus includes a library, bookstore, and student center."),
        (16, "https://example.com/16", "Hartnell promotes diversity, equity, and inclusion across campus."),
        (17, "https://example.com/17", "Programs in agriculture and business align with local industries."),
        (18, "https://example.com/18", "Faculty at Hartnell bring both academic and industry experience."),
        (19, "https://example.com/19", "Workforce development initiatives support community growth."),
        (20, "https://example.com/20", "Students can take advantage of transfer agreements with UC campuses.")
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
    bulk_store_embeddings_in_opensearch()           # Store all embeddings
    # store_embeddings_in_opensearch()