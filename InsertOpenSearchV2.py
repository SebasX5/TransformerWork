import numpy as np
from opensearchpy import OpenSearch
import mariadb
import urllib3
from opensearchpy.connection import RequestsHttpConnection
import logging
from opensearchpy.helpers import bulk
from opensearchpy.exceptions import TransportError
import openai
import random

import time
import logging

openai.api_key =" "



# Disable SSL warnings (use cautiously in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# OpenAI Configuration
# ----------------------------
OPENAI_API_KEY = ""



# ----------------------------
# OpenSearch Configuration
# ----------------------------
OPENSEARCH_HOST = "172.31.30.137:9200"
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
    timeout=60,  # Set timeout to 30 seconds or longer
    retries=5,  # Increase retries
    max_retries=10,  # Allow multiple retries,
    max_connections=5
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
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "knn": True
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
        cursor.execute("SELECT id, url, content FROM big_scraped_data")
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except mariadb.Error as e:
        print(f"Error fetching data: {e}")
        return []
    finally:
        if conn:
            conn.close()


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
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

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
# def bulk_store_embeddings_in_opensearch():
#     # data = fetch_scraped_data_with_retry()
#     # data = fetch_scraped_data()
#     data=fetch_mock_data()
#     if not data:
#         logger.error("No data fetched from MariaDB.")
#         return

#     actions = []
#     failed_docs = []

#     # Prepare documents for bulk indexing
#     for doc_id, url, text in data:
#         embedding = generate_openai_embedding(text)
#         document = {
#             "_op_type": "index",
#             "_index": OPENSEARCH_INDEX,
#             "_id": str(doc_id),
#             "_source": {
#                 "url": url,
#                 "content": text,
#                 "vector": embedding.tolist()
#             }
#         }
#         actions.append(document)

#     # Perform bulk indexing
#     try:
#         success, details = bulk(client, actions)
#         logger.info(f"Successfully indexed {success} documents.")
        
#         errors = [d for d in details if d.get('index', {}).get('error')]
#         if errors:
#             logger.error(f"{len(errors)} items failed.")
#             # optionally print or collect failed _id's

#     except (ConnectionError, TimeoutError) as e:
#         logger.error(f"Error during bulk indexing: {e}")
    
#     # Retry indexing for failed documents
#     if failed_docs:
#         logger.info(f"Retrying failed documents: {failed_docs}")
#         store_failed_documents(failed_docs)



def bulk_store_embeddings_in_opensearch2():
    data = fetch_mock_data()  # or fetch_scraped_data_with_retry()

    if not data:
        logger.error("No data fetched from MariaDB.")
        return

    actions = []
    for doc_id, url, text in data:
        try:
            embedding = generate_openai_embedding(text)

            if embedding is None or len(embedding) != 1536:
                logger.warning(f"Skipping doc {doc_id}: Invalid embedding shape.")
                continue

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
            continue

    if not actions:
        logger.warning("No valid documents to index.")
        return

    try:
        success, errors = bulk(client, actions, raise_on_error=False)
        logger.info(f"✅ Indexed {success} documents successfully.")
        
        failed = [e for e in errors if e.get("index", {}).get("error")]
        if failed:
            logger.error(f"❌ {len(failed)} documents failed to index.")
            for f in failed[:5]:  # log only the first few for sanity
                logger.error(f)

    except TransportError as e:
        logger.error(f"🚨 OpenSearch transport error: {e}")
    except Exception as e:
        logger.error(f"🚨 Unexpected error during bulk insert: {e}")

# Logger setup
logger = logging.getLogger(__name__)


CHUNK_SIZE = 5
MAX_RETRIES = 3
RETRY_DELAY = 2  # base seconds, will double on each retry

def bulk_store_embeddings_in_opensearch():
    data = fetch_scraped_data()
  # or fetch_scraped_data_with_retry()

    if not data:
        logger.error("No data fetched from MariaDB.")
        return

    actions = []
    for doc_id, url, text in data:
        try:
            embedding = generate_openai_embedding(text)

            if embedding is None or len(embedding) != 1536:
                logger.warning(f"Skipping doc {doc_id}: Invalid embedding shape.")
                continue

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
            continue

        if len(actions) >= CHUNK_SIZE:
            if not try_bulk_insert(actions):
                logger.error("❌ Aborting further indexing due to repeated bulk insert failures.")
                return
            actions = []
            logger.info("Waiting for 5 seconds before indexing next batch...")
            time.sleep(2)

    # Final batch
    if actions:
        if not try_bulk_insert(actions):
            logger.error("❌ Aborting due to failure in final batch insert.")
            return


def try_bulk_insert(actions):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            success, errors = bulk(client, actions, raise_on_error=False)
            logger.info(f"✅ Indexed {success} documents successfully.")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * attempt
                logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("❌ Max retries reached. Bulk insert failed.")
                return False

# Retry failed documents by fetching them and re-indexing
def store_failed_documents(failed_ids, max_retries=1):
    retry_count = 0
    failed_docs = []

    if max_retries <= 0:
        logger.error("Max retries exhausted.")
        return
    
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
            store_failed_documents(failed_docs, max_retries=max_retries-1)
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
        (5, "https://example.com/5", "Online and hybrid classes are available to meet student needs."),
        (6, "https://example.com/6", "Financial aid and scholarships help students afford their education."),
        (7, "https://example.com/7", "The Alisal Campus focuses on agricultural and industrial technology programs."),
        (8, "https://example.com/8", "Hartnell’s STEM program supports students in science, technology, engineering, and math."),
        (9, "https://example.com/9", "Counseling services help students plan their academic and career goals."),
        (10, "https://example.com/10", "Hartnell's nursing program is well-regarded in the region."),
        (11, "https://example.com/11", "Workshops and tutoring are available at the Panther Learning Lab."),
        (12, "https://example.com/12", "Hartnell College Foundation provides community and donor support."),
        (13, "https://example.com/13", "The King City Education Center serves South Monterey County."),
        (14, "https://example.com/14", "Hartnell students participate in a wide range of clubs and activities."),
        (15, "https://example.com/15", "Student government gives learners a voice in campus decisions."),
        (16, "https://example.com/16", "Career Services offers resume help and job placement assistance."),
        (17, "https://example.com/17", "The Main Campus features modern facilities and student lounges."),
        (18, "https://example.com/18", "Hartnell has a vibrant performing arts program, including theater and music."),
        (19, "https://example.com/19", "Veterans Services support military-affiliated students."),
        (20, "https://example.com/20", "Library resources include research databases and one-on-one support."),
        (21, "https://example.com/21", "Hartnell offers ESL classes to support English language learners."),
        (22, "https://example.com/22", "The college emphasizes equity, inclusion, and student success."),
        (23, "https://example.com/23", "Programs in early childhood education prepare students for teaching careers."),
        (24, "https://example.com/24", "Hartnell’s athletics program includes soccer, baseball, and track."),
        (25, "https://example.com/25", "Many students take advantage of free bus passes for transportation."),
        (26, "https://example.com/26", "Students can access mental health counseling on campus."),
        (27, "https://example.com/27", "Technology support helps students with online learning tools."),
        (28, "https://example.com/28", "The Salinas Valley Promise covers tuition for first-time, full-time students."),
        (29, "https://example.com/29", "Hartnell collaborates with local high schools on dual enrollment."),
        (30, "https://example.com/30", "Academic calendars and class schedules are available online."),
        (31, "https://example.com/31", "Student health services offer medical care and wellness support."),
        (32, "https://example.com/32", "The library has a collection of digital textbooks for student use."),
        (33, "https://example.com/33", "Hartnell College's website provides detailed course descriptions."),
        (34, "https://example.com/34", "The Food Pantry is available to help students facing food insecurity."),
        (35, "https://example.com/35", "Career fairs and networking events are held throughout the year."),
        (36, "https://example.com/36", "The college provides a variety of student discounts through local businesses."),
        (37, "https://example.com/37", "Hartnell offers programs in digital media and graphic design."),
        (38, "https://example.com/38", "The Honors Program provides academic challenges and rewards for top students."),
        (39, "https://example.com/39", "The college hosts cultural and diversity events to enrich student life."),
        (40, "https://example.com/40", "Peer mentoring helps incoming students transition to college life."),
        (41, "https://example.com/41", "Student engagement opportunities include internships and volunteer work."),
        (42, "https://example.com/42", "The college offers study abroad programs in various countries."),
        (43, "https://example.com/43", "Hartnell College has a dedicated service for veterans and military families."),
        (44, "https://example.com/44", "The college offers a range of non-credit courses for personal development."),
        (45, "https://example.com/45", "The college provides a robust online course catalog for distance learning."),
        (46, "https://example.com/46", "Hartnell’s art program features exhibitions and student showcases."),
        (47, "https://example.com/47", "The college’s student housing options provide affordable living on-campus."),
        (48, "https://example.com/48", "Hartnell’s music department offers a variety of classes and performance groups."),
        (49, "https://example.com/49", "The student radio station provides hands-on media experience."),
        (50, "https://example.com/50", "The college’s Environmental Science program focuses on sustainability."),
        (51, "https://example.com/51", "The dental hygiene program is a popular career pathway for students."),
        (52, "https://example.com/52", "The college offers leadership development programs for students."),
        (53, "https://example.com/53", "Hartnell’s campus features a large sports complex for athletic events."),
        (54, "https://example.com/54", "The math department offers tutoring and supplemental instruction."),
        (55, "https://example.com/55", "The college has an active student newspaper and journalism program."),
        (56, "https://example.com/56", "Hartnell offers certificate programs for career-focused students."),
        (57, "https://example.com/57", "The Student Success Center provides academic coaching and advising."),
        (58, "https://example.com/58", "The campus has a variety of dining options for students."),
        (59, "https://example.com/59", "The community service program offers opportunities to give back to the local area."),
        (60, "https://example.com/60", "The Outdoor Education program offers adventure-based learning opportunities."),
        (61, "https://example.com/61", "Hartnell has a thriving student art gallery that showcases student work."),
        (62, "https://example.com/62", "The career services department hosts resume and interview workshops."),
        (63, "https://example.com/63", "The college offers a variety of professional certifications in healthcare fields."),
        (64, "https://example.com/64", "Hartnell has a popular culinary arts program with hands-on experience."),
        (65, "https://example.com/65", "The campus is committed to sustainability through energy-efficient initiatives."),
        (66, "https://example.com/66", "The college provides access to a wide range of academic scholarships."),
        (67, "https://example.com/67", "Hartnell offers adult education programs for career and personal development."),
        (68, "https://example.com/68", "The college has a vibrant campus life with student leadership opportunities."),
        (69, "https://example.com/69", "Hartnell offers a strong foundation for students interested in teaching careers."),
        (70, "https://example.com/70", "The college's theater program is known for its professional productions."),
        (71, "https://example.com/71", "Hartnell provides a full-service fitness center for student health and wellness."),
        (72, "https://example.com/72", "The college’s campus includes a beautiful garden for student relaxation."),
        (73, "https://example.com/73", "The Student Success Center offers a wealth of resources for academic achievement."),
        (74, "https://example.com/74", "Hartnell offers a variety of online courses for working students."),
        (75, "https://example.com/75", "The faculty at Hartnell is dedicated to student success and personal growth."),
        (76, "https://example.com/76", "Hartnell offers a variety of majors for students interested in the sciences."),
        (77, "https://example.com/77", "The college provides opportunities for students to participate in research projects."),
        (78, "https://example.com/78", "Hartnell has an active student body that engages in leadership and volunteerism."),
        (79, "https://example.com/79", "The student-run food pantry ensures that no one goes hungry on campus."),
        (80, "https://example.com/80", "The college provides access to free tutoring services in various subjects."),
        (81, "https://example.com/81", "The Hartnell College Foundation hosts fundraising events to support student scholarships."),
        (82, "https://example.com/82", "The college provides career counseling to help students with job placements."),
        (83, "https://example.com/83", "Hartnell's business program offers courses in entrepreneurship and management."),
        (84, "https://example.com/84", "The college’s history program offers students an in-depth understanding of the past."),
        (85, "https://example.com/85", "Hartnell has partnerships with local industries for internship opportunities."),
        (86, "https://example.com/86", "The engineering program at Hartnell offers both theoretical and practical training."),
        (87, "https://example.com/87", "Hartnell’s computer science program prepares students for high-demand tech careers."),
        (88, "https://example.com/88", "The college’s political science department offers courses in government and law."),
        (89, "https://example.com/89", "The college’s philosophy program helps students develop critical thinking skills."),
        (90, "https://example.com/90", "The campus features a student-run cafe that offers coffee and snacks."),
        (91, "https://example.com/91", "Hartnell College offers a variety of clubs and organizations for student engagement."),
        (92, "https://example.com/92", "The campus offers a peaceful environment with scenic views of the Salinas Valley."),
        (93, "https://example.com/93", "Hartnell offers extensive support for students with disabilities."),
        (94, "https://example.com/94", "The college offers workshops to help students develop effective study habits."),
        (95, "https://example.com/95", "Hartnell’s science department offers modern laboratories and equipment for research."),
        (96, "https://example.com/96", "The college offers evening classes for students with busy schedules."),
        (97, "https://example.com/97", "The campus is conveniently located near downtown Salinas for easy access."),
        (98, "https://example.com/98", "Hartnell's library has a vast collection of physical and digital resources."),
        (99, "https://example.com/99", "The college offers a wide variety of volunteer opportunities for students."),
        (100, "https://example.com/100", "Hartnell offers academic counseling to help students with course selection.")
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