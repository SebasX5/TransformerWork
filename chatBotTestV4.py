import cohere
import mariadb
import numpy as np
import openai
from opensearchpy import OpenSearch
import os
from pprint import pprint

# API Keys
COHERE_API_KEY = "ekac8v5OlcEJ1TGGOFC30hzY73zwSqcQPfnv6hJN"
OPENAI_API_KEY = ""
OPENSEARCH_INDEX = "mock_knn_vector_index_2"

# MariaDB Config
mariadb_config = {
    "host": "172.31.30.137",
    "user": "root",
    "password": "H*W7]nD-C(4:#EfsV?MA5G$bQ",
    "port": 3306,
    "database": "hartnell_scraped_data"
}

# OpenSearch Config
client = OpenSearch(
    hosts=["172.31.30.137:9200"],
    http_auth=("admin", "H@RTn311_ROCKS"),
    use_ssl=False,
    verify_certs=False,
    timeout=60,
)

# Init clients
co = cohere.Client(COHERE_API_KEY)
# openai.api_key = OPENAI_API_KEY

# Generate embedding (must match what's in OpenSearch)
def generate_query_embedding(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

# Search OpenSearch with k-NN
def search_opensearch(query_embedding, top_k=3):
    body = {
        "size": top_k,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_embedding.tolist(),
                    "k": top_k
                }
            }
        }
    }
    response = client.search(index=OPENSEARCH_INDEX, body=body)
    hits = response["hits"]["hits"]

    print(f"\n🔍 Raw OpenSearch Hits ({len(hits)}):")
    for hit in hits:
        print(f"- _id: {hit['_id']}, _score: {hit.get('_score')}")

    return [int(hit["_id"]) for hit in hits]



def check_index_mapping(index_name):
    try:
        mapping = client.indices.get_mapping(index=index_name)
        pprint(mapping)
    except Exception as e:
        print(f"Error retrieving mapping for index '{index_name}': {e}")


# Fetch results from MariaDB using document IDs
def fetch_results(doc_ids):
    if not doc_ids:
        print("⚠️ No document IDs returned from OpenSearch — skipping DB query.")
        return []

    try:
        conn = mariadb.connect(**mariadb_config)
        cursor = conn.cursor()
        format_strings = ','.join(['%s'] * len(doc_ids))
        cursor.execute(f"SELECT id, url, content FROM big_scraped_data WHERE id IN ({format_strings})", tuple(doc_ids))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [{"url": row[1], "content": row[2]} for row in rows]
    except mariadb.Error as e:
        print(f"Error fetching results: {e}")
        return []

# Translation and detection logic (unchanged)
def detect_language(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Detect the language of this text and respond only with the name of the language (like 'English', 'Spanish', 'French')."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

def translate_to_english(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Translate the following into English."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

def translate_to_original_language(text, target_language):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Translate the following from English to {target_language}. Keep the links or sources as is."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

def ask_openai(user_question, context_docs):
    context_text = "\n\n".join([doc["content"] for doc in context_docs])
    prompt = f"""You're a helpful chatbot for Hartnell College. Answer the question below using a friendly, casual tone — like you're texting a student. 
                 Keep it short and easy to follow. If there are steps, list them in a quick, clear format. 
                 Include any useful links from the sources.

                 Question: {user_question}

                 Relevant information:
                 {context_text}
              """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and friendly assistant for Hartnell College's website chatbot. Use only the sources provided to answer questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message["content"]

# Main chatbot loop
def chatbot():
    print("Hi there! Ask me anything about Hartnell College — I’ve got your back!\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print(" Goodbye! Take care!")
            break

        user_language = detect_language(query)
        translated_query = translate_to_english(query) if user_language.lower() != "english" else query

        query_embedding = generate_query_embedding(translated_query)
        doc_ids = search_opensearch(query_embedding)
        context_docs = fetch_results(doc_ids)
        answer = ask_openai(translated_query, context_docs)

        if user_language.lower() != "english":
            answer = translate_to_original_language(answer, user_language)

        print("\n💬 Answer:")
        print(answer)

        print("\n🔗 Sources:")
        for doc in context_docs:
            print(f"- {doc['url']}")

        print("\n💭 Got another question? Just ask or type 'exit' to quit!\n")

def check_cluster_settings():
    try:
        settings = client.cluster.get_settings()
        pprint(settings)
    except Exception as e:
        print(f"Error fetching cluster settings: {e}")

def enable_knn_plugin_cluster():
    try:
        response = client.cluster.put_settings(body={
            "persistent": {
                "knn.plugin.enabled": True
            }
        })
        print("✅ k-NN plugin enabled at cluster level:", response)
    except Exception as e:
        print(f"Error updating cluster settings: {e}")


# === Run Bot ===
if __name__ == "__main__":
    count = client.count(index=OPENSEARCH_INDEX)
    print("📦 Total documents in index:", count['count'])
    # chatbot()
    # enable_knn_plugin_cluster()
