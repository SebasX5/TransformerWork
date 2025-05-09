import chromadb
import mariadb
import numpy as np
import openai
from pprint import pprint

# API Keys
COHERE_API_KEY = "ekac8v5OlcEJ1TGGOFC30hzY73zwSqcQPfnv6hJN"
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# ChromaDB Setup (replaces OpenSearch)
client = chromadb.HttpClient(host="172.31.30.137", port=9200)
collection = client.get_or_create_collection("hartnell")

# MariaDB Config
mariadb_config = {
    "host": "172.31.30.137",
    "user": "root",
    "password": "H*W7]nD-C(4:#EfsV?MA5G$bQ",
    "port": 3306,
    "database": "hartnell_scraped_data"
}

# Generate OpenAI embedding (must match Chroma embeddings)
def generate_query_embedding(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

# Search Chroma
def search_chroma(query_embedding, top_k=3):
    print(collection.count())
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return [int(doc_id) for doc_id in result["ids"][0]]

# Fetch results from MariaDB using document IDs
def fetch_results(doc_ids):
    if not doc_ids:
        print("⚠️ No document IDs returned from ChromaDB — skipping DB query.")
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

# Language detection and translation using OpenAI
def detect_language(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Detect the language of this text and respond only with the name of the language."},
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
            {"role": "system", "content": f"Translate this from English to {target_language}. Keep links unchanged."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

# Generate OpenAI answer using context
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

# Chat loop
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
        doc_ids = search_chroma(query_embedding)
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

# === Run Bot ===
if __name__ == "__main__":
    print("✅ ChromaDB is ready. Launching chatbot...")
    chatbot()
