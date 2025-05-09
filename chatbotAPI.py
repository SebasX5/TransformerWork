from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import chromadb
import mariadb
import numpy as np

# Final API 


# === API Keys ===
openai.api_key = ""

# === ChromaDB Setup ===
chroma = chromadb.HttpClient(host="172.31.30.137", port=9200)
collection = chroma.get_or_create_collection("hartnell")

# === MariaDB Setup ===
mariadb_config = {
    "host": "172.31.30.137",
    "user": "root",
    "password": "H*W7]nD-C(4:#EfsV?MA5G$bQ",
    "port": 3306,
    "database": "hartnell_scraped_data"
}

# === FastAPI App ===
app = FastAPI()

class Question(BaseModel):
    query: str

def generate_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def detect_language(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Detect the language of this text and return just the name of the language."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

def translate_to_english(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Translate the following to English."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

def translate_to_original_language(text: str, target_language: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Translate the following into {target_language}. Keep any links unchanged."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

def fetch_docs_by_ids(ids: List[str]):
    try:
        conn = mariadb.connect(**mariadb_config)
        cursor = conn.cursor()
        format_strings = ','.join(['%s'] * len(ids))
        cursor.execute(f"SELECT id, url, content FROM big_scraped_data WHERE id IN ({format_strings})", tuple(ids))
        rows = cursor.fetchall()
        return [{"url": row[1], "content": row[2]} for row in rows]
    except mariadb.Error as e:
        print("❌ Error fetching data from MariaDB:", e)
        return []
    finally:
        if conn:
            conn.close()

def ask_openai(question: str, context: List[dict]) -> str:
    context_text = "\n\n".join([doc["content"] for doc in context])
    prompt = f"""You're a helpful chatbot for Hartnell College. Answer the question below using a friendly, casual tone — like you're texting a student. 
                 Keep it short and clear. Include any links from the sources if they’re relevant.

                 Question: {question}

                 Relevant info:
                 {context_text}"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're Hartnell College's helpful and friendly website chatbot."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message["content"]

@app.post("/ask")
def ask_question(data: Question):
    try:
        original_language = detect_language(data.query)
        query_in_english = (
            translate_to_english(data.query)
            if original_language.lower() != "english" else data.query
        )

        query_embedding = generate_embedding(query_in_english)
        result = collection.query(query_embeddings=[query_embedding], n_results=3)

        if not result["ids"] or not result["ids"][0]:
            return {"answer": "Sorry, I couldn't find anything relevant in the knowledge base.", "sources": []}

        doc_ids = result["ids"][0]
        docs = fetch_docs_by_ids(doc_ids)
        answer_in_english = ask_openai(query_in_english, docs)

        final_answer = (
            translate_to_original_language(answer_in_english, original_language)
            if original_language.lower() != "english" else answer_in_english
        )

        sources = [doc["url"] for doc in docs]
        return {"answer": final_answer, "sources": sources}

    except Exception as e:
        return {"error": f"Something went wrong: {str(e)}"}
    
@app.get("/")
def root():
    return{"message": "Hartnell Chatbot API is running now..."}


#To run the API locally
# 1. pip install fastapi uvicorn
#2. uvicorn chat_api:app --reload --host 127.0.0.1 --port 8080


#Future update: add Gunicorn to use more workers

