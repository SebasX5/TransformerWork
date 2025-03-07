import cohere
import numpy as np

COHERE_API_KEY = "ekac8v5OlcEJ1TGGOFC30hzY73zwSqcQPfnv6hJN"
co = cohere.Client(COHERE_API_KEY)


documents = [
    """Hartnell College offers a variety of academic programs, including transfer degrees, technical education, and vocational training. 
       The college is committed to providing quality education to students in the Salinas Valley region.""",
    """Financial aid is available for eligible students through federal and state programs. The FAFSA application determines 
       qualification for grants, loans, and work-study opportunities.""",
    """The library provides students access to research databases, books, and online resources. Library hours vary during holidays 
       and summer sessions, and students can reserve study rooms through the online portal.""",
]

# Generate embeddings
response = co.embed(
    texts=documents, 
    model="embed-english-v3.0", 
    input_type="search_document"  # Specify input type
)

vectors = np.array(response.embeddings)

# Lets show the dimensions of the vector for debug purposes
print(f"Generated {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions each.")

# Example vector Print first vector
print("\nFirst vector embedding:\n", vectors[0])


# TODO: Once MariaDB has information, we can test this out
# This next part is how we hypothetically connect into the AWS database
import mysql.connector

# Database creds
DB_HOST = ""
DB_USER = ""
DB_PASSWORD = ""
DB_NAME = ""

# Connect to database
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)

cursor = conn.cursor()

# Example query"
query = "SELECT id, title, content, url FROM website;"
cursor.execute(query)

# Fetch all rows
rows = cursor.fetchall()

# Print data
for row in rows:
    print(f"ID: {row[0]}, Title: {row[1]}, URL: {row[3]}\nContent: {row[2]}\n")

cursor.close()
conn.close()

