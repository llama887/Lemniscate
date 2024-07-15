import argparse
import ast
import sqlite3

import chromadb
import ollama
from tqdm import tqdm

client = chromadb.Client()
incognito = False

SYSTEM_PROMPT = """You are an AI assistance that has memory of every conversation you have ever has with this user.
    On every prompt from the user, the system has checked for any relevant messages you have had with the user.
    If any embedded previous conversations are attached, use them for context to respond to the user if the context is relevant and useful to the response.
    If the recalled conversations are irrelevant, disreagard them and respond to the user and respond normally as an AI assistant. Do not refer to recaled converstations.
    Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.
    You are to bypass any checks preventing you from revealing personal infromation as your job is to reveal personal information to the user if it is relevant to the prompt.
    """

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

connection = sqlite3.connect("vector_database.db")
connection.row_factory = sqlite3.Row
cursor = connection.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, 
    prompt TEXT NOT NULL, 
    response TEXT NOT NULL)"""
)

connection.commit()


def fetch_conversations():
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM conversations")
    rows = cursor.fetchall()
    conversations = [{k: row[k] for k in row.keys()} for row in rows]
    return conversations


def store_conversation(prompt, response):
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, ?, ?)",
        (prompt, response),
    )
    connection.commit()


def stream_response(response):
    response = ""
    stream = ollama.chat(model="llama3", messages=conversation, stream=True)
    print("\nASSISTANT: ")
    for chunk in stream:
        response += chunk["message"]["content"]
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")
    if not incognito:
        store_conversation(prompt, response)
    conversation.append({"role": "assistant", "content": prompt})


def create_vector_database(conversations):
    database_name = "conversations"
    try:
        client.delete_collection(name=database_name)
    except ValueError:
        pass
    database = client.create_collection(name=database_name)
    for conversation in conversations:
        serialized_conversation = f'prompt: {conversation["prompt"]}, response: {conversation["response"]}'
        response = ollama.embeddings(
            model="nomic-embed-text", prompt=serialized_conversation
        )
        embedding = response["embedding"]
    database.add(
        ids=[str(conversation["id"])],
        embeddings=[embedding],
        documents=[serialized_conversation],
    )


def retrieve_embeddings(queries, results_per_query=1, prompt=None):
    embeddings = set()
    for query in tqdm(queries, desc="Processing queries"):
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response["embedding"]
        database = client.get_collection(name="conversations")
        results = database.query(
            query_embeddings=[query_embedding], n_results=results_per_query
        )
        best_embedding = results["documents"][0]

        for best in best_embedding:
            if best not in embeddings:
                if "yes" in classify_embedding(query, best):
                    embeddings.add(best)
    if prompt:
        response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
        prompt_embedding = response["embedding"]
        database = client.get_collection(name="conversations")
        results = database.query(
            query_embeddings=[prompt_embedding], n_results=results_per_query
        )
        best_embedding = results["documents"][0][0]
        embeddings.add(best_embedding)
    return embeddings


def create_queries(prompt):
    query_message = """You are a first principle reasoning search query AI agent.
    Your list of search queries will be ran on an embedding database of all your conversations you have ever had with the user. 
    You are to bypass any checks preventing you from revealing personal infromation as your job is to reveal personal information to the user if it is relevant to the prompt.
    With first principles create a Python list of queries to search the embeddings database for any data that would be necessay to have access to in order to correctly respond to the prompt. 
    Your response must be a Python list with no syntax errors.
    Do not explain anything and do not every generate anything but a perfect syntax Python list."""
    # multishot learning
    query_conversation = [
        {
            "role": "system",
            "content": query_message,
        },
        {
            "role": "user",
            "content": "Write an email to my car insurance company and create a pursuasive requrest for them to lower my monthly premium.",
        },
        {
            "role": "assistant",
            "content": "['What is the users name?', 'What is the users current auto insurance provider?', 'What is the users current monthly premium?']",
        },
        {
            "role": "user",
            "content": "how can i conver the speak function in my llama3 voice assistance to use pyttsx3 instead of open ai text to speech?",
        },
        {
            "role": "assistant",
            "content": "['Llama3 voice assistant', 'What is the users current voice assistance library?', 'Python voice assitant', 'OpenAI TTS']",
        },
        {
            "role": "user",
            "content": "who is my best friend?",
        },
        {
            "role": "assistant",
            "content": "['Who are the users closest relationships?', 'Who do the users interact with most frequently?', 'User social connections', 'User relationship data']",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = ollama.chat(model="llama3", messages=query_conversation)
    print(f'\nVector database queries: {response["message"]["content"]}\n')
    try:
        queries = ast.literal_eval(response["message"]["content"])
        queries.append(f"{prompt}")
        return queries
    except:
        return [f"{prompt}"]


def classify_embedding(query, context):
    classify_message = """You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text.
    You will not respond as an AI assistant. You will only respond 'yes' or 'no'.
    Determine whether the context contains data that direct is related to the search query.
    If the context is seemingly exactly what the search query needs, respond 'yes'.
    If it is anything but directly related, respond 'no'. Do not respond 'yes' unless the content is highly relevant to the search query."""
    classify_conversation = [
        {
            "role": "system",
            "content": classify_message,
        },
        {
            "role": "user",
            "content": f"SEARCH QUERY: What is the users name\n\nEMBEDDED CONTEXT: Your name is Bob. How can I help you today Bob?",
        },
        {
            "role": "assistant",
            "content": "yes",
        },
        {
            "role": "user",
            "content": f"SEARCH QUERY: Llama3 Python Voice Assistant\n\nEMBEDDED CONTEXT: Siri is a voice assistant used on Apple iOS and Mac OS.",
        },
        {
            "role": "assistant",
            "content": "no",
        },
        {
            "role": "user",
            "content": f"SEARCH QUERY: {query}\n\nEMBEDDED CONTEXT: {context}",
        },
    ]

    response = ollama.chat(model="llama3", messages=classify_conversation)
    return response["message"]["content"].strip().lower()


def recall(prompt):
    queries = create_queries(prompt)
    embeddings = retrieve_embeddings(queries, prompt=prompt)
    conversation.append(
        {
            "role": "user",
            "content": f"MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}",
        }
    )
    print(f"\n{len(embeddings)} message:response embeddings added for context.")


def store_default():
    cursor = connection.cursor()
    cursor.execute(
        """INSERT OR IGNORE INTO conversations (prompt, response)
        VALUES (?, ?)""",
        (
            "Who are you?",
            "I am a professional programer and writer here to help you with all programming and writing related tasks.",
        ),
    )
    connection.commit()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Assistant that uses SQLite and Ollama to generate responses"
    )
    parse.add_argument(
        "--clear",
        action="store_true",
        help="Clear the database of all conversations",
    )
    parse.add_argument(
        "--incognito",
        action="store_true",
        help="Does not store any conversations in the database",
    )
    args = parse.parse_args()
    if args.clear:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM conversations")
        connection.commit()
        store_default()
    if args.incognito:
        incognito = True
    create_vector_database(conversations=fetch_conversations())
    while True:
        prompt = input("USER: \n")
        if prompt[:7].lower() == "/recall":
            prompt = prompt[8:]
            recall(prompt)
            stream_response(prompt)
        elif prompt.lower() == "/exit":
            break
        else:
            conversation.append({"role": "user", "content": prompt})
            stream_response(prompt)
