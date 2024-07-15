import chromadb
import ollama

client = chromadb.Client()
message_history = [
    {
        "id": 1,
        "prompt": "What is my name?",
        "response": "Your name is Franklin Yiu",
    }
]
conversation = []


def stream_response(response):
    conversation.append({"role": "user", "content": prompt})
    response = ""
    stream = ollama.chat(model="phi3:mini", messages=conversation, stream=True)
    print("\nASSISTANT: ")
    for chunk in stream:
        response += chunk["message"]["content"]
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")


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


def retrieve_embeddings(prompt):
    response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
    prompt_embedding = response["embedding"]
    database = client.get_collection(name="conversations")
    results = database.query(query_embeddings=[prompt_embedding], n_results=1)
    best_embedding = results["documents"][0][0]
    return best_embedding


create_vector_database(conversations=message_history)
while True:
    prompt = input("USER: \n")
    context = retrieve_embeddings(prompt)
    prompt = f"USER PROMPT: {prompt} \n CONTEXT FROM EMBEDDINGS: {context}"
    stream_response(prompt)
