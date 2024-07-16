# Lemniscate
You must download Ollama on you machine. Check the install and pull llama3 by running `ollama run llama3`. Pull the text embedding model by running `ollama pull nomic-embed-text`.
Windows users can download dependencies via conda using the `environment.yaml` file. Alternatively, download from `requirements.txt`.

# Usage
Run `assistant.py` to start the chat bot. Adding the `--clear` flag will clear the assitants memory and `--incognito` will prevent the agent from storing the current sessions into memory.

For speed sake, the assitant will not try to recall from memory but will store all user inputs and assistant responses into the response (even in incognito mode).

To recall from memory, use the "/recall" flag prior to your prompt.


