from llama_index.llms.ollama import Ollama

llm = Ollama(model = 'mistral', request_timeout = 10)

result = llm.complete("Halo")
print(result)