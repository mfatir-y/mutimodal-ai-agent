import json
import os
import streamlit as st

from llama_index.core.agent import ReActAgent
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from pydantic import BaseModel
from dotenv import load_dotenv

from code_reader import code_reader
from prompts import context, code_parser_template

load_dotenv()   # load the .env file for api key

llm = Ollama(model = 'mistral', request_timeout = 300)

parser = LlamaParse(api_key = os.getenv("LLAMACLOUD_API_KEY"), result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model= embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

tools = [QueryEngineTool(query_engine = query_engine,
                         metadata = ToolMetadata(name = "api_documentation",
                                                 description = "this gives documentation about code for an API."
                                                               " Use this for reading docs for the API")),
         code_reader]

code_llm = Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_template = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain = [json_prompt_template, llm])

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = str(next_result).replace("assistant:", "").strip()
            cleaned_json = json.loads(cleaned_json)
            break
        except Exception as e:
            retries += 1
            print(f'Retry #{retries} | An error occurred: {e}')

    if retries >= 3:
        print("Unable to process your input. Try again.")
        continue

    print(str.upper("Code Generated:"))
    print(cleaned_json['code'])
    print(str.upper("Description:"))
    print(cleaned_json['description'])
    print(str.upper("Filename:"))
    print(cleaned_json['filename'])

    try:
        with open(os.path.join("output", cleaned_json['filename']), "w") as file:
            file.write(cleaned_json['code'])
        print(f'Code written to: {cleaned_json["filename"]}')
    except Exception as e:
        print(f'Error writing to file: {e}')
