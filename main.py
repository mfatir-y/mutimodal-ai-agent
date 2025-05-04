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


# Function to initialize the AI components
@st.cache_resource
def initialize_ai_components():
    load_dotenv()   # load the .env file for api key

    llm = Ollama(model = 'mistral', request_timeout = 300)
    code_llm = Ollama(model="codellama")

    parser = LlamaParse(api_key = os.getenv("LLAMACLOUD_API_KEY"), result_type="markdown")
    documents = SimpleDirectoryReader("./data", file_extractor={".pdf": parser}).load_data()
    embed_model = resolve_embed_model("local:BAAI/bge-m3")
    vector_index = VectorStoreIndex.from_documents(documents, embed_model= embed_model)
    query_engine = vector_index.as_query_engine(llm=llm)

    tools = [QueryEngineTool(query_engine = query_engine,
                             metadata = ToolMetadata(name = "api_documentation",
                                                     description = "this gives documentation about code for an API."
                                                                   " Use this for reading docs for the API.")),
             code_reader]

    agent = ReActAgent.from_tools(tools,
                                  llm=code_llm,
                                  verbose=True,
                                  context=context)

    class CodeOutput(BaseModel):
        code: str
        description: str
        filename: str

    parser = PydanticOutputParser(CodeOutput)
    json_prompt_template = PromptTemplate(parser.format(code_parser_template))
    output_pipeline = QueryPipeline(chain = [json_prompt_template, llm])
    return agent, output_pipeline

