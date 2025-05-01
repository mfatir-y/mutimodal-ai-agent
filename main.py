import os

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

llm = Ollama(model = 'mistral', request_timeout = 120)

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
    result = agent.query(prompt)
    next_result = output_pipeline.run(response=result)
    print(next_result)

