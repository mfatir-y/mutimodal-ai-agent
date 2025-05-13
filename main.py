import os
import streamlit as st
import json
from typing import Dict, Any, Optional, Union, List

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
from model_evaluator import ModelEvaluator

# Global LLM instances
_chat_llm = None
_code_llm = None

def get_llm(model_name: str = "mistral") -> Ollama:
    """Get or create an LLM instance."""
    global _chat_llm, _code_llm
    
    if model_name == "codellama":
        if _code_llm is None:
            _code_llm = Ollama(model=model_name, request_timeout=300)
        return _code_llm
    else:
        if _chat_llm is None:
            _chat_llm = Ollama(model=model_name, request_timeout=300)
        return _chat_llm

def query_llm(prompt: str, 
              model: str = "mistral", 
              response_format: Optional[str] = None) -> Union[str, Dict[str, Any], List[str]]:
    """
    Centralized function to query the LLM.    
    Args: prompt: The prompt to send to the LLM
          model: The model to use (default: "mistral")
          response_format: Optional format specification ("json" or "json_array")
    Returns: Union[str, Dict, List]: The LLM's response in the specified format
    """
    llm = get_llm(model)
    
    try:
        response = llm.complete(prompt)
        result = response.text
        
        if response_format == "json":
            return json.loads(result)
        elif response_format == "json_array":
            return json.loads(result)
        return result
    except Exception as e:
        print(f"Error querying LLM: {e}")
        if response_format == "json":
            return {"error": str(e)}
        elif response_format == "json_array":
            return [f"Error: {str(e)}"]
        return f"Error: {str(e)}"

# Function to initialize the AI components
@st.cache_resource
def initialize_ai_components(chat_model: str = "mistral", code_model: str = "codellama"):
    load_dotenv()   # load the .env file for api key

    llm = Ollama(model = chat_model, request_timeout = 300)
    code_llm = Ollama(model = code_model, request_timeout = 300)

    parser = LlamaParse(api_key = os.getenv("LLAMACLOUD_API_KEY"), result_type="markdown")
    documents = SimpleDirectoryReader("./data", file_extractor={".pdf": parser}).load_data()
    embed_model = resolve_embed_model("local:BAAI/bge-m3")
    vector_index = VectorStoreIndex.from_documents(documents, embed_model= embed_model)
    query_engine = vector_index.as_query_engine(llm=llm)

    tools = [QueryEngineTool(query_engine = query_engine,
                             metadata = ToolMetadata(name = "documentation_reader",
                                                     description= "This gives documentation about uploaded code and/or "
                                                                  "reference files. Use this for reading documentation "
                                                                  "or analyzing uploaded files.")),
             code_reader]

    agent = ReActAgent.from_tools(tools,
                                  llm=code_llm,
                                  verbose=False,
                                  context=context)

    class CodeOutput(BaseModel):
        code: str
        description: str
        filename: str

    parser = PydanticOutputParser(CodeOutput)
    json_prompt_template = PromptTemplate(parser.format(code_parser_template))
    output_pipeline = QueryPipeline(chain = [json_prompt_template, llm])

    model_evaluator = ModelEvaluator()

    return agent, output_pipeline, model_evaluator

