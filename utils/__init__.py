from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(os.getcwd(), '.env'), env_file_encoding='utf-8')

    documents_path: str
    collection_name: str
    persist_directory: str
    reranking: bool
    hallucinations: bool
    top_k: int
    embedding_model: str
    llm: str

config = Settings()

from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
llm = ChatOllama(model=config.llm, temperature=0)
memory = MemorySaver()