from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(os.getcwd(), '.env'), env_file_encoding='utf-8')

    documents_path: str
    collection_name: str
    persist_directory: str
    reranking: bool
    top_k: int
    embedding_model: str
    llm: str

config = Settings()
