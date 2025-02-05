from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    documents_path: str
    collection_name: str
    persist_directory: str
    reranking: bool
    top_k: int
    embedding_model: str
    llm: str

config = Settings()
