from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "RAG FastAPI"
    APP_VERSION: str = "0.1.0"
    HF_TOKEN: str | None = None
    CHROMA_DIR: str = "./data/chroma"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDINGS_BACKEND: str = "local"

    LLM_BACKEND: str = "local"  
    LLM_MODEL: str = "google/flan-t5-small"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
