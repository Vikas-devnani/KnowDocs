from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = "llama-3.3-70b-versatile"
    top_k_results: int = 4

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()