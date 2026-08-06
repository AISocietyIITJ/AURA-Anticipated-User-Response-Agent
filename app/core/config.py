from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Neo4j AuraDB ---
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    # --- LLM provider — any OpenAI-compatible /v1/chat/completions server ---
    # Used for: main chat, intent classification, sentiment analysis.
    # Switch providers by editing these three values in .env — code does not change.
    llm_base_url: str
    llm_model: str
    llm_api_key: str

    # --- Speech-to-Text (NVIDIA NIM cloud API) ---
    # Get key at: https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2
    stt_api_key: str

    dataset_path: str = "data/final_master_dataset_complete_final.json"

    # --- Planner (Neo4j graph is primary; LLM is a fallback) ---
    # If the effective planner confidence is below this value, the planner
    # asks the LLM to generate the next-agent plan.
    planner_confidence_threshold: float = 0.60

    # Optional LLM overrides for the planner fallback call. When planner_llm_model
    # is unset, the shared llm_model is used.
    planner_llm_model: str | None = None
    planner_llm_temperature: float = 0.0
    planner_llm_max_tokens: int = 256
    planner_llm_timeout: float = 60.0

    log_level: str = "INFO"


settings = Settings()
