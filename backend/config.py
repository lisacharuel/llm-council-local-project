"""Configuration for the LLM Council with Ollama."""
import os
from dotenv import load_dotenv

load_dotenv()

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Council members - list of Ollama model identifiers
COUNCIL_MODELS = [
    "llama3:latest",
    "mistral:latest",
    "phi3:latest",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "llama3:latest"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
