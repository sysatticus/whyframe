"""Interactive setup wizard for Whyframe."""
import sys
from pathlib import Path
import json
import requests

from whyframe.core.config import Config


def fetch_models(base_url: str) -> list[str]:
    """Fetch available embedding models from API."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/v1/embeddings/models",
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                return [m.get("id", m.get("model", "unknown")) for m in data["data"]]
            elif isinstance(data, list):
                return [m.get("id", m.get("model", "unknown")) for m in data]
    except Exception:
        pass
    return []


def print_header():
    """Print setup header."""
    print("\n" + "=" * 50)
    print("  Whyframe Setup Wizard")
    print("=" * 50 + "\n")


def prompt_choice(prompt: str, options: list[str]) -> str:
    """Prompt user to choose from a list."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        try:
            choice = input("\nEnter choice (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, IndexError):
            pass
        print("Invalid choice. Try again.")


def prompt_input(prompt: str, default: str = "") -> str:
    """Prompt user for input."""
    if default:
        result = input(f"\n{prompt} [{default}]: ").strip()
        return result or default
    else:
        result = input(f"\n{prompt}: ").strip()
        return result


def run_setup():
    """Run the interactive setup wizard."""
    print_header()
    
    # Step 1: Embedding provider
    print("Step 1: Choose Embedding Provider")
    print("-" * 40)
    providers = [
        "OpenAI (api.openai.com)",
        "OpenAI-Compatible API (custom endpoint)",
        "Local (sentence-transformers)",
    ]
    provider_choice = prompt_choice("Select provider:", providers)
    
    config = Config()
    
    if "OpenAI-Compatible" in provider_choice:
        config.embedding.provider = "openai"
        config.embedding.base_url = prompt_input("Enter base URL", "http://localhost:11434/v1")
        config.embedding.api_key = prompt_input("Enter API key (or press enter for none)", "")
        
        # Step 2: Fetch or enter models
        print("\nStep 2: Choose Embedding Model")
        print("-" * 40)
        
        model_options = ["Enter manually"]
        if config.embedding.base_url:
            print("Fetching available models...")
            models = fetch_models(config.embedding.base_url)
            if models:
                model_options = ["Fetch from API", "Enter manually"]
                print(f"Found {len(models)} models!")
            else:
                print("Could not fetch models. Enter manually.")
        
        model_choice = prompt_choice("How to choose model:", model_options)
        
        if model_choice == "Fetch from API":
            models = fetch_models(config.embedding.base_url)
            if models:
                config.embedding.model = prompt_choice("Select model:", models)
            else:
                config.embedding.model = prompt_input("Enter model name:", "text-embedding-3-small")
        else:
            print("\nAvailable models for reference:")
            print("  - qwen/qwen3-embedding-8b")
            print("  - mistralai/codestral-embed-2505")
            print("  - openai/text-embedding-3-small")
            print("  - openai/text-embedding-3-large")
            print("  - thenlper/gte-large")
            print("  - intfloat/multilingual-e5-large")
            config.embedding.model = prompt_input("Enter model name:")
        
        # Set dimension based on model
        from whyframe.pipeline.embeddings import EmbeddingPipeline
        config.embedding.dimension = EmbeddingPipeline.get_dimension(config.embedding.model)
        
    elif "OpenAI" in provider_choice and "Compatible" not in provider_choice:
        config.embedding.provider = "openai"
        config.embedding.model = prompt_choice("Select model:", [
            "text-embedding-3-small",
            "text-embedding-3-large",
        ])
        config.embedding.dimension = 1536 if "small" in config.embedding.model else 3072
        config.embedding.api_key = prompt_input("Enter OpenAI API key:")
    else:
        config.embedding.provider = "local"
        config.embedding.model = "sentence-transformers"
        config.embedding.dimension = 384
    
    # Step 3: Vector DB
    print("\nStep 3: Choose Vector Database")
    print("-" * 40)
    vdb_options = ["Skip for now (in-memory)", "Pinecone", "Weaviate", "PostgreSQL (pgvector)"]
    vdb_choice = prompt_choice("Select vector DB:", vdb_options)
    
    if "Pinecone" in vdb_choice:
        config.vector_db.provider = "pinecone"
        config.vector_db.api_key = prompt_input("Enter Pinecone API key:")
        config.vector_db.environment = prompt_input("Enter environment:", "us-west1")
        config.vector_db.index_name = prompt_input("Enter index name:", "whyframe")
    elif "Weaviate" in vdb_choice:
        config.vector_db.provider = "weaviate"
        config.vector_db.api_key = prompt_input("Enter Weaviate API key (optional):")
        config.vector_db.environment = prompt_input("Enter Weaviate URL:", "http://localhost:8080")
    elif "PostgreSQL" in vdb_choice:
        config.vector_db.provider = "pgvector"
    
    # Step 4: Git config
    print("\nStep 4: Git Configuration")
    print("-" * 40)
    config.git.max_commit_history = int(prompt_input("Max commits to index:", "10000"))
    ignored = prompt_input("Ignored paths (comma-separated):", ".git,__pycache__,node_modules")
    config.git.ignored_paths = [p.strip() for p in ignored.split(",")]
    
    # Save config
    print("\n" + "=" * 50)
    print("  Setup Complete!")
    print("=" * 50)
    
    config_path = Path("whyframe-config.yaml")
    config.to_file(config_path)
    print(f"\nConfig saved to: {config_path}")
    
    print("\nNext steps:")
    print(f"  1. Review config: nano {config_path}")
    print(f"  2. Index a repo: whyframe index /path/to/repo --config {config_path}")
    
    return config


if __name__ == "__main__":
    run_setup()