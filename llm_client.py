"""Thin wrapper around the Ollama local LLM HTTP API.

Provides a simple interface for generating text with Ollama models while
tracking inference latency for performance monitoring.
"""

import json
import sys
import time

try:
    import requests
except ImportError:
    requests = None


# -----------------
# Configuration
# -----------------
DEFAULT_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 300


def get_ollama_readiness(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 2,
) -> dict:
    """Return Ollama server/model readiness details for lightweight preflights."""
    status = {
        "ready": False,
        "server_available": False,
        "model_available": False,
        "models": [],
        "error": "",
    }

    if requests is None:
        status["error"] = "The 'requests' package is not installed."
        return status

    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = [model["name"] for model in response.json().get("models", [])]
    except requests.ConnectionError:
        status["error"] = "Cannot connect to Ollama. Is it running? (ollama serve)"
        return status
    except requests.Timeout:
        status["error"] = f"Ollama readiness check timed out after {timeout}s."
        return status
    except requests.HTTPError as exc:
        status["error"] = f"Ollama API error: {exc}"
        return status

    model_available = any(
        model_name == model or model.startswith(f"{model_name}:")
        for model in models
    )
    status.update({
        "ready": model_available,
        "server_available": True,
        "model_available": model_available,
        "models": models,
    })

    if not model_available:
        status["error"] = f"Model '{model_name}' is not pulled."

    return status


def is_ollama_ready(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 2,
) -> bool:
    """Return True when Ollama is reachable and the requested model is present."""
    return get_ollama_readiness(
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
    )["ready"]


class OllamaClient:
    """Client for the Ollama REST API (http://localhost:11434).

    Attributes:
        model: Name of the Ollama model to use for generation.
        base_url: Base URL of the running Ollama server.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        if requests is None:
            print("Error: 'requests' package is required.")
            print("Install with: pip install requests")
            sys.exit(1)

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -----------------
    # Health & Info
    # -----------------

    def is_available(self) -> bool:
        """Check if the Ollama server is running and reachable."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False

    def list_models(self) -> list[str]:
        """List locally available Ollama models."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except (requests.ConnectionError, requests.Timeout) as exc:
            print(f"Warning: cannot reach Ollama server — {exc}")
            return []
        except requests.HTTPError as exc:
            print(f"Warning: Ollama API error — {exc}")
            return []

    def model_is_pulled(self) -> bool:
        """Check if the configured model is available locally."""
        models = self.list_models()
        # Ollama names may include tag, e.g. "llama3:latest"
        return any(
            self.model == m or m.startswith(f"{self.model}:")
            for m in models
        )

    # -----------------
    # Generation
    # -----------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict:
        """Generate a response from the Ollama model.

        Args:
            prompt: The user prompt / question.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with keys: response, model, duration_ms, prompt_eval_count,
            eval_count, error (if any).
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.ConnectionError:
            return {
                "response": "",
                "model": self.model,
                "duration_ms": 0,
                "error": "Cannot connect to Ollama. Is it running? (ollama serve)",
            }
        except requests.Timeout:
            elapsed = (time.perf_counter() - start_time) * 1000
            return {
                "response": "",
                "model": self.model,
                "duration_ms": round(elapsed),
                "error": f"Request timed out after {self.timeout}s",
            }
        except requests.HTTPError as exc:
            return {
                "response": "",
                "model": self.model,
                "duration_ms": 0,
                "error": str(exc),
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "response": data.get("response", ""),
            "model": data.get("model", self.model),
            "duration_ms": round(elapsed_ms),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
            "error": None,
        }


# -----------------
# Standalone test
# -----------------
def main() -> None:
    """Quick connectivity and generation test."""
    client = OllamaClient()

    print(f"Ollama server: {client.base_url}")
    print(f"Model: {client.model}")
    print(f"Server reachable: {client.is_available()}")

    if not client.is_available():
        print("\nOllama is not running. Start it with: ollama serve")
        print("Then pull a model: ollama pull llama3")
        return

    models = client.list_models()
    print(f"Available models: {models}")

    if not client.model_is_pulled():
        print(f"\nModel '{client.model}' is not pulled.")
        print(f"Pull it with: ollama pull {client.model}")
        return

    print("\nSending test prompt...")
    result = client.generate(
        prompt="What are the three most common symptoms of type 2 diabetes? Answer briefly.",
        system_prompt="You are a helpful medical assistant. Be concise.",
        temperature=0.3,
        max_tokens=256,
    )

    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"\nResponse ({result['duration_ms']}ms):")
        print(result["response"])
        print(f"\nTokens — prompt: {result['prompt_eval_count']}, "
              f"generated: {result['eval_count']}")


if __name__ == "__main__":
    main()
