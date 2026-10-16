"""LLM client. Any OpenAI-compatible endpoint works: AMD tokenfactory or vLLM on AMD Developer Cloud."""
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def __getattr__(name: str):
    if name == "MOCK":
        load_dotenv(override=True)
        return os.getenv("MOCK_LLM") == "1" or not bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_API_KEY"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_client() -> OpenAI:
    global _client
    load_dotenv(override=True)
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    if not base_url or not api_key:
        raise KeyError(
            "Missing LLM credentials: Set LLM_BASE_URL and LLM_API_KEY in your .env file, or set MOCK_LLM=1 for offline testing."
        )
    return OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)


def chat(system: str, user: str, temperature: float = 0.1, max_tokens: int = 350) -> str:
    load_dotenv(override=True)
    model = os.getenv("LLM_MODEL", "")
    r = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (r.choices[0].message.content or "").strip()


def chat_json(system: str, user: str, max_tokens: int = 300) -> dict:
    for _ in range(2):  # one retry if the model returns invalid JSON
        raw = chat(system + "\nReply with ONLY a valid JSON object, no markdown.", user, max_tokens=max_tokens)
        # Strip potential think / reasoning tags
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        # Strip code fences
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: extract outermost JSON object {...}
            m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            continue
    raise ValueError("Model did not return valid JSON")
