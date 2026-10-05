"""Server configuration for GLM-API."""

import os

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
API_KEY = os.getenv("API_KEY", "").strip()
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

GLM_WORKERS = int(os.getenv("GLM_WORKERS", "4"))
GLM_REQUEST_TIMEOUT = float(os.getenv("GLM_REQUEST_TIMEOUT", "30.0"))
UPSTREAM_RETRY_AFTER = os.getenv("UPSTREAM_RETRY_AFTER", "30")
DEBUG_REQUESTS = os.getenv("DEBUG_REQUESTS", "").lower() in ("1", "true", "yes", "on")

MODEL_MAP = {
    "glm-4-flash":        "glm-4-flash",        # Free fast default model
    "glm-4":              "glm-4",              # Standard flagship model
    "glm-4-plus":         "glm-4-plus",         # Strongest flagship
    "glm-4-air":          "glm-4-air",          # Lightweight high-speed
    "glm-4v":             "glm-4v",             # Multimodal vision model
    "glm-4-reasoner":     "glm-zero-preview",   # Reasoning model with thinking tokens
    "glm-zero-preview":   "glm-zero-preview",   # Reasoning model
    # Common IDE agent aliases
    "gpt-4o":             "glm-4-plus",
    "gpt-4":              "glm-4",
    "gpt-3.5-turbo":      "glm-4-flash",
}

DEFAULT_MODEL = "glm-4-flash"


def is_known_model(name: str) -> bool:
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    return MODEL_MAP[name]


def should_enable_thinking(model: str, requested_thinking: bool = False) -> bool:
    return requested_thinking or model in ("glm-4-reasoner", "glm-zero-preview")
