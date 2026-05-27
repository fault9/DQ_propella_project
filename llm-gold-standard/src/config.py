"""Configuration: datasets, sampling/ordinal constants, defaults, provider presets.

Provider model: there is ONE OpenAI-compatible code path. "openai" talks to the
real OpenAI API; "berget" talks to Berget AI (EU, OpenAI-compatible) by setting a
base_url; any other OpenAI-compatible endpoint works via --base_url. "anthropic"
uses the native Anthropic SDK.
"""
from __future__ import annotations

# --- Datasets --------------------------------------------------------------
PROPELLA_DATASET = "openeurollm/propella-annotations"
PROPELLA_SUBSET = "finepdfs"                 # NOTE: finepdfs, NOT fineweb-2
FINEPDFS_DATASET = "HuggingFaceFW/finepdfs"

DEFAULT_LANGUAGE = "swe_Latn"
PREFERRED_LANGUAGES = [
    "swe_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn",
    "nld_Latn", "por_Latn", "pol_Latn", "dan_Latn", "nob_Latn",
]

# --- Sampling --------------------------------------------------------------
POOL_SIZE = 100_000         # docs streamed into memory before stratified sampling
                            # (larger pool -> rare strata like excellent×excellent are findable)
MIN_PER_STRATUM = 2         # proportional mode: every non-empty stratum gets at least this many
DEFAULT_SAMPLING_MODE = "uniform"   # 'uniform' (balanced, best for a ranking gold standard) or 'proportional'

# Ordinal maps for the two stratification features (higher = better).
ORDINAL_MAPS = {
    "educational_value": {"none": 0, "minimal": 1, "basic": 2, "moderate": 3, "high": 4},
    "content_quality": {"unacceptable": 0, "poor": 1, "adequate": 2, "good": 3, "excellent": 4},
}
STRATA_FEATURES = ["educational_value", "content_quality"]
MAX_STRATA = 25             # 5 x 5

# --- Text fetching ---------------------------------------------------------
MAX_TEXT_CHARS = 50_000

# --- LLM scoring -----------------------------------------------------------
DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_SEED = 42
DEFAULT_PROVIDER = "berget"          # Berget AI (EU, OpenAI-compatible)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # strong judge; same family FineWeb-Edu used as annotator
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"        # default when provider=anthropic (--anthropic shortcut)
DEFAULT_MAX_CHARS = 3000             # chars sent to the LLM (cost lever; ~$2 for 2000 docs on Llama-70B)


def default_model_for(provider: str) -> str:
    """Pick the default model for a provider (so --anthropic needs no --model)."""
    return DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else DEFAULT_MODEL

# Approximate USD per 1M tokens (input, output). VERIFY against current pricing.
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
    # Berget (open models). EUR list price -> USD (~1.08), ex VAT — verify current rate/VAT.
    "openai/gpt-oss-120b": (0.22, 0.81),                       # €0.20 / €0.75 (reasoning model)
    "google/gemma-4-31B-it": (0.27, 0.54),                     # €0.25 / €0.50
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": (0.32, 0.32),  # €0.30 / €0.30
    "meta-llama/Llama-3.3-70B-Instruct": (0.97, 0.97),         # €0.90 / €0.90
    "meta-llama/Llama-3.1-8B-Instruct": (0.22, 0.22),          # €0.20 / €0.20
    "zai-org/GLM-4.7-FP8": (0.76, 2.70),                       # €0.70 / €2.50
    "moonshotai/Kimi-K2.6": (0.81, 3.78),                      # €0.75 / €3.50
    "mistralai/Mistral-Medium-3.5-128B": (1.62, 5.40),         # €1.50 / €5.00
}
DEFAULT_PRICING = (3.0, 15.0)        # fallback when the model isn't in the table


def get_pricing(model: str):
    """Return (price_in_per_M, price_out_per_M, is_known)."""
    if model in MODEL_PRICING:
        return (*MODEL_PRICING[model], True)
    return (*DEFAULT_PRICING, False)
DEFAULT_BATCH_SIZE = 5               # parallel API requests
DEFAULT_MAX_TOKENS = 100   # room for the echoed id + score JSON (a urn:uuid id is ~45 chars)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_PROMPT_FILE = "prompts/quality_prompt.txt"

CHECKPOINT_EVERY = 50               # save partial results every N docs
MAX_RETRIES = 3                     # API retries before giving up on a doc
BACKOFF_BASE = 2.0                  # exponential backoff base (seconds)

# How the two LLM axes (educational_value, content_quality) combine into the final
# quality_score. 'geometric' = sqrt(edu*quality): a doc scores high only if BOTH
# axes are strong (penalizes lopsided docs). Alternatives: 'mean', 'min'. Both raw
# axes are stored, so this is recomputable in code without re-running the LLM.
COMBINE_MODE = "geometric"

# Per-provider defaults. "berget" is OpenAI-compatible with a preset base_url.
PROVIDER_PRESETS = {
    "anthropic": {"base_url": None, "api_key_env": "ANTHROPIC_API_KEY", "openai_compatible": False},
    "openai":    {"base_url": None, "api_key_env": "OPENAI_API_KEY", "openai_compatible": True},
    "berget":    {"base_url": "https://api.berget.ai/v1", "api_key_env": "BERGET_API_KEY",
                  "openai_compatible": True},
}


def build_api_config(provider: str, model: str, max_tokens: int, temperature: float,
                     base_url: str | None = None, api_key_env: str | None = None) -> dict:
    """Resolve a full api_config from a provider name + overrides."""
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDER_PRESETS)}.")
    return {
        "provider": provider,
        "openai_compatible": preset["openai_compatible"],
        "model": model,
        "base_url": base_url if base_url is not None else preset["base_url"],
        "api_key_env": api_key_env or preset["api_key_env"],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
