"""
Configuration dataclasses for the RAG evaluation pipeline.

Defines model provider settings, per-run evaluation parameters, and a set of
ready-made model presets so callers don't have to look up model IDs.

Example::

    from evaluation.config import EvaluationConfig, PRESET_MODELS

    config = EvaluationConfig(
        rag_model=PRESET_MODELS["phi3"],
        judge_model=PRESET_MODELS["gpt4o-mini"],
        metrics=["faithfulness", "context_precision"],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelProvider(str, Enum):
    """Supported LLM backend providers."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    """
    Settings for a single LLM.

    Attributes:
        provider: Backend that hosts the model.
        model_id: Provider-specific model identifier (HuggingFace Hub path,
            OpenAI model name, or Anthropic model ID).
        api_key: API key for cloud providers.  Falls back to the corresponding
            environment variable (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``)
            when ``None``.
        max_new_tokens: Upper bound on tokens generated per response.
        temperature: Sampling temperature; lower values are more deterministic.
    """

    provider: ModelProvider
    model_id: str
    api_key: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.1


@dataclass
class EvaluationConfig:
    """
    Top-level configuration for one evaluation run.

    Attributes:
        rag_model: Model used for answer generation inside the RAG chain.
        judge_model: Model used by the LLM-as-a-Judge and RAGAS evaluators.
            Falls back to ``rag_model`` when ``None``.
        metrics: RAGAS metric names to compute.  Supported values:
            ``"faithfulness"``, ``"context_precision"``,
            ``"context_recall"``, ``"answer_relevancy"``.
        output_dir: Directory where CSV result files are written.
        dataset_path: Optional path to a custom JSON evaluation dataset.
            Uses the built-in sample dataset when ``None``.
        vectorstore_dir: Path to the persisted Chroma vector store.
    """

    rag_model: ModelConfig
    judge_model: Optional[ModelConfig] = None
    metrics: list[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
        ]
    )
    output_dir: str = "evaluation_results"
    dataset_path: Optional[str] = None
    vectorstore_dir: str = "vectorstore"


# ------------------------------------------------------------------ #
#  Ready-made model presets                                            #
# ------------------------------------------------------------------ #

PRESET_MODELS: dict[str, ModelConfig] = {
    # --- Local / HuggingFace Hub ---
    "phi3": ModelConfig(
        provider=ModelProvider.HUGGINGFACE,
        model_id="microsoft/Phi-3-mini-4k-instruct",
    ),
    "mistral": ModelConfig(
        provider=ModelProvider.HUGGINGFACE,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "llama3": ModelConfig(
        provider=ModelProvider.HUGGINGFACE,
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    # --- OpenAI ---
    "gpt4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
    ),
    "gpt4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
    ),
    # --- Anthropic ---
    "claude-haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
    ),
    "claude-sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-6",
    ),
}
