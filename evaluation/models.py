"""
Multi-model factory for the RAG evaluation pipeline.

Returns LangChain-compatible LLM and Embeddings objects for any supported
provider so the rest of the evaluation code stays provider-agnostic.

Supported providers
-------------------
- **HuggingFace** — local models loaded via ``transformers`` + ``torch``.
  No API key required; models are downloaded from HuggingFace Hub on first use.
- **OpenAI** — requires ``pip install langchain-openai`` and ``OPENAI_API_KEY``.
- **Anthropic** — requires ``pip install langchain-anthropic`` and ``ANTHROPIC_API_KEY``.

Usage::

    from evaluation.config import PRESET_MODELS
    from evaluation.models import create_llm, create_embeddings

    llm = create_llm(PRESET_MODELS["gpt4o-mini"])
    embeddings = create_embeddings(PRESET_MODELS["gpt4o-mini"])
"""

from __future__ import annotations

import os
from typing import Any

from .config import ModelConfig, ModelProvider


def create_llm(config: ModelConfig) -> Any:
    """
    Instantiate a LangChain-compatible LLM from a :class:`~evaluation.config.ModelConfig`.

    Args:
        config: Describes the provider, model ID, optional API key, and
            generation hyperparameters.

    Returns:
        A LangChain ``BaseChatModel`` (OpenAI / Anthropic) or
        ``HuggingFacePipeline`` (HuggingFace).

    Raises:
        ImportError: If the required provider package is not installed.
        ValueError: If an unrecognised provider is specified.
    """
    if config.provider == ModelProvider.HUGGINGFACE:
        return _create_hf_llm(config)
    if config.provider == ModelProvider.OPENAI:
        return _create_openai_llm(config)
    if config.provider == ModelProvider.ANTHROPIC:
        return _create_anthropic_llm(config)
    raise ValueError(f"Unsupported provider: {config.provider!r}")


def create_embeddings(config: ModelConfig) -> Any:
    """
    Instantiate LangChain-compatible embeddings for the given provider.

    OpenAI returns ``OpenAIEmbeddings``; all other providers use
    ``HuggingFaceEmbeddings`` (sentence-transformers) so a GPU is not needed
    for evaluation runs that use cloud generation models.

    Args:
        config: Provider config — only the provider type is used for selection.

    Returns:
        A LangChain ``Embeddings`` instance.

    Raises:
        ImportError: If ``langchain-openai`` is not installed for OpenAI.
    """
    if config.provider == ModelProvider.OPENAI:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "OpenAI embeddings require: pip install langchain-openai"
            ) from exc
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddings(openai_api_key=api_key)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# ------------------------------------------------------------------ #
#  Private provider helpers                                            #
# ------------------------------------------------------------------ #


def _create_hf_llm(config: ModelConfig):
    """Load a local HuggingFace model wrapped in a LangChain pipeline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from langchain_huggingface import HuggingFacePipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=True,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)


def _create_openai_llm(config: ModelConfig):
    """Instantiate a ChatOpenAI model."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI provider requires: pip install langchain-openai"
        ) from exc

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model=config.model_id,
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        openai_api_key=api_key,
    )


def _create_anthropic_llm(config: ModelConfig):
    """Instantiate a ChatAnthropic model."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "Anthropic provider requires: pip install langchain-anthropic"
        ) from exc

    api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
    return ChatAnthropic(
        model=config.model_id,
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        anthropic_api_key=api_key,
    )
