"""
RAGAS-based evaluation for the RAG pipeline.

Computes four standard RAGAS metrics against a labelled evaluation dataset:

- **Faithfulness** — are all answer claims supported by the retrieved context?
- **Context Precision** — are the retrieved chunks relevant to the question?
- **Context Recall** — does the retrieved context cover the ground-truth answer?
- **Answer Relevancy** — does the answer actually address the question?

RAGAS requires an LLM for faithfulness / relevancy scoring and an embedding
model for answer-relevancy scoring.  By default the evaluator wraps whatever
is specified in ``config.judge_model``.  If ``judge_model`` is ``None``,
RAGAS falls back to its built-in default (OpenAI GPT-4o-mini), which requires
``OPENAI_API_KEY`` to be set.

Usage::

    from evaluation.config import EvaluationConfig, PRESET_MODELS
    from evaluation.dataset import EvalDataset
    from evaluation.ragas_eval import RAGASEvaluator

    config = EvaluationConfig(
        rag_model=PRESET_MODELS["phi3"],
        judge_model=PRESET_MODELS["gpt4o-mini"],
    )
    evaluator = RAGASEvaluator(config)
    df = evaluator.run(dataset, answers, contexts)
    evaluator.save_results(df)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset

from .config import EvaluationConfig
from .dataset import EvalDataset
from .models import create_embeddings, create_llm

# Metric names that require embedding similarity (answer relevancy).
_EMBEDDING_METRICS: frozenset[str] = frozenset({"answer_relevancy"})


def _load_ragas_metrics(names: list[str]) -> list:
    """
    Resolve a list of metric name strings to RAGAS metric objects.

    Args:
        names: Metric names to load.  Supported values: ``"faithfulness"``,
            ``"context_precision"``, ``"context_recall"``,
            ``"answer_relevancy"``.

    Returns:
        List of RAGAS metric instances in the same order as *names*.

    Raises:
        ImportError: If ``ragas`` is not installed.
        ValueError: If any name in *names* is not a supported metric.
    """
    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError("RAGAS evaluation requires: pip install ragas") from exc

    available: dict = {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_relevancy": answer_relevancy,
    }
    unknown = set(names) - available.keys()
    if unknown:
        raise ValueError(
            f"Unknown RAGAS metrics: {unknown}. "
            f"Supported: {sorted(available.keys())}"
        )
    return [available[n] for n in names]


class RAGASEvaluator:
    """
    Thin wrapper around ``ragas.evaluate`` with pluggable LLM/embeddings.

    The judge LLM and embeddings are built once at construction time from
    ``config.judge_model`` (if set) and reused across calls to :meth:`run`.

    Example::

        evaluator = RAGASEvaluator(config)
        results_df = evaluator.run(dataset, answers, contexts)
        evaluator.save_results(results_df, "phi3_ragas.csv")
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """
        Args:
            config: Evaluation configuration.  ``config.judge_model``
                determines the LLM and embeddings used inside RAGAS.
        """
        self.config = config
        self._llm = None
        self._embeddings = None

        if config.judge_model:
            self._llm = create_llm(config.judge_model)
            self._embeddings = create_embeddings(config.judge_model)

    def run(
        self,
        dataset: EvalDataset,
        answers: list[str],
        contexts: list[list[str]],
    ) -> pd.DataFrame:
        """
        Compute RAGAS metrics for every sample and return a results DataFrame.

        A ``MEAN`` row is appended at the bottom for quick aggregate inspection.

        Args:
            dataset: Evaluation dataset supplying questions and ground truths.
            answers: Generated answers, one per dataset sample.
            contexts: Retrieved context strings per sample; each element is a
                list of ``page_content`` strings.

        Returns:
            DataFrame with one row per sample and one column per metric, plus
            a trailing ``MEAN`` row with column averages.

        Raises:
            ImportError: If ``ragas`` or ``datasets`` are not installed.
        """
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        ragas_data = dataset.to_ragas_format(answers, contexts)
        hf_dataset = Dataset.from_dict(ragas_data)
        metrics = _load_ragas_metrics(self.config.metrics)

        kwargs: dict = {}
        if self._llm:
            kwargs["llm"] = LangchainLLMWrapper(self._llm)
        if self._embeddings:
            # Only supply embeddings when answer_relevancy is requested —
            # other metrics don't need them and supplying unused wrappers
            # can slow down evaluation unnecessarily.
            needs_embeddings = any(
                m in self.config.metrics for m in _EMBEDDING_METRICS
            )
            if needs_embeddings:
                kwargs["embeddings"] = LangchainEmbeddingsWrapper(self._embeddings)

        result = evaluate(hf_dataset, metrics=metrics, **kwargs)
        df = result.to_pandas()

        mean_values = df[self.config.metrics].mean().to_dict()
        mean_values["question"] = "MEAN"
        df = pd.concat([df, pd.DataFrame([mean_values])], ignore_index=True)
        return df

    def save_results(
        self,
        df: pd.DataFrame,
        filename: str = "ragas_results.csv",
    ) -> Path:
        """
        Write the results DataFrame to a CSV file.

        Args:
            df: Results from :meth:`run`.
            filename: Output filename inside ``config.output_dir``.

        Returns:
            Absolute path to the written file.
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        df.to_csv(path, index=False)
        print(f"RAGAS results saved → {path}")
        return path
