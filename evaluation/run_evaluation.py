"""
CLI entry point for RAG evaluation.

Generates answers from the RAG pipeline, computes RAGAS metrics
(faithfulness, context precision/recall, answer relevancy), and scores
outputs with an LLM judge (faithfulness, relevance, correctness, coherence).
Supports single-model evaluation and side-by-side multi-model comparison.

Quick start
-----------
::

    # Single model, built-in sample dataset
    python -m evaluation.run_evaluation --model phi3

    # Cloud judge, custom dataset
    python -m evaluation.run_evaluation \\
        --model phi3 \\
        --judge gpt4o-mini \\
        --dataset evaluation/sample_dataset.json

    # Compare multiple models
    python -m evaluation.run_evaluation \\
        --compare phi3 mistral gpt4o-mini \\
        --judge claude-sonnet

    # RAGAS only (skip judge)
    python -m evaluation.run_evaluation --model phi3 --skip-judge

    # Judge only (skip RAGAS)
    python -m evaluation.run_evaluation --model phi3 --skip-ragas

Available model presets
-----------------------
phi3, mistral, llama3          — local HuggingFace models
gpt4o-mini, gpt4o              — OpenAI  (requires OPENAI_API_KEY)
claude-haiku, claude-sonnet    — Anthropic (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow running as ``python -m evaluation.run_evaluation`` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import PRESET_MODELS, EvaluationConfig, ModelConfig
from evaluation.dataset import EvalDataset
from evaluation.llm_judge import LLMJudgeEvaluator
from evaluation.models import create_llm
from evaluation.ragas_eval import RAGASEvaluator


# ------------------------------------------------------------------ #
#  Answer generation                                                   #
# ------------------------------------------------------------------ #


def generate_answers(
    rag_model_config: ModelConfig,
    dataset: EvalDataset,
    vectorstore_dir: str = "vectorstore",
) -> tuple[list[str], list[list[str]]]:
    """
    Run every dataset question through the RAG pipeline.

    Builds a fresh RAG chain using *rag_model_config*, then calls
    :func:`rag.query` for each sample.

    Args:
        rag_model_config: Config for the generation model.
        dataset: Evaluation dataset whose questions are used.
        vectorstore_dir: Path to the persisted Chroma vector store.

    Returns:
        ``(answers, contexts)`` — parallel lists where ``answers[i]`` is
        the RAG answer for ``dataset.samples[i]`` and ``contexts[i]`` is
        the list of retrieved ``page_content`` strings.
    """
    from rag import build_rag_chain, query  # imported here to avoid loading torch at CLI parse time

    print(
        f"\nLoading RAG model: {rag_model_config.model_id} "
        f"({rag_model_config.provider})"
    )
    llm = create_llm(rag_model_config)
    chain, retriever = build_rag_chain(llm, vectorstore_dir=vectorstore_dir)

    answers: list[str] = []
    contexts: list[list[str]] = []

    for i, sample in enumerate(dataset.samples, 1):
        print(f"  [{i}/{len(dataset)}] {sample.question[:70]}...")
        answer, _ = query(chain, retriever, sample.question)
        docs = retriever.invoke(sample.question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])

    return answers, contexts


# ------------------------------------------------------------------ #
#  Single-model evaluation                                             #
# ------------------------------------------------------------------ #


def evaluate_single_model(
    model_name: str,
    config: EvaluationConfig,
    dataset: EvalDataset,
    run_ragas: bool = True,
    run_judge: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run the full evaluation suite for one RAG model.

    Args:
        model_name: Human-readable label used as a prefix for output files.
        config: Full evaluation configuration for this run.
        dataset: Evaluation dataset.
        run_ragas: Whether to compute RAGAS metrics.
        run_judge: Whether to run the LLM-as-a-Judge evaluation.

    Returns:
        Dict mapping ``"ragas"`` and/or ``"judge"`` to their result DataFrames.
    """
    _section(f"Evaluating model: {model_name}")

    answers, contexts = generate_answers(
        config.rag_model, dataset, config.vectorstore_dir
    )
    results: dict[str, pd.DataFrame] = {}

    if run_ragas:
        print("\n--- RAGAS evaluation ---")
        ragas_ev = RAGASEvaluator(config)
        ragas_df = ragas_ev.run(dataset, answers, contexts)
        ragas_ev.save_results(ragas_df, f"{model_name}_ragas_results.csv")
        results["ragas"] = ragas_df
        _print_summary("RAGAS", ragas_df, config.metrics)

    if run_judge:
        print("\n--- LLM-as-a-Judge evaluation ---")
        judge_ev = LLMJudgeEvaluator(config)
        judge_df = judge_ev.run(dataset, answers, contexts)
        judge_ev.save_results(judge_df, f"{model_name}_judge_results.csv")
        results["judge"] = judge_df
        _print_summary(
            "LLM Judge",
            judge_df,
            ["faithfulness", "relevance", "correctness", "coherence"],
        )

    return results


# ------------------------------------------------------------------ #
#  Multi-model comparison                                              #
# ------------------------------------------------------------------ #


def compare_models(
    model_names: list[str],
    judge_model_name: Optional[str],
    dataset: EvalDataset,
    output_dir: str,
    run_ragas: bool,
    run_judge: bool,
    vectorstore_dir: str,
    metrics: list[str],
) -> None:
    """
    Evaluate multiple models and write a side-by-side comparison CSV.

    Each model is evaluated independently.  Mean scores from RAGAS and the
    LLM judge are merged into a single ``model_comparison.csv``.

    Args:
        model_names: Ordered list of preset model names to compare.
        judge_model_name: Preset name for the judge/evaluator model.
            ``None`` lets each provider use its default.
        dataset: Shared evaluation dataset for all models.
        output_dir: Directory for all output files.
        run_ragas: Whether to include RAGAS metrics.
        run_judge: Whether to include LLM-as-a-Judge metrics.
        vectorstore_dir: Path to the Chroma vector store.
        metrics: RAGAS metric names to compute.
    """
    comparison_rows: list[dict] = []

    for model_name in model_names:
        if model_name not in PRESET_MODELS:
            print(f"[WARN] Unknown preset '{model_name}' — skipping.")
            continue

        rag_model = PRESET_MODELS[model_name]
        judge_model = PRESET_MODELS[judge_model_name] if judge_model_name else None

        config = EvaluationConfig(
            rag_model=rag_model,
            judge_model=judge_model,
            metrics=metrics,
            output_dir=output_dir,
            vectorstore_dir=vectorstore_dir,
        )
        result_dfs = evaluate_single_model(
            model_name, config, dataset, run_ragas=run_ragas, run_judge=run_judge
        )

        row: dict = {"model": model_name}
        if "ragas" in result_dfs:
            mean = result_dfs["ragas"][result_dfs["ragas"]["question"] == "MEAN"].iloc[0]
            for m in metrics:
                if m in mean:
                    row[f"ragas_{m}"] = round(float(mean[m]), 4)
        if "judge" in result_dfs:
            mean = result_dfs["judge"][result_dfs["judge"]["question"] == "MEAN"].iloc[0]
            for c in ["faithfulness", "relevance", "correctness", "coherence"]:
                if c in mean:
                    row[f"judge_{c}"] = round(float(mean[c]), 4)

        comparison_rows.append(row)

    if not comparison_rows:
        print("No valid models evaluated — nothing to compare.")
        return

    comparison_df = pd.DataFrame(comparison_rows)
    out_path = Path(output_dir) / "model_comparison.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_path, index=False)

    _section("MODEL COMPARISON")
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved → {out_path}")


# ------------------------------------------------------------------ #
#  Display helpers                                                     #
# ------------------------------------------------------------------ #


def _section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def _print_summary(label: str, df: pd.DataFrame, metric_cols: list[str]) -> None:
    """Print mean scores for a quick visual check."""
    try:
        mean_row = df[df["question"] == "MEAN"].iloc[0]
    except IndexError:
        return
    print(f"\n{label} — mean scores:")
    for col in metric_cols:
        if col in mean_row and pd.notna(mean_row[col]):
            print(f"  {col:<30} {float(mean_row[col]):.4f}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_evaluation",
        description="RAG pipeline evaluation — RAGAS metrics + LLM-as-a-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available model presets: {', '.join(PRESET_MODELS)}",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model",
        metavar="PRESET",
        help="Single model preset to evaluate.",
    )
    mode.add_argument(
        "--compare",
        nargs="+",
        metavar="PRESET",
        help="Two or more model presets to compare side-by-side.",
    )

    parser.add_argument(
        "--judge",
        metavar="PRESET",
        default=None,
        help="Model preset to use as the evaluator/judge (default: same as --model).",
    )
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        default=None,
        help="Path to a custom evaluation dataset JSON (default: built-in sample).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "context_precision", "context_recall", "answer_relevancy"],
        metavar="METRIC",
        help="RAGAS metrics to compute.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        metavar="DIR",
        help="Directory for CSV output files (default: evaluation_results/).",
    )
    parser.add_argument(
        "--vectorstore-dir",
        default="vectorstore",
        metavar="DIR",
        help="Path to the persisted Chroma vector store (default: vectorstore/).",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS metric computation.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-a-Judge evaluation.",
    )
    return parser


def main() -> None:
    """Entry point for the evaluation CLI."""
    args = _build_parser().parse_args()

    dataset = (
        EvalDataset.from_json(args.dataset) if args.dataset else EvalDataset.sample_dataset()
    )
    print(f"Dataset loaded: {dataset}")

    if args.compare:
        compare_models(
            model_names=args.compare,
            judge_model_name=args.judge,
            dataset=dataset,
            output_dir=args.output_dir,
            run_ragas=not args.skip_ragas,
            run_judge=not args.skip_judge,
            vectorstore_dir=args.vectorstore_dir,
            metrics=args.metrics,
        )
    else:
        if args.model not in PRESET_MODELS:
            print(f"Error: unknown preset '{args.model}'. Available: {list(PRESET_MODELS)}")
            sys.exit(1)

        config = EvaluationConfig(
            rag_model=PRESET_MODELS[args.model],
            judge_model=PRESET_MODELS[args.judge] if args.judge else None,
            metrics=args.metrics,
            output_dir=args.output_dir,
            vectorstore_dir=args.vectorstore_dir,
        )
        evaluate_single_model(
            model_name=args.model,
            config=config,
            dataset=dataset,
            run_ragas=not args.skip_ragas,
            run_judge=not args.skip_judge,
        )


if __name__ == "__main__":
    main()
