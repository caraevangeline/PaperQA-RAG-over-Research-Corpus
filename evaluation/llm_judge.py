"""
LLM-as-a-Judge evaluation for the RAG pipeline.

Uses a language model to score RAG outputs on four criteria using structured
rubric prompts that ask the judge to respond with a JSON object containing a
``score`` (1–5) and a brief ``explanation``.

Criteria
--------
- **Faithfulness** — every claim in the answer is supported by the context.
- **Relevance** — the answer directly addresses the question.
- **Correctness** — the answer matches the factual content of the reference.
- **Coherence** — the answer is well-structured and grammatically sound.

The judge can be any LangChain-compatible model (local HuggingFace, OpenAI,
Anthropic) configured via :class:`~evaluation.config.EvaluationConfig`.

Usage::

    from evaluation.config import EvaluationConfig, PRESET_MODELS
    from evaluation.dataset import EvalDataset
    from evaluation.llm_judge import LLMJudgeEvaluator

    config = EvaluationConfig(
        rag_model=PRESET_MODELS["phi3"],
        judge_model=PRESET_MODELS["claude-sonnet"],
    )
    evaluator = LLMJudgeEvaluator(config)
    df = evaluator.run(dataset, answers, contexts)
    evaluator.save_results(df)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from .config import EvaluationConfig
from .dataset import EvalDataset
from .models import create_llm


# ------------------------------------------------------------------ #
#  Criteria & score types                                              #
# ------------------------------------------------------------------ #


class JudgeCriteria(str, Enum):
    """Evaluation criteria available to the LLM judge."""

    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"


@dataclass
class JudgementScore:
    """
    A single score produced by the LLM judge.

    Attributes:
        criteria: Which criterion was evaluated.
        score: Integer score on a 1–5 rubric (0 indicates a parse failure).
        explanation: Judge's natural-language rationale for the score.
        question: The question that was evaluated.
        answer: The answer that was scored.
    """

    criteria: JudgeCriteria
    score: int
    explanation: str
    question: str
    answer: str


# ------------------------------------------------------------------ #
#  Rubric prompts                                                      #
# ------------------------------------------------------------------ #

_FAITHFULNESS_PROMPT = """\
You are an objective evaluator. Score the FAITHFULNESS of the answer on a \
scale from 1 to 5.

FAITHFULNESS: every claim in the answer is directly supported by the context.
5 — All claims fully supported by context.
4 — Most claims supported; minor unsupported details.
3 — Some claims supported; notable unsupported content.
2 — Few claims supported; mostly fabricated or hallucinated.
1 — Answer contradicts or ignores the context entirely.

Context:
{context}

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "explanation": "<brief reason>"}}\
"""

_RELEVANCE_PROMPT = """\
You are an objective evaluator. Score the RELEVANCE of the answer on a \
scale from 1 to 5.

RELEVANCE: the answer directly addresses what the question asked.
5 — Fully and precisely addresses the question.
4 — Addresses the question with minor tangents.
3 — Partially addresses the question.
2 — Mostly off-topic.
1 — Does not address the question at all.

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "explanation": "<brief reason>"}}\
"""

_CORRECTNESS_PROMPT = """\
You are an objective evaluator. Score the CORRECTNESS of the answer on a \
scale from 1 to 5.

CORRECTNESS: the generated answer contains the same key facts as the reference.
5 — All key facts match the reference; may include correct additions.
4 — Most key facts match; minor omissions or inaccuracies.
3 — Some key facts match; notable omissions or inaccuracies.
2 — Few facts match; mostly incorrect or incomplete.
1 — Factually wrong or contradicts the reference.

Question: {question}
Reference Answer: {ground_truth}
Generated Answer: {answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "explanation": "<brief reason>"}}\
"""

_COHERENCE_PROMPT = """\
You are an objective evaluator. Score the COHERENCE of the answer on a \
scale from 1 to 5.

COHERENCE: the answer is well-structured, grammatically correct, and easy to follow.
5 — Perfectly clear, logically organised, no grammar issues.
4 — Clear and mostly well-organised; trivial grammar issues.
3 — Somewhat clear; noticeable structural or grammar problems.
2 — Difficult to follow; significant issues.
1 — Incoherent or incomprehensible.

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "explanation": "<brief reason>"}}\
"""

_PROMPTS: dict[JudgeCriteria, str] = {
    JudgeCriteria.FAITHFULNESS: _FAITHFULNESS_PROMPT,
    JudgeCriteria.RELEVANCE: _RELEVANCE_PROMPT,
    JudgeCriteria.CORRECTNESS: _CORRECTNESS_PROMPT,
    JudgeCriteria.COHERENCE: _COHERENCE_PROMPT,
}


# ------------------------------------------------------------------ #
#  Score parsing                                                       #
# ------------------------------------------------------------------ #


def _parse_score(
    raw: str,
    criteria: JudgeCriteria,
    question: str,
    answer: str,
) -> JudgementScore:
    """
    Extract a numeric score and explanation from the judge's raw text output.

    Searches for the first JSON object in *raw* and reads the ``score`` and
    ``explanation`` keys.  Returns score=0 on any parse failure so the
    caller can detect and flag bad responses without crashing.

    Args:
        raw: Full text response from the judge LLM.
        criteria: The criterion that was evaluated.
        question: The original question (stored in the returned object).
        answer: The answer that was scored (stored in the returned object).

    Returns:
        :class:`JudgementScore` with parsed values, or score=0 on failure.
    """
    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return JudgementScore(
                criteria=criteria,
                score=int(data["score"]),
                explanation=data.get("explanation", ""),
                question=question,
                answer=answer,
            )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    return JudgementScore(
        criteria=criteria,
        score=0,
        explanation=f"[Parse error] Raw: {raw[:300]}",
        question=question,
        answer=answer,
    )


# ------------------------------------------------------------------ #
#  LLM judge                                                           #
# ------------------------------------------------------------------ #


class LLMJudge:
    """
    Uses an LLM to score individual RAG outputs against rubric prompts.

    The judge LLM is built once at construction time.  Methods for each
    criterion are available individually or via :meth:`judge_all` for
    convenience.

    Example::

        judge = LLMJudge(config)
        result = judge.judge_faithfulness(
            question="What is YOLO?",
            answer="YOLO is ...",
            context="YOLO: You Only Look Once ...",
        )
        print(result.score, result.explanation)
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """
        Args:
            config: Evaluation config; ``judge_model`` is used for scoring.
                Falls back to ``rag_model`` when ``judge_model`` is ``None``.
        """
        judge_cfg = config.judge_model or config.rag_model
        self._llm = create_llm(judge_cfg)

    def _invoke(self, prompt: str) -> str:
        """Send *prompt* to the judge LLM and return the raw text."""
        response = self._llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def judge_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgementScore:
        """
        Score whether every claim in *answer* is supported by *context*.

        Args:
            question: The original question posed to the RAG system.
            answer: The RAG-generated answer.
            context: Retrieved context string shown to the RAG system.

        Returns:
            :class:`JudgementScore` on the 1–5 faithfulness rubric.
        """
        prompt = _PROMPTS[JudgeCriteria.FAITHFULNESS].format(
            context=context, question=question, answer=answer
        )
        return _parse_score(self._invoke(prompt), JudgeCriteria.FAITHFULNESS, question, answer)

    def judge_relevance(self, question: str, answer: str) -> JudgementScore:
        """
        Score whether *answer* directly addresses *question*.

        Args:
            question: The original question.
            answer: The RAG-generated answer.

        Returns:
            :class:`JudgementScore` on the 1–5 relevance rubric.
        """
        prompt = _PROMPTS[JudgeCriteria.RELEVANCE].format(
            question=question, answer=answer
        )
        return _parse_score(self._invoke(prompt), JudgeCriteria.RELEVANCE, question, answer)

    def judge_correctness(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> JudgementScore:
        """
        Score factual agreement between *answer* and the *ground_truth*.

        Args:
            question: The original question.
            answer: The RAG-generated answer.
            ground_truth: Reference answer to compare against.

        Returns:
            :class:`JudgementScore` on the 1–5 correctness rubric.
        """
        prompt = _PROMPTS[JudgeCriteria.CORRECTNESS].format(
            question=question, answer=answer, ground_truth=ground_truth
        )
        return _parse_score(self._invoke(prompt), JudgeCriteria.CORRECTNESS, question, answer)

    def judge_coherence(self, question: str, answer: str) -> JudgementScore:
        """
        Score the grammatical quality and logical structure of *answer*.

        Args:
            question: The original question (provides context for the judge).
            answer: The RAG-generated answer.

        Returns:
            :class:`JudgementScore` on the 1–5 coherence rubric.
        """
        prompt = _PROMPTS[JudgeCriteria.COHERENCE].format(
            question=question, answer=answer
        )
        return _parse_score(self._invoke(prompt), JudgeCriteria.COHERENCE, question, answer)

    def judge_all(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
    ) -> dict[JudgeCriteria, JudgementScore]:
        """
        Run all four criteria in sequence and return a criteria-keyed dict.

        Args:
            question: The original question.
            answer: The RAG-generated answer.
            context: Retrieved context string.
            ground_truth: Reference answer.

        Returns:
            Dict mapping each :class:`JudgeCriteria` to its
            :class:`JudgementScore`.
        """
        return {
            JudgeCriteria.FAITHFULNESS: self.judge_faithfulness(question, answer, context),
            JudgeCriteria.RELEVANCE: self.judge_relevance(question, answer),
            JudgeCriteria.CORRECTNESS: self.judge_correctness(question, answer, ground_truth),
            JudgeCriteria.COHERENCE: self.judge_coherence(question, answer),
        }


# ------------------------------------------------------------------ #
#  Batch evaluator                                                     #
# ------------------------------------------------------------------ #


class LLMJudgeEvaluator:
    """
    Runs :class:`LLMJudge` over an entire :class:`~evaluation.dataset.EvalDataset`.

    Collects per-sample scores into a tidy DataFrame with one row per sample
    and four score columns plus four explanation columns.

    Example::

        evaluator = LLMJudgeEvaluator(config)
        df = evaluator.run(dataset, answers, contexts)
        evaluator.save_results(df, "phi3_judge.csv")
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """
        Args:
            config: Evaluation configuration used to instantiate the judge.
        """
        self.config = config
        self.judge = LLMJudge(config)

    def run(
        self,
        dataset: EvalDataset,
        answers: list[str],
        contexts: list[list[str]],
    ) -> pd.DataFrame:
        """
        Score every sample and return a results DataFrame.

        A ``MEAN`` row is appended at the bottom for quick aggregate inspection.

        Args:
            dataset: Evaluation dataset supplying questions and ground truths.
            answers: Generated answers, one per sample.
            contexts: Retrieved context lists per sample.

        Returns:
            DataFrame with columns: ``question``, ``answer``,
            ``faithfulness``, ``faithfulness_explanation``, ``relevance``,
            ``relevance_explanation``, ``correctness``,
            ``correctness_explanation``, ``coherence``,
            ``coherence_explanation``, plus a trailing ``MEAN`` row.
        """
        rows = []
        for i, (sample, answer, ctx_list) in enumerate(
            zip(dataset.samples, answers, contexts), 1
        ):
            print(f"  [{i}/{len(dataset)}] Judging: {sample.question[:60]}...")
            context_str = "\n\n".join(ctx_list)
            scores = self.judge.judge_all(
                question=sample.question,
                answer=answer,
                context=context_str,
                ground_truth=sample.ground_truth,
            )
            row: dict = {"question": sample.question, "answer": answer}
            for criteria, score in scores.items():
                row[criteria.value] = score.score
                row[f"{criteria.value}_explanation"] = score.explanation
            rows.append(row)

        df = pd.DataFrame(rows)
        score_cols = [c.value for c in JudgeCriteria]
        mean_values = df[score_cols].mean().to_dict()
        mean_values["question"] = "MEAN"
        df = pd.concat([df, pd.DataFrame([mean_values])], ignore_index=True)
        return df

    def save_results(
        self,
        df: pd.DataFrame,
        filename: str = "judge_results.csv",
    ) -> Path:
        """
        Write judge results to a CSV file in the configured output directory.

        Args:
            df: Results DataFrame from :meth:`run`.
            filename: Output filename inside ``config.output_dir``.

        Returns:
            Absolute path to the written file.
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        df.to_csv(path, index=False)
        print(f"Judge results saved → {path}")
        return path
