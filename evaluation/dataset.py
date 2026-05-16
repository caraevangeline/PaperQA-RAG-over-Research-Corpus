"""
Evaluation dataset management for the RAG pipeline.

Provides :class:`EvalDataset` — a lightweight wrapper around a list of
:class:`EvalSample` objects that handles JSON I/O and conversion to the
dict format expected by RAGAS.

Dataset JSON schema
-------------------
::

    {
        "samples": [
            {
                "question": "What is YOLO?",
                "ground_truth": "YOLO is ...",
                "expected_sources": ["yolo.pdf"]   // optional
            }
        ]
    }

Usage::

    dataset = EvalDataset.from_json("evaluation/sample_dataset.json")
    # or use the built-in smoke-test questions
    dataset = EvalDataset.sample_dataset()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    """
    A single question/answer pair used during evaluation.

    Attributes:
        question: Natural-language question to pose to the RAG system.
        ground_truth: Reference answer used for correctness and context-recall
            scoring.
        expected_sources: Optional list of PDF filenames that should appear
            in the retrieved documents (used for source accuracy audits).
    """

    question: str
    ground_truth: str
    expected_sources: list[str] = field(default_factory=list)


class EvalDataset:
    """
    Ordered collection of :class:`EvalSample` instances.

    Provides factory methods for common data sources and a conversion helper
    that produces the dict layout consumed by ``datasets.Dataset.from_dict``
    (and therefore by RAGAS).
    """

    def __init__(self, samples: list[EvalSample]) -> None:
        """
        Args:
            samples: Ordered list of evaluation samples.
        """
        self.samples = samples

    # ------------------------------------------------------------------ #
    #  Factory constructors                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, path: str) -> EvalDataset:
        """
        Load an evaluation dataset from a JSON file.

        Args:
            path: Filesystem path to the JSON file.

        Returns:
            Populated ``EvalDataset``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If a required field is missing from a sample entry.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        samples = [
            EvalSample(
                question=s["question"],
                ground_truth=s["ground_truth"],
                expected_sources=s.get("expected_sources", []),
            )
            for s in data["samples"]
        ]
        return cls(samples)

    @classmethod
    def sample_dataset(cls) -> EvalDataset:
        """
        Return a small built-in dataset for smoke-testing the pipeline.

        The questions cover common computer vision topics and do not assume
        any specific paper is present in the vector store.

        Returns:
            ``EvalDataset`` with seven CV research questions and reference
            answers.
        """
        samples = [
            EvalSample(
                question="What is the main contribution of the YOLO object detection framework?",
                ground_truth=(
                    "YOLO reframes object detection as a single regression problem, "
                    "predicting bounding boxes and class probabilities directly from full "
                    "images in one forward pass, enabling real-time detection speeds."
                ),
            ),
            EvalSample(
                question="How does the self-attention mechanism work in Vision Transformers?",
                ground_truth=(
                    "Vision Transformers split an image into fixed-size non-overlapping patches, "
                    "embed each patch as a token, then apply multi-head self-attention so every "
                    "token can attend to every other token, capturing global spatial context."
                ),
            ),
            EvalSample(
                question="What are the main trade-offs between speed and accuracy in object detection?",
                ground_truth=(
                    "Faster models use lightweight backbones and single-stage heads but miss small "
                    "or occluded objects; accurate models use larger two-stage pipelines with "
                    "feature pyramids at the cost of higher latency."
                ),
            ),
            EvalSample(
                question="How do multi-object tracking algorithms handle occlusion?",
                ground_truth=(
                    "Tracking algorithms combine appearance re-identification embeddings with "
                    "Kalman filter motion prediction to estimate hidden positions, and use track "
                    "management grace periods before deleting lost tracks."
                ),
            ),
            EvalSample(
                question="What is a Feature Pyramid Network and what problem does it solve?",
                ground_truth=(
                    "FPN builds a top-down pathway with lateral connections that merges "
                    "high-resolution low-level features with semantically rich high-level "
                    "features, solving multi-scale detection by providing strong representations "
                    "at every scale."
                ),
            ),
            EvalSample(
                question="What roles do the classification and regression heads play in anchor-based detectors?",
                ground_truth=(
                    "The classification head predicts object category probabilities for each "
                    "anchor while the regression head predicts coordinate offsets to refine "
                    "the anchor to better fit the target bounding box."
                ),
            ),
            EvalSample(
                question="How does non-maximum suppression work in object detection?",
                ground_truth=(
                    "NMS iteratively selects the highest-scoring box, suppresses all remaining "
                    "boxes whose IoU with it exceeds a threshold, and repeats until no boxes "
                    "remain, eliminating duplicate detections of the same object."
                ),
            ),
        ]
        return cls(samples)

    # ------------------------------------------------------------------ #
    #  Format conversion                                                   #
    # ------------------------------------------------------------------ #

    def to_ragas_format(
        self,
        answers: list[str],
        contexts: list[list[str]],
    ) -> dict:
        """
        Build the dict required by ``datasets.Dataset.from_dict`` for RAGAS.

        Args:
            answers: Generated answers, one per sample in the same order.
            contexts: Retrieval results per sample — each element is a list
                of ``page_content`` strings from the retrieved documents.

        Returns:
            Dict with keys ``question``, ``answer``, ``contexts``, and
            ``ground_truth``.

        Raises:
            ValueError: If the lengths of *answers* or *contexts* do not
                match the number of samples.
        """
        n = len(self.samples)
        if len(answers) != n:
            raise ValueError(f"Expected {n} answers, got {len(answers)}")
        if len(contexts) != n:
            raise ValueError(f"Expected {n} context lists, got {len(contexts)}")

        return {
            "question": [s.question for s in self.samples],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [s.ground_truth for s in self.samples],
        }

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """
        Write the dataset to a JSON file in the standard schema.

        Args:
            path: Destination file path (created or overwritten).
        """
        data = {
            "samples": [
                {
                    "question": s.question,
                    "ground_truth": s.ground_truth,
                    "expected_sources": s.expected_sources,
                }
                for s in self.samples
            ]
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"EvalDataset({len(self.samples)} samples)"
