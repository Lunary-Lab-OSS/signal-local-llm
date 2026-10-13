"""
Core ML Prompt Complexity Classifier (CPU, macOS Apple Silicon).

Lightweight wrapper around the converted NVIDIA prompt-task-and-complexity
classifier (DeBERTa-v3) Core ML model. Produces a task type and a
`prompt_complexity_score` (0-1) that the IntentRouter uses as a difficulty
signal for routing (analogous to RouteLLM's strong-win-rate).

Runtime deps: coremltools + transformers (tokenizer) + numpy. NO torch.
The model is produced offline by:
    models/nvidia_prompt-task-and-complexity-classifier-coreml/convert_to_coreml.py
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_SUBDIR = "nvidia_prompt-task-and-complexity-classifier-coreml"

# Head output order (matches conversion OUTPUT_NAMES / config target order).
_HEAD_ORDER = [
    "task_type",
    "creativity_scope",
    "reasoning",
    "contextual_knowledge",
    "number_of_few_shots",
    "domain_knowledge",
    "no_label_reason",
    "constraint_ct",
]


def _softmax(x, axis=-1):
    # Max-subtracted softmax is stable for finite inputs; NaN/inf rows are
    # rejected here so invalid logits can never become routing decisions (R07).
    if not np.isfinite(x).all():
        raise ValueError("classifier produced non-finite logits")
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CoreMLComplexityClassifier:
    """Tokenize -> Core ML neural core -> numpy post-processing -> scores."""

    def __init__(self, model_dir: Path, seq_len: int = 128, compute_units: str = "cpu_only"):
        """
        Args:
            model_dir: Path to the *-coreml model directory (contains the
                .mlpackage, config.json and tokenizer files).
            seq_len: Static sequence length variant to load (128 = fast routing,
                512 = full fidelity). Falls back to the 512 model if the
                requested variant is missing.
            compute_units: "cpu_only" (default), "cpu_and_gpu", "cpu_and_ne", or "all".
        """
        import coremltools as ct  # local import: heavy, macOS-only
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.seq_len = seq_len

        with open(self.model_dir / "config.json") as f:
            cfg = json.load(f)
        self._task_type_map = cfg["task_type_map"]
        self._weights_map = cfg["weights_map"]
        self._divisor_map = cfg["divisor_map"]

        mlpackage = self.model_dir / (
            "model.mlpackage" if seq_len == 512 else f"model_seq{seq_len}.mlpackage"
        )
        if not mlpackage.exists():
            fallback = self.model_dir / "model.mlpackage"
            if fallback.exists():
                logger.warning(
                    "Core ML variant %s not found; falling back to 512-token model.",
                    mlpackage.name,
                )
                mlpackage = fallback
                self.seq_len = 512
            else:
                raise FileNotFoundError(
                    f"No Core ML package found in {self.model_dir}. Run convert_to_coreml.py first."
                )

        cu_map = {
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
            "all": ct.ComputeUnit.ALL,
        }
        compute_unit = cu_map.get(compute_units.lower(), ct.ComputeUnit.CPU_ONLY)

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = ct.models.MLModel(str(mlpackage), compute_units=compute_unit)
        logger.info(
            "✅ Core ML complexity classifier loaded (seq_len=%d, compute_units=%s)",
            self.seq_len,
            self._model.compute_unit,
        )

    def _compute(self, preds, target, decimal=4):
        if target == "task_type":
            probs = _softmax(preds, axis=1)
            top2_idx = np.argsort(-preds, axis=1)[:, :2]
            top2_prob = np.take_along_axis(probs, top2_idx, axis=1)
            strings = [[self._task_type_map[str(int(i))] for i in row] for row in top2_idx]
            prob_rounded = [[round(float(v), 3) for v in row] for row in top2_prob]
            for i, row in enumerate(prob_rounded):
                if row[1] < 0.1:
                    strings[i][1] = "NA"
            return ([r[0] for r in strings], [r[1] for r in strings], [r[0] for r in prob_rounded])

        probs = _softmax(preds, axis=1)
        weights = np.array(self._weights_map[target])
        scores = np.sum(probs * weights, axis=1) / self._divisor_map[target]
        scores = [round(float(v), decimal) for v in scores]
        if target == "number_of_few_shots":
            scores = [x if x >= 0.05 else 0 for x in scores]
        return scores

    def classify(self, text: str) -> dict:
        enc = self._tokenizer(
            [text],
            return_tensors="np",
            add_special_tokens=True,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
        )
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int32),
            "attention_mask": enc["attention_mask"].astype(np.int32),
        }
        out = self._model.predict(feeds)
        logits = {name: np.asarray(out[name]) for name in _HEAD_ORDER}

        result = {}
        tt1, tt2, ttp = self._compute(logits["task_type"], "task_type")
        result["task_type_1"], result["task_type_2"], result["task_type_prob"] = tt1, tt2, ttp
        for target in [
            "creativity_scope",
            "reasoning",
            "contextual_knowledge",
            "number_of_few_shots",
            "domain_knowledge",
            "no_label_reason",
            "constraint_ct",
        ]:
            result[target] = self._compute(logits[target], target)
        result["prompt_complexity_score"] = [
            round(0.35 * cr + 0.25 * re + 0.15 * co + 0.15 * dk + 0.05 * ck + 0.05 * fs, 5)
            for cr, re, co, dk, ck, fs in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
                result["number_of_few_shots"],
                strict=True,
            )
        ]
        return result

    def complexity_score(self, text: str) -> float:
        """Return a validated scalar complexity score in [0, 1] (R07)."""
        score = float(self.classify(text)["prompt_complexity_score"][0])
        if not np.isfinite(score):
            raise ValueError(f"complexity score is not finite: {score!r}")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"complexity score outside [0, 1]: {score!r}")
        return score

    @staticmethod
    def default_model_dir(models_dir) -> Path:
        return Path(models_dir) / MODEL_SUBDIR
