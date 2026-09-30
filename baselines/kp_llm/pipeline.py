"""End-to-end KP-LLM scorer."""

from __future__ import annotations

from typing import Any, Optional

from baselines.kp_llm.aggregator import aggregate_labels
from baselines.kp_llm.aligner import KeyphraseAligner, split_sentences
from baselines.kp_llm.checker import FactualityChecker
from baselines.kp_llm.extractor import KeyphraseExtractor
from baselines.llm_client import LLMClient


class KPLLMScorer:
    """Keyphrase-guided LLM factual consistency evaluator."""

    def __init__(
        self,
        extractor: KeyphraseExtractor,
        aligner: KeyphraseAligner,
        checker: FactualityChecker,
        extractor_model_name: str,
        checker_model_name: str,
    ):
        self.extractor = extractor
        self.aligner = aligner
        self.checker = checker
        self.extractor_model_name = extractor_model_name
        self.checker_model_name = checker_model_name

    def score(
        self,
        source: str,
        summary: str,
        *,
        sample_id: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> dict[str, Any]:
        base = {
            "id": sample_id,
            "dataset": dataset,
            "method": "KP-LLM",
            "extractor_model": self.extractor_model_name,
            "checker_model": self.checker_model_name,
            "source_mode": "full",
            "fallback_type": None,
            "error": None,
            "keyphrase_results": [],
        }

        if not summary or not summary.strip():
            base.update(
                {
                    "score": None,
                    "num_keyphrases": 0,
                    "supported_count": 0,
                    "contradicted_count": 0,
                    "not_enough_information_count": 0,
                    "error": "empty_summary",
                    "error_type": "empty_summary",
                }
            )
            return base

        keyphrases = self.extractor.extract(summary)
        fallback_type = None
        if not keyphrases:
            # Fallback: treat each summary sentence as a fact unit.
            sents = split_sentences(summary)
            keyphrases = sents[:]
            fallback_type = "summary_sentence_units"
            alignments = [
                {
                    "keyphrase": s,
                    "summary_sentence": s,
                    "sentence_index": i,
                    "alignment_type": "fallback_sentence",
                    "candidate_indices": [i],
                }
                for i, s in enumerate(sents)
            ]
        else:
            alignments = self.aligner.align(summary, keyphrases)

        keyphrase_results = []
        labels = []
        source_mode = "full"
        num_semantic = 0
        for item in alignments:
            kp = item["keyphrase"]
            sent = item.get("summary_sentence") or ""
            alignment_type = item.get("alignment_type", "exact")
            if alignment_type == "semantic":
                num_semantic += 1
            check = self.checker.check(source, sent, kp)
            # track truncation mode from checker internals via length heuristic
            if len(source) > self.checker.max_source_chars:
                source_mode = "retrieved"
            label = check.get("label", "NOT_ENOUGH_INFORMATION")
            labels.append(label)
            keyphrase_results.append(
                {
                    "keyphrase": kp,
                    "summary_sentence": sent,
                    "alignment_type": alignment_type,
                    "label": label,
                    "evidence": check.get("evidence", ""),
                    "reason": check.get("reason", ""),
                    "invalid_evidence": check.get("invalid_evidence", False),
                    "error": check.get("error"),
                }
            )

        agg = aggregate_labels(labels)
        base.update(agg)
        base["keyphrase_results"] = keyphrase_results
        base["source_mode"] = source_mode
        base["fallback_type"] = fallback_type
        base["num_semantic_alignments"] = num_semantic
        base["num_fallbacks"] = 1 if fallback_type else 0
        return base


def build_default_kp_llm_scorer(
    client: LLMClient,
    extractor_model: str = "/root/autodl-fs/zyq/models/DisT5",
    max_keyphrases: int = 8,
) -> KPLLMScorer:
    extractor = KeyphraseExtractor(model_name=extractor_model, max_keyphrases=max_keyphrases)
    aligner = KeyphraseAligner()
    checker = FactualityChecker(client=client)
    return KPLLMScorer(
        extractor=extractor,
        aligner=aligner,
        checker=checker,
        extractor_model_name=extractor_model,
        checker_model_name=client.config.model,
    )
