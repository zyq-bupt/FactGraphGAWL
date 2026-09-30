"""Align keyphrases to containing summary sentences."""

from __future__ import annotations

import re
from typing import Optional

from baselines.kp_llm.schemas import KeyphraseAlignment


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter with fallback to nltk if available."""
    text = text.strip()
    if not text:
        return []
    try:
        import nltk

        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except Exception:  # noqa: BLE001
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower())


def _token_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class KeyphraseAligner:
    """Map each keyphrase to its most relevant summary sentence."""

    def __init__(self, semantic_threshold: float = 0.25):
        self.semantic_threshold = semantic_threshold

    def align(self, summary: str, keyphrases: list[str]) -> list[KeyphraseAlignment]:
        sentences = split_sentences(summary)
        if not sentences:
            sentences = [summary] if summary.strip() else []

        results: list[KeyphraseAlignment] = []
        for kp in keyphrases:
            item = self._align_one(kp, sentences)
            results.append(item)
        return results

    def _align_one(self, keyphrase: str, sentences: list[str]) -> KeyphraseAlignment:
        if not sentences:
            return KeyphraseAlignment(
                keyphrase=keyphrase,
                summary_sentence="",
                sentence_index=-1,
                alignment_type="none",
                candidate_indices=[],
            )

        # 1) exact
        for i, sent in enumerate(sentences):
            if keyphrase in sent:
                return KeyphraseAlignment(
                    keyphrase=keyphrase,
                    summary_sentence=sent,
                    sentence_index=i,
                    alignment_type="exact",
                    candidate_indices=[i],
                )

        # 2) casefold / punctuation-insensitive
        kp_n = _norm(keyphrase)
        candidates = []
        for i, sent in enumerate(sentences):
            if kp_n and kp_n in _norm(sent):
                candidates.append(i)
        if candidates:
            i = candidates[0]
            return KeyphraseAlignment(
                keyphrase=keyphrase,
                summary_sentence=sentences[i],
                sentence_index=i,
                alignment_type="casefold",
                candidate_indices=candidates,
            )

        # 3) token overlap
        kp_toks = _token_set(keyphrase)
        best_i, best_score = 0, -1.0
        overlap_hits = []
        for i, sent in enumerate(sentences):
            st = _token_set(sent)
            if not kp_toks or not st:
                score = 0.0
            else:
                score = len(kp_toks & st) / len(kp_toks)
            if score > best_score:
                best_score, best_i = score, i
            if score > 0:
                overlap_hits.append(i)
        if best_score >= 0.5:
            return KeyphraseAlignment(
                keyphrase=keyphrase,
                summary_sentence=sentences[best_i],
                sentence_index=best_i,
                alignment_type="overlap",
                candidate_indices=overlap_hits or [best_i],
            )

        # 4) semantic fallback via simple Jaccard on character n-grams
        best_i, best_score = 0, -1.0
        for i, sent in enumerate(sentences):
            score = self._char_jaccard(keyphrase, sent)
            if score > best_score:
                best_score, best_i = score, i
        return KeyphraseAlignment(
            keyphrase=keyphrase,
            summary_sentence=sentences[best_i],
            sentence_index=best_i,
            alignment_type="semantic",
            candidate_indices=[best_i],
        )

    @staticmethod
    def _char_jaccard(a: str, b: str, n: int = 3) -> float:
        def grams(s: str) -> set[str]:
            s = _norm(s).replace(" ", "")
            if len(s) < n:
                return {s} if s else set()
            return {s[i : i + n] for i in range(len(s) - n + 1)}

        ga, gb = grams(a), grams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)
