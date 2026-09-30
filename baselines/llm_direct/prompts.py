"""Prompt templates for Direct-LLM baselines (Luo et al. 2023 / Wang et al. 2023)."""

ZS_PROMPT = """Decide if the following summary is consistent with the corresponding article. Note that consistency means all information in the summary is supported by the article.
Article: {article}
Summary: {summary}
Answer (yes or no):"""

STAR_PROMPT = """Score the following news summarization given the corresponding news with respect to consistency with one to five stars, where one star means "inconsistency" and five stars means "perfect consistency". Note that consistency measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
Article: {article}
Summary: {summary}
Stars:"""
