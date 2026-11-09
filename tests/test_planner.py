from datetime import datetime
from agent.query_planner import HeuristicPlanner, LLMPlanner, normalize_file_types

NOW = datetime(2026, 11, 3, 12, 0)
hp = HeuristicPlanner()


def ts(y, m, d):
    return datetime(y, m, d).timestamp()


def test_last_month():
    p = hp.plan("Find that document from last month where I calculated the stress tensor for the gear shaft", NOW)
    assert p.filters.modified_after == ts(2026, 10, 1)
    assert "stress tensor" in p.semantic_query and "last" not in p.semantic_query


def test_january_rollover():
    p = hp.plan("report from last month", datetime(2026, 1, 15))
    assert p.filters.modified_after == ts(2025, 12, 1)


def test_exact_codes_and_types():
    p = hp.plan("the PDF quote that lists part XJ-4471-B", NOW)
    assert "XJ-4471-B" in p.exact_keywords and p.filters.file_types == [".pdf"]


def test_explicit_month_range_and_iso():
    p = hp.plan("spreadsheet from October 2026 about sales", NOW)
    assert p.filters.modified_after == ts(2026, 10, 1) and p.filters.modified_before > ts(2026, 10, 30)
    assert ".xlsx" in p.filters.file_types
    p = hp.plan('slides with "runway" since 2026-09-01', NOW)
    assert p.filters.modified_after == ts(2026, 9, 1) and p.exact_keywords == ["runway"]


def test_quotes_names_units():
    p = hp.plan('notes about Maria Chen and 450 MPa "yield point"', NOW)
    assert "yield point" in p.exact_keywords and "Maria Chen" in p.exact_keywords and "MPa" in p.exact_keywords


def test_no_filters_for_plain_query():
    p = hp.plan("sourdough starter feeding schedule", NOW)
    assert p.filters.is_empty() and p.semantic_query == "sourdough starter feeding schedule"


def test_ambiguous_word_not_a_type():
    p = hp.plan("the word that describes pump cavitation", NOW)
    assert p.filters.file_types == []


def test_llm_output_is_sanitised():
    class Bad:
        model = "x"
        def chat_json(self, s, u):
            return {"semantic_query": "", "exact_keywords": "ABC-1", "modified_after": "not-a-date",
                    "modified_before": "2026-10-31", "file_types": ["Word", "pdf"], "path_contains": 5}
    p = LLMPlanner(Bad()).plan("find my thing", NOW)
    assert p.semantic_query == "find my thing" and p.exact_keywords == ["ABC-1"]
    assert p.filters.modified_after is None and p.filters.modified_before
    assert p.filters.file_types == [".docx", ".pdf"] and p.filters.path_contains is None


def test_llm_failure_falls_back():
    class Boom:
        def chat_json(self, s, u):
            raise RuntimeError("down")
    p = LLMPlanner(Boom()).plan("pdf from last month about pumps", NOW)
    assert p.filters.file_types == [".pdf"] and "LLM planner failed" in p.notes


def test_normalize_types():
    assert normalize_file_types(["spreadsheet"]) == [".xlsx", ".xlsm", ".csv", ".tsv"]
    assert normalize_file_types([".PDF", "md"]) == [".pdf", ".md"]
