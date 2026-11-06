"""Offline tests for the BinaryBridge evaluation harness.

These tests never call a real LLM, embedding model, or RAGAS judge.  External
calls are mocked so the harness's validation, failure handling, and report
generation can be checked without spending API credits.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import pandas as pd
import pytest

import RAGAS_evaluation_script as harness


@dataclass
class FakeDocument:
    page_content: str


def make_dataframe(rows):
    return pd.DataFrame(rows).reset_index(drop=True)


def make_records(answers, metric_scores=None):
    """Build records and optionally attach per-question metric values."""
    records = []
    for index, (question, ground_truth) in enumerate(
        [
            ("Question one?", "Answer one."),
            ("Question two?", "Answer two."),
            ("Question three?", "Answer three."),
        ][: len(answers)],
        start=1,
    ):
        record = harness.QuestionRecord(
            number=index,
            question=question,
            ground_truth=ground_truth,
            answer=answers[index - 1],
            contexts=["context"],
        )
        if metric_scores is not None:
            record.metrics = metric_scores[index - 1]
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def test_real_dataset_has_expected_schema_and_every_row_is_loaded():
    dataframe = harness.load_golden_dataset()
    assert len(dataframe) == 68
    assert list(dataframe.columns) == ["question", "answer"]
    assert dataframe["question"].is_unique
    assert dataframe["question"].notna().all()
    assert dataframe["answer"].notna().all()


def test_missing_dataset_has_actionable_error(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(harness.EvaluationError, match="Could not find"):
        harness.load_golden_dataset(str(missing))


def test_missing_columns_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"query": ["q"], "response": ["a"]}).to_csv(path, index=False)
    with pytest.raises(harness.EvaluationError, match="missing required column"):
        harness.load_golden_dataset(str(path))


def test_empty_dataset_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"question": [], "answer": []}).to_csv(path, index=False)
    with pytest.raises(harness.EvaluationError, match="no data rows"):
        harness.load_golden_dataset(str(path))


def test_blank_values_are_rejected_not_silently_dropped(tmp_path):
    path = tmp_path / "blank.csv"
    pd.DataFrame(
        {"question": ["q1", "  "], "answer": ["a1", "a2"]}
    ).to_csv(path, index=False)
    with pytest.raises(harness.EvaluationError, match="empty question or answer"):
        harness.load_golden_dataset(str(path))


# ---------------------------------------------------------------------------
# Interface validation
# ---------------------------------------------------------------------------


def test_valid_ask_question_result_accepts_documents_and_strings():
    answer, contexts = harness.validate_ask_question_result(
        ("answer", [FakeDocument("doc one"), "plain text", FakeDocument("   ")])
    )
    assert answer == "answer"
    assert contexts == ["doc one", "plain text"]


@pytest.mark.parametrize(
    "bad_result, message",
    [
        (None, "must return a tuple"),
        (("answer",), "must return a tuple"),
        ((123, []), "must return a string"),
        (("answer", "not-a-list"), "must return a list/tuple"),
        (("answer", [object()]), "page_content"),
        (("answer", [FakeDocument(123)]), "must be a string"),
        (("   ", []), "empty answer"),
    ],
)
def test_invalid_ask_question_results_are_rejected(bad_result, message):
    with pytest.raises((TypeError, ValueError), match=message):
        harness.validate_ask_question_result(bad_result)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def test_generate_answers_evaluates_every_question_in_order():
    dataframe = harness.load_golden_dataset()
    seen = []

    def fake_ask_question(question):
        seen.append(question)
        return "answer", [FakeDocument("context")]

    records = harness.generate_answers(
        dataframe, fake_ask_question, delay_seconds=0
    )

    assert len(records) == len(dataframe) == 68
    assert seen == dataframe["question"].tolist()
    assert [record.number for record in records] == list(range(1, 69))
    assert all(record.generated for record in records)
    assert all(record.answer == "answer" for record in records)


def test_generate_answers_records_failures_and_continues():
    dataframe = make_dataframe(
        {
            "question": ["q1", "q2", "q3"],
            "answer": ["a1", "a2", "a3"],
        }
    )

    def fake_ask_question(question):
        if question == "q2":
            raise RuntimeError("boom")
        return "answer", [FakeDocument("context")]

    records = harness.generate_answers(
        dataframe, fake_ask_question, delay_seconds=0
    )

    assert len(records) == 3
    assert [record.generated for record in records] == [True, False, True]
    assert records[1].error == "RuntimeError: boom"
    assert records[2].answer == "answer"


def test_empty_context_is_recorded_as_a_warning():
    dataframe = make_dataframe({"question": ["q1"], "answer": ["a1"]})
    records = harness.generate_answers(
        dataframe, lambda _question: ("answer", []), delay_seconds=0
    )
    assert records[0].generated is True
    assert records[0].contexts == []
    assert records[0].warnings
    assert "Context-based metrics" in records[0].warnings[0]


# ---------------------------------------------------------------------------
# RAGAS result handling
# ---------------------------------------------------------------------------


def _fake_ragas_dependencies(evaluate_impl):
    class FakeDataset:
        @staticmethod
        def from_dict(mapping):
            return mapping

    class FakeWrapper:
        def __init__(self, *_args, **_kwargs):
            pass

    class FakeMetric:
        def __init__(self, **_kwargs):
            pass

    class FakeRunConfig:
        def __init__(self, **_kwargs):
            pass

    return {
        "Dataset": FakeDataset,
        "evaluate": evaluate_impl,
        "LangchainEmbeddingsWrapper": FakeWrapper,
        "LangchainLLMWrapper": FakeWrapper,
        "AnswerCorrectness": FakeMetric,
        "Faithfulness": FakeMetric,
        "LLMContextPrecisionWithReference": FakeMetric,
        "LLMContextRecall": FakeMetric,
        "RunConfig": FakeRunConfig,
        "ChatGoogleGenerativeAI": FakeWrapper,
        "HuggingFaceEmbeddings": FakeWrapper,
    }


def test_ragas_scores_are_mapped_from_supported_column_names(monkeypatch):
    records = make_records(["a", "b"])

    class FakeResults:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "faithfulness": [1.0, 0.5],
                    "answer_correctness": [0.8, 0.4],
                    # ragas 0.2.x uses this column name for context precision.
                    "llm_context_precision_with_reference": [0.6, 0.2],
                    "context_recall": [0.9, 0.3],
                }
            )

    def fake_evaluate(_dataset, **_kwargs):
        return FakeResults()

    monkeypatch.setattr(
        harness,
        "_import_ragas_dependencies",
        lambda: _fake_ragas_dependencies(fake_evaluate),
    )

    metric_rows, fatal_error, notes = harness.run_ragas_evaluation(records)

    assert fatal_error is None
    assert notes == []
    assert metric_rows[0]["faithfulness"] == 1.0
    assert metric_rows[1]["context_precision"] == 0.2
    assert len(metric_rows) == 2


def test_ragas_failure_is_reported_without_losing_generation_results(monkeypatch):
    records = make_records(["a"])
    records_after = records

    def failing_evaluate(_dataset, **_kwargs):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(
        harness,
        "_import_ragas_dependencies",
        lambda: _fake_ragas_dependencies(failing_evaluate),
    )

    metric_rows, fatal_error, notes = harness.run_ragas_evaluation(records)

    assert metric_rows is None
    assert "RAGAS evaluation failed" in fatal_error
    assert records_after[0].generated is True


def test_missing_metric_column_produces_a_note(monkeypatch):
    records = make_records(["a"])

    class FakeResults:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "faithfulness": [1.0],
                    "answer_correctness": [1.0],
                    "context_recall": [1.0],
                }
            )

    monkeypatch.setattr(
        harness,
        "_import_ragas_dependencies",
        lambda: _fake_ragas_dependencies(lambda *_a, **_k: FakeResults()),
    )

    metric_rows, fatal_error, notes = harness.run_ragas_evaluation(records)

    assert fatal_error is None
    assert metric_rows[0]["context_precision"] is None
    assert any("Context Precision" in note for note in notes)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def test_complete_report_uses_actual_average_of_valid_values():
    records = make_records(
        ["answer one", "answer two"],
        [
            {
                "faithfulness": 1.0,
                "answer_correctness": 0.5,
                "context_precision": 0.25,
                "context_recall": 1.0,
            },
            {
                "faithfulness": 0.0,
                "answer_correctness": 1.0,
                "context_precision": 0.75,
                "context_recall": 0.0,
            },
        ],
    )

    report = harness.build_report("John-Doe", records)

    assert "# RAG Evaluation Report: John Doe" in report
    assert "- Total test cases: 2" in report
    assert "- Successful (answer generated): 2" in report
    assert "- Failed (answer generation): 0" in report
    assert "- Evaluation status: COMPLETE" in report
    assert "| Faithfulness | 0.5000 | 2/2 |" in report
    assert "| Answer Correctness | 0.7500 | 2/2 |" in report
    assert "| Context Precision | 0.5000 | 2/2 |" in report
    assert "| Context Recall | 0.5000 | 2/2 |" in report


def test_incomplete_report_excludes_missing_values_and_lists_failures():
    records = make_records(["answer one", "answer two"])
    records[1].error = "ValueError: bad return value"
    records[1].answer = None
    records[0].metrics = {
        "faithfulness": 1.0,
        "answer_correctness": None,
        "context_precision": 0.5,
        "context_recall": 0.5,
    }

    report = harness.build_report("Jane-Doe", records)

    assert "- Successful (answer generated): 1" in report
    assert "- Failed (answer generation): 1" in report
    assert "- Evaluation status: INCOMPLETE" in report
    assert "| Answer Correctness | N/A | 0/2 |" in report
    assert "| Faithfulness | 1.0000 | 1/2 |" in report
    assert "ValueError: bad return value" in report
    assert "Answer Correctness" in report
    assert "0.0000" not in report


def test_report_remains_useful_when_ragas_fails_entirely():
    records = make_records(["answer one"])
    report = harness.build_report(
        "Student", records, fatal_error="RAGAS evaluation failed: timeout"
    )

    assert "- Evaluation status: INCOMPLETE" in report
    assert "RAGAS evaluation failed: timeout" in report
    assert "| Faithfulness | N/A | 0/1 |" in report
    assert "## Student Summary" in report


def test_report_escapes_table_breaking_characters():
    records = [
        harness.QuestionRecord(
            number=1,
            question="What is A | B?",
            ground_truth="A",
            answer="A",
            contexts=["A"],
            metrics={
                "faithfulness": 1.0,
                "answer_correctness": 1.0,
                "context_precision": 1.0,
                "context_recall": 1.0,
            },
        )
    ]
    report = harness.build_report("Student", records)
    assert r"What is A \| B?" in report


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def test_missing_api_key_has_actionable_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(harness.EvaluationError, match="No Gemini API key"):
        harness.ensure_api_key_configured()


def test_gemini_key_is_aliased_without_being_logged(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    harness.ensure_api_key_configured()
    import os

    assert os.environ["GOOGLE_API_KEY"] == "test-key"


def test_student_name_slug_removes_path_characters():
    assert harness.student_name_to_slug("  John   Doe/../X ") == "John-DoeX"
    assert harness.student_name_to_slug("") == "Student"
    assert harness.student_name_to_slug("../../etc/passwd") == "etcpasswd"


def test_error_messages_redact_configured_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    message = harness.redact_secrets("request failed for super-secret-key")
    assert "super-secret-key" not in message
    assert "[REDACTED]" in message

    dataframe = make_dataframe({"question": ["q1"], "answer": ["a1"]})

    def leaky_ask_question(_question):
        raise RuntimeError("bad key super-secret-key")

    records = harness.generate_answers(
        dataframe, leaky_ask_question, delay_seconds=0
    )
    assert "super-secret-key" not in records[0].error
    assert "[REDACTED]" in records[0].error


# ---------------------------------------------------------------------------
# End-to-end orchestration (all external calls mocked)
# ---------------------------------------------------------------------------


def test_main_writes_report_and_returns_complete(monkeypatch, tmp_path):
    dataframe = make_dataframe(
        {"question": ["q1", "q2"], "answer": ["a1", "a2"]}
    )
    records = make_records(["answer one", "answer two"])

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(harness, "load_golden_dataset", lambda: dataframe)
    monkeypatch.setattr(
        harness, "prompt_for_student_name", lambda: "Test-Student"
    )
    monkeypatch.setattr(
        harness,
        "generate_answers",
        lambda _dataframe, _ask_question: records,
    )
    monkeypatch.setattr(
        harness,
        "run_ragas_evaluation",
        lambda _records: (
            [
                {
                    "faithfulness": 1.0,
                    "answer_correctness": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                },
                {
                    "faithfulness": 1.0,
                    "answer_correctness": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                },
            ],
            None,
            [],
        ),
    )
    monkeypatch.chdir(tmp_path)

    exit_code = harness.main([])

    assert exit_code == 0
    report_path = tmp_path / "evaluation_report_Test-Student.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# RAG Evaluation Report: Test Student" in content
    assert "Evaluation status: COMPLETE" in content
    assert "| Faithfulness | 1.0000 | 2/2 |" in content


def test_main_returns_two_when_a_question_fails(monkeypatch, tmp_path):
    dataframe = make_dataframe(
        {"question": ["q1", "q2"], "answer": ["a1", "a2"]}
    )
    records = make_records(["answer one", "answer two"])
    records[1].error = "RuntimeError: broken pipeline"

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(harness, "load_golden_dataset", lambda: dataframe)
    monkeypatch.setattr(
        harness, "prompt_for_student_name", lambda: "Test-Student"
    )
    monkeypatch.setattr(
        harness,
        "generate_answers",
        lambda _dataframe, _ask_question: records,
    )
    monkeypatch.setattr(
        harness,
        "run_ragas_evaluation",
        lambda _records: (
            [
                {
                    "faithfulness": 1.0,
                    "answer_correctness": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                }
            ],
            None,
            [],
        ),
    )
    monkeypatch.chdir(tmp_path)

    exit_code = harness.main([])

    assert exit_code == 2
    report_path = tmp_path / "evaluation_report_Test-Student.md"
    content = report_path.read_text(encoding="utf-8")
    assert "- Successful (answer generated): 1" in content
    assert "- Failed (answer generation): 1" in content
    assert "Evaluation status: INCOMPLETE" in content
    assert "RuntimeError: broken pipeline" in content

