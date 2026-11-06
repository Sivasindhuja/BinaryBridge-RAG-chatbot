"""
BinaryBridge RAGAS evaluation harness.

This script evaluates a student's RAG implementation (``RAG.py``) against the
shared golden question-answer dataset.  It intentionally knows nothing about
the retrieval strategy used by the student: the only contract is::

    ask_question(question: str) -> (answer: str, retrieved_documents: List[Document])

Students may use any LLM, embedding model, vector store, chunking strategy, or
retrieval/reranking approach as long as that interface is preserved.

The four RAGAS metrics used here are the *same* for every student so that
results remain comparable across branches.
"""

from __future__ import annotations

import inspect
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv

# Import the student's RAG pipeline.  Keep the failure around so ``main`` can
# print an actionable message instead of a bare traceback.
RAG_IMPORT_ERROR: Optional[BaseException] = None
try:
    from RAG import ask_question  # type: ignore
except Exception as _rag_import_exc:  # pragma: no cover - depends on student code
    ask_question = None  # type: ignore[assignment]
    RAG_IMPORT_ERROR = _rag_import_exc


CSV_FILENAME = "golden_question_answer_pairs.csv"
REQUIRED_COLUMNS = ("question", "answer")
REPORT_PREFIX = "evaluation_report_"
DEFAULT_REPORT_STUDENT = "Student"
ASK_DELAY_SECONDS = 3.0
DEFAULT_EVAL_MODEL = "models/gemini-2.0-flash"

# Canonical metric key, human-readable label, and possible RAGAS column names.
METRIC_SPECS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("faithfulness", "Faithfulness", ("faithfulness",)),
    ("answer_correctness", "Answer Correctness", ("answer_correctness",)),
    (
        "context_precision",
        "Context Precision",
        ("llm_context_precision_with_reference", "context_precision"),
    ),
    ("context_recall", "Context Recall", ("context_recall",)),
)


class EvaluationError(Exception):
    """Raised for configuration or dataset problems the student can fix."""


def redact_secrets(text: str) -> str:
    """Remove configured API-key values from a message before logging it."""
    if not text:
        return text
    for variable in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
    ):
        value = os.getenv(variable)
        if value and len(value) >= 4:
            text = text.replace(value, "[REDACTED]")
    return text


@dataclass
class QuestionRecord:
    """Everything the harness knows about one golden-dataset question."""

    number: int
    question: str
    ground_truth: str
    answer: Optional[str] = None
    contexts: List[str] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def generated(self) -> bool:
        """True when an answer was produced without a generation error."""
        return self.error is None

    @property
    def missing_metrics(self) -> List[str]:
        """Canonical metric keys that have no usable score for this question."""
        if not self.generated:
            return [key for key, _label, _cols in METRIC_SPECS]
        return [
            key
            for key, _label, _cols in METRIC_SPECS
            if self.metrics.get(key) is None
        ]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def load_golden_dataset(csv_path: str = CSV_FILENAME) -> pd.DataFrame:
    """Load and validate the golden dataset, returning it in file order."""
    if not os.path.isfile(csv_path):
        raise EvaluationError(
            f"Could not find '{csv_path}'. Run the script from the repository "
            "root (the folder that contains RAG.py and the Documents/ folder)."
        )

    try:
        dataframe = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise EvaluationError(
            f"'{csv_path}' is empty. The golden dataset must contain a header "
            "row with at least the columns 'question' and 'answer'."
        ) from exc
    except Exception as exc:
        raise EvaluationError(
            f"Could not read '{csv_path}': {type(exc).__name__}: {exc}"
        ) from exc

    if dataframe.empty:
        raise EvaluationError(
            f"'{csv_path}' contains headers but no data rows. Add the golden "
            "question-answer pairs before running the evaluation."
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise EvaluationError(
            f"'{csv_path}' is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}. "
            f"Found: {', '.join(str(c) for c in dataframe.columns)}."
        )

    # Detect blank values without silently dropping rows: dropping data would
    # make a partial evaluation look like a complete one.
    invalid_rows: List[int] = []
    for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
        question = row["question"]
        answer = row["answer"]
        if (
            pd.isna(question)
            or pd.isna(answer)
            or not str(question).strip()
            or not str(answer).strip()
        ):
            invalid_rows.append(position)

    if invalid_rows:
        shown = ", ".join(str(number) for number in invalid_rows[:10])
        if len(invalid_rows) > 10:
            shown += f", ... ({len(invalid_rows)} total)"
        raise EvaluationError(
            f"'{csv_path}' contains empty question or answer values at data "
            f"row(s): {shown}. Fix the dataset before evaluating; rows are not "
            "silently skipped because that would give a misleading score."
        )

    # Use a clean 0..n-1 index so question numbers are deterministic.
    return dataframe.reset_index(drop=True)


def ensure_api_key_configured() -> None:
    """Validate the Gemini key and expose the alias used by LangChain.

    LangChain's Google integration historically looked for ``GOOGLE_API_KEY``
    while many course instructions use ``GEMINI_API_KEY``.  Both are accepted;
    neither value is ever printed.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not gemini_key and not google_key:
        raise EvaluationError(
            "No Gemini API key was found. Create a .env file in the repository "
            "root containing:\n"
            "    GEMINI_API_KEY=your_key_here\n"
            "(GOOGLE_API_KEY is also accepted.) The key is required for the "
            "RAGAS judge model."
        )

    if gemini_key and not google_key:
        # Do not log or echo this value.
        os.environ["GOOGLE_API_KEY"] = gemini_key


def student_name_to_slug(raw_name: str) -> str:
    """Convert a display name into a safe, deterministic report filename slug."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", raw_name or "").strip()
    slug = re.sub(r"[\s_]+", "-", cleaned).strip("-")
    return slug or DEFAULT_REPORT_STUDENT


def prompt_for_student_name() -> str:
    """Prompt for the student's name, with a safe non-interactive fallback."""
    try:
        raw_name = input("Enter your First and Last Name (for the report): ")
    except (EOFError, KeyboardInterrupt):
        raw_name = ""
    return student_name_to_slug(raw_name)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def validate_ask_question_result(result: Any) -> Tuple[str, List[str]]:
    """Validate and normalise the value returned by ``ask_question``.

    Returns ``(answer, contexts)`` where contexts are plain strings extracted
    from ``page_content``.  String contexts are tolerated for students who
    adapt another framework's output, but a Document-like object is the
    documented interface.
    """
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError(
            "ask_question() must return a tuple of (answer, retrieved_documents); "
            f"got {type(result).__name__}."
        )

    answer, documents = result

    if not isinstance(answer, str):
        raise TypeError(
            "ask_question() must return a string as its first value; "
            f"got {type(answer).__name__}."
        )
    if not answer.strip():
        raise ValueError("ask_question() returned an empty answer string.")

    if not isinstance(documents, (list, tuple)):
        raise TypeError(
            "ask_question() must return a list/tuple of retrieved documents as "
            f"its second value; got {type(documents).__name__}."
        )

    contexts: List[str] = []
    for position, document in enumerate(documents, start=1):
        if isinstance(document, str):
            text = document
        else:
            if not hasattr(document, "page_content"):
                raise TypeError(
                    "Each retrieved document must expose a 'page_content' "
                    f"attribute (or be a string). Item {position} is "
                    f"{type(document).__name__}."
                )
            text = document.page_content

        if not isinstance(text, str):
            raise TypeError(
                "The page_content of retrieved documents must be a string. "
                f"Item {position} contains {type(text).__name__}."
            )

        text = text.strip()
        if text:
            contexts.append(text)

    return answer.strip(), contexts


def generate_answers(
    dataframe: pd.DataFrame,
    ask_question_fn: Callable[[str], Any],
    delay_seconds: float = ASK_DELAY_SECONDS,
) -> List[QuestionRecord]:
    """Run every question in dataset order, recording successes and failures."""
    records: List[QuestionRecord] = []
    total = len(dataframe)

    for number, (_, row) in enumerate(dataframe.iterrows(), start=1):
        question = str(row["question"]).strip()
        ground_truth = str(row["answer"]).strip()
        record = QuestionRecord(
            number=number, question=question, ground_truth=ground_truth
        )

        try:
            result = ask_question_fn(question)
            answer, contexts = validate_ask_question_result(result)
            record.answer = answer
            record.contexts = contexts
            if not contexts:
                record.warnings.append(
                    "No retrieved documents/context were returned. "
                    "Context-based metrics may be unavailable for this question."
                )
            print(f"  [Q{number}/{total}] answer generated")
        except Exception as exc:  # keep evaluating the remaining questions
            record.error = redact_secrets(f"{type(exc).__name__}: {exc}")
            print(f"  [Q{number}/{total}] FAILED: {record.error}")

        records.append(record)

        # Avoid hammering the student's LLM/vector store provider.
        if delay_seconds > 0 and number < total:
            time.sleep(delay_seconds)

    return records


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------


def _coerce_score(value: Any) -> Optional[float]:
    """Return a finite float score, or ``None`` for missing/NaN/non-numeric."""
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def _import_ragas_dependencies():
    """Import the optional, pinned evaluation stack with a helpful error."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    return {
        "Dataset": Dataset,
        "evaluate": evaluate,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "AnswerCorrectness": AnswerCorrectness,
        "Faithfulness": Faithfulness,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
        "LLMContextRecall": LLMContextRecall,
        "RunConfig": RunConfig,
        "ChatGoogleGenerativeAI": ChatGoogleGenerativeAI,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    }


def run_ragas_evaluation(
    records: Sequence[QuestionRecord],
) -> Tuple[Optional[List[Dict[str, Optional[float]]]], Optional[str], List[str]]:
    """Score generated answers with RAGAS.

    Returns ``(metric_rows, fatal_error, notes)``.

    ``metric_rows`` is aligned with ``records`` (one dict per generated
    question) when RAGAS ran; otherwise it is ``None``.  A fatal error means no
    metric results exist.  Notes describe non-fatal issues such as a metric
    column missing from the RAGAS output.
    """
    generated = [record for record in records if record.generated]
    if not generated:
        return None, "No question produced an answer, so RAGAS could not run.", []

    try:
        dependencies = _import_ragas_dependencies()
    except Exception as exc:
        return (
            None,
            redact_secrets(
                "Could not import the RAGAS evaluation dependencies: "
                f"{type(exc).__name__}: {exc}. Run "
                "'pip install -r requirements.txt' and check that your Python "
                "environment supports the pinned versions."
            ),
            [],
        )

    try:
        eval_llm = dependencies["ChatGoogleGenerativeAI"](
            model=os.getenv("EVAL_LLM_MODEL", DEFAULT_EVAL_MODEL)
        )
        eval_embeddings = dependencies["HuggingFaceEmbeddings"](
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        ragas_llm = dependencies["LangchainLLMWrapper"](eval_llm)
        ragas_emb = dependencies["LangchainEmbeddingsWrapper"](eval_embeddings)
        run_config = dependencies["RunConfig"](max_workers=1, timeout=180)

        dataset = dependencies["Dataset"].from_dict(
            {
                "user_input": [record.question for record in generated],
                "response": [record.answer for record in generated],
                "retrieved_contexts": [record.contexts for record in generated],
                "reference": [record.ground_truth for record in generated],
            }
        )

        metrics = [
            dependencies["Faithfulness"](llm=ragas_llm),
            dependencies["AnswerCorrectness"](llm=ragas_llm),
            dependencies["LLMContextPrecisionWithReference"](llm=ragas_llm),
            dependencies["LLMContextRecall"](llm=ragas_llm),
        ]

        evaluate_kwargs = {
            "metrics": metrics,
            "embeddings": ragas_emb,
            "run_config": run_config,
        }
        try:
            supports_raise_exceptions = (
                "raise_exceptions"
                in inspect.signature(dependencies["evaluate"]).parameters
            )
        except (TypeError, ValueError):
            supports_raise_exceptions = False
        if supports_raise_exceptions:
            # Keep a single bad metric from aborting the whole run; RAGAS
            # reports the affected value as NaN, which the report shows as N/A.
            evaluate_kwargs["raise_exceptions"] = False

        evaluation_results = dependencies["evaluate"](dataset, **evaluate_kwargs)
        results_df = evaluation_results.to_pandas()
    except Exception as exc:
        return (
            None,
            redact_secrets(
                "RAGAS evaluation failed: "
                f"{type(exc).__name__}: {exc}. No metric scores were produced. "
                "Check your API key/quota/network and re-run."
            ),
            [],
        )

    if len(results_df) != len(generated):
        return (
            None,
            "RAGAS returned "
            f"{len(results_df)} result row(s) for {len(generated)} question(s); "
            "refusing to attach scores to the wrong questions.",
            [],
        )

    notes: List[str] = []
    column_lookup: Dict[str, Optional[str]] = {}
    for key, label, candidates in METRIC_SPECS:
        column = next((name for name in candidates if name in results_df.columns), None)
        column_lookup[key] = column
        if column is None:
            notes.append(
                f"RAGAS output did not contain a column for {label}; "
                "scores for that metric are reported as N/A."
            )

    metric_rows: List[Dict[str, Optional[float]]] = []
    for position in range(len(generated)):
        row = results_df.iloc[position]
        scores: Dict[str, Optional[float]] = {}
        for key, _label, _candidates in METRIC_SPECS:
            column = column_lookup[key]
            scores[key] = _coerce_score(row[column]) if column else None
        metric_rows.append(scores)

    return metric_rows, None, notes


def attach_metric_results(
    records: Sequence[QuestionRecord],
    metric_rows: Optional[Sequence[Dict[str, Optional[float]]]],
) -> None:
    """Attach RAGAS scores to the generated records in deterministic order."""
    if metric_rows is None:
        return
    generated = [record for record in records if record.generated]
    for record, scores in zip(generated, metric_rows):
        record.metrics = dict(scores)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _md_cell(value: Any) -> str:
    """Escape a value for use inside a Markdown table cell."""
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")


def _format_score(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _average(values: Sequence[Optional[float]]) -> Tuple[Optional[float], int]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None, 0
    return sum(valid) / len(valid), len(valid)


def _evaluation_status(
    records: Sequence[QuestionRecord],
    fatal_error: Optional[str],
) -> Tuple[str, List[str]]:
    """Return (status text, reasons) for the evaluation summary."""
    total = len(records)
    generation_failures = [record for record in records if not record.generated]
    generated = [record for record in records if record.generated]

    reasons: List[str] = []
    if generation_failures:
        reasons.append(
            f"{len(generation_failures)} of {total} question(s) failed during "
            "answer generation"
        )

    if fatal_error:
        reasons.append("RAGAS produced no metric scores")

    missing_metric_questions = [
        record for record in generated if record.missing_metrics
    ]
    if missing_metric_questions:
        missing_values = sum(
            len(record.missing_metrics) for record in missing_metric_questions
        )
        reasons.append(
            f"{missing_values} metric value(s) were unavailable across "
            f"{len(missing_metric_questions)} question(s)"
        )

    if not reasons:
        return "COMPLETE", reasons
    return "INCOMPLETE", reasons


def build_report(
    student_name: str,
    records: Sequence[QuestionRecord],
    fatal_error: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
) -> str:
    """Build the complete Markdown report from actual result data."""
    notes = list(notes or [])
    total = len(records)
    successful = sum(1 for record in records if record.generated)
    failed = total - successful
    generated = [record for record in records if record.generated]

    status, reasons = _evaluation_status(records, fatal_error)
    if status == "COMPLETE":
        status_text = (
            f"COMPLETE — all {total} question(s) were evaluated and every "
            "metric value is available."
        )
    else:
        status_text = "INCOMPLETE — " + "; ".join(reasons) + "."

    complete_metric_questions = sum(
        1 for record in generated if not record.missing_metrics
    )

    lines: List[str] = [
        f"# RAG Evaluation Report: {student_name.replace('-', ' ')}",
        "",
        "## Evaluation Summary",
        "",
        f"- Total test cases: {total}",
        f"- Successful (answer generated): {successful}",
        f"- Failed (answer generation): {failed}",
        f"- Questions with complete metric results: {complete_metric_questions}",
        f"- Evaluation status: {status_text}",
        "",
        "## Aggregate Metrics",
        "",
        "Averages are calculated only from valid metric values. Missing values "
        "are reported as N/A and are never replaced with zero.",
        "",
        "| Metric | Score | Valid results |",
        "|---|---|---|",
    ]

    for key, label, _candidates in METRIC_SPECS:
        values = [record.metrics.get(key) for record in generated]
        average, valid_count = _average(values)
        lines.append(
            f"| {label} | {_format_score(average)} | {valid_count}/{total} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Results",
            "",
            "| # | Question | Faithfulness | Answer Correctness | Context Precision | Context Recall |",
            "|---|---|---|---|---|---|",
        ]
    )

    for record in records:
        if record.generated:
            scores = [
                _format_score(record.metrics.get(key))
                for key, _label, _candidates in METRIC_SPECS
            ]
        else:
            scores = ["N/A"] * len(METRIC_SPECS)
        lines.append(
            f"| {record.number} | {_md_cell(record.question)} | "
            + " | ".join(scores)
            + " |"
        )

    lines.extend(["", "## Failed Questions", ""])
    generation_failures = [record for record in records if not record.generated]
    if generation_failures:
        lines.extend(
            [
                "These questions could not be scored because the RAG pipeline "
                "raised an error or returned an invalid value:",
                "",
                "| # | Question | Error |",
                "|---|---|---|",
            ]
        )
        for record in generation_failures:
            lines.append(
                f"| {record.number} | {_md_cell(record.question)} | "
                f"{_md_cell(record.error)} |"
            )
    else:
        lines.append("None — every question produced an answer.")

    lines.extend(["", "### Missing metric values", ""])
    missing_questions = [
        record for record in generated if record.missing_metrics
    ]
    if missing_questions:
        lines.extend(
            [
                "RAGAS did not return a usable score for the following values. "
                "They are excluded from the aggregates rather than scored as zero.",
                "",
                "| # | Question | Missing metrics |",
                "|---|---|---|",
            ]
        )
        labels = {key: label for key, label, _candidates in METRIC_SPECS}
        for record in missing_questions:
            missing_labels = ", ".join(
                labels[key] for key in record.missing_metrics
            )
            lines.append(
                f"| {record.number} | {_md_cell(record.question)} | "
                f"{_md_cell(missing_labels)} |"
            )
    else:
        lines.append("None — all available metric values were scored.")

    warnings = [
        (record, warning)
        for record in records
        for warning in record.warnings
    ]
    if warnings:
        lines.extend(
            [
                "",
                "## Data Quality Warnings",
                "",
                "| # | Question | Warning |",
                "|---|---|---|",
            ]
        )
        for record, warning in warnings:
            lines.append(
                f"| {record.number} | {_md_cell(record.question)} | "
                f"{_md_cell(warning)} |"
            )

    if fatal_error or notes:
        lines.extend(["", "## Evaluation Errors and Notes", ""])
        if fatal_error:
            lines.append(f"- **Fatal:** {_md_cell(fatal_error)}")
        for note in notes:
            lines.append(f"- {_md_cell(note)}")

    lines.extend(
        [
            "",
            "## Student Summary",
            "",
            "*[Write a brief summary of how your RAG system performed, any areas "
            "where it struggled, and how your chunking/retrieval strategy "
            "impacted the results here.]*",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the full evaluation workflow and write the Markdown report."""
    load_dotenv()

    print("BinaryBridge - RAGAS Evaluation")
    print("Initializing the evaluation harness...")

    if RAG_IMPORT_ERROR is not None:
        print(
            "ERROR: Could not import ask_question() from RAG.py: "
            f"{type(RAG_IMPORT_ERROR).__name__}: "
            f"{redact_secrets(str(RAG_IMPORT_ERROR))}\n"
            "Fix the import/initialization errors in your RAG.py, then re-run.",
            file=sys.stderr,
        )
        return 1

    try:
        ensure_api_key_configured()
        dataframe = load_golden_dataset()
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    student_name = prompt_for_student_name()
    total = len(dataframe)
    print(f"Loaded {total} test cases from {CSV_FILENAME}.")
    print(f"Student: {student_name.replace('-', ' ')}")

    print("\nStep 1: Generating answers from your RAG system...")
    records = generate_answers(dataframe, ask_question)  # type: ignore[arg-type]

    print("\nStep 2: Scoring answers with RAGAS...")
    metric_rows, fatal_error, notes = run_ragas_evaluation(records)
    attach_metric_results(records, metric_rows)

    report = build_report(student_name, records, fatal_error, notes)
    report_filename = f"{REPORT_PREFIX}{student_name}.md"
    try:
        with open(report_filename, "w", encoding="utf-8") as report_file:
            report_file.write(report)
    except OSError as exc:
        print(
            f"ERROR: Could not write '{report_filename}': {exc}", file=sys.stderr
        )
        return 1

    successful = sum(1 for record in records if record.generated)
    failed = total - successful
    status, reasons = _evaluation_status(records, fatal_error)

    print(f"\nReport saved as '{report_filename}'.")
    print(f"Questions: {total} total | {successful} generated | {failed} failed")
    if status == "COMPLETE":
        print("Evaluation status: COMPLETE")
        print("Remember to fill in the 'Student Summary' before committing.")
        return 0

    print("Evaluation status: INCOMPLETE - " + "; ".join(reasons) + ".")
    print("The report records the missing values; review it before committing.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
