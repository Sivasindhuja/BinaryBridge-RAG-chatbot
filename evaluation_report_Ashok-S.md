# RAG Evaluation Report: Ashok S

**Evaluation mode:** Lightweight (token-overlap metrics)

**Questions evaluated:** 5  |  **Time taken:** 22.0s

## Aggregate Metrics

| Metric | Score |
| --- | --- |
| Faithfulness       | 1.0000 |
| Answer Correctness | 0.3244 |
| Context Precision  | 1.0000 |
| Context Recall     | 0.9009 |

## Per-Question Results

| question | faithfulness | answer_correctness | context_precision | context_recall |
| --- | --- | --- | --- | --- |
| Who is PMKVY Short Term Training meant to benefit? | 1.0 | 0.1538 | 1.0 | 0.95 |
| What extra modules are taught in STT apart from NSQF training? | 1.0 | 0.3294 | 1.0 | 0.9474 |
| Is STT only for first-time learners? | 1.0 | 0.4615 | 1.0 | 0.9048 |
| How is STT implemented? | 1.0 | 0.5091 | 1.0 | 0.9333 |
| What is the age limit for STT eligibility? | 1.0 | 0.1681 | 1.0 | 0.7692 |

## Summary

The RAG pipeline uses semantic embeddings (sentence-transformers/all-MiniLM-L6-v2) for dense retrieval and markdown-header-aware chunking to keep each chunk topically focused. Evaluation uses lightweight token-overlap metrics (precision, recall, F1) which are computed locally with no API calls required.
