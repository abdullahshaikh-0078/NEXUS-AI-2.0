from __future__ import annotations

from pathlib import Path

from rag_service.evaluation.dataset import bootstrap_dataset_from_chunks, load_evaluation_dataset
from rag_service.evaluation.metrics import faithfulness_score, hallucination_rate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_evaluation_metrics_compute_expected_scores() -> None:
    relevant = {"doc-a", "doc-c"}
    ranked = ["doc-b", "doc-c", "doc-a"]

    assert recall_at_k(relevant, ranked, 1) == 0.0
    assert recall_at_k(relevant, ranked, 2) == 0.5
    assert reciprocal_rank(relevant, ranked) == 0.5
    assert ndcg_at_k(relevant, ranked, 3) > 0.6


def test_faithfulness_and_hallucination_are_complements() -> None:
    answer = "Hybrid retrieval combines lexical and dense retrieval. It tracks metadata for each chunk."
    evidence = [
        "Combine lexical and dense retrieval for higher recall.",
        "Track source, section, and chunk identifiers for every segment.",
    ]

    faithfulness = faithfulness_score(answer, evidence)
    hallucination = hallucination_rate(answer, evidence)

    assert faithfulness is not None
    assert faithfulness >= 0.5
    assert hallucination == round(1.0 - faithfulness, 4)


def test_bootstrap_dataset_from_chunks_creates_document_level_queries(sample_document_dir: Path, tmp_path: Path) -> None:
    from rag_service.core.config import Settings
    from rag_service.ingestion.pipeline import ingest_directory

    settings = Settings(ingestion={"input_dir": str(sample_document_dir), "output_dir": str(tmp_path)})
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)

    dataset_path = tmp_path / "eval.jsonl"
    bootstrap_dataset_from_chunks(chunks_path, dataset_path, max_documents=2)
    dataset = load_evaluation_dataset(dataset_path)

    assert len(dataset) == 2
    assert all(sample.target == "document" for sample in dataset)
    assert all(sample.relevant_ids for sample in dataset)
