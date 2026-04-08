from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_service.evaluation.models import ExperimentArtifacts, ExperimentRunResult, QueryEvaluationResult, SystemBenchmark


def write_experiment_artifacts(result: ExperimentRunResult, output_dir: Path, *, generate_plot: bool = True) -> ExperimentArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_json = output_dir / "aggregate_metrics.json"
    summary_csv = output_dir / "benchmark_summary.csv"
    per_query_csv = output_dir / "per_query_metrics.csv"
    benchmark_table_md = output_dir / "benchmark_table.md"
    benchmark_plot_svg = output_dir / "benchmark_plot.svg"

    aggregate_json.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, result.benchmarks, result.primary_k)
    _write_per_query_csv(per_query_csv, result.query_results, result.primary_k)
    benchmark_table_md.write_text(result.benchmark_table, encoding="utf-8")

    plot_path: str | None = None
    if generate_plot:
        benchmark_plot_svg.write_text(_build_svg_plot(result.benchmarks, result.primary_k), encoding="utf-8")
        plot_path = str(benchmark_plot_svg)

    artifacts = ExperimentArtifacts(
        run_dir=str(output_dir),
        aggregate_json=str(aggregate_json),
        summary_csv=str(summary_csv),
        per_query_csv=str(per_query_csv),
        benchmark_table_md=str(benchmark_table_md),
        benchmark_plot_svg=plot_path,
    )
    return artifacts


def build_benchmark_table(benchmarks: list[SystemBenchmark], primary_k: int) -> str:
    headers = ["System", f"Recall@{primary_k}", "MRR", f"NDCG@{primary_k}", "Faithfulness", "Hallucination", "Latency"]
    divider = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(divider) + " |"]
    for benchmark in benchmarks:
        rows.append(
            "| " + " | ".join(
                [
                    benchmark.system,
                    _format_float(benchmark.recall_at_k.get(str(primary_k))),
                    _format_float(benchmark.mrr),
                    _format_float(benchmark.ndcg_at_k.get(str(primary_k))),
                    _format_float(benchmark.faithfulness),
                    _format_float(benchmark.hallucination_rate),
                    f"{benchmark.total_latency_ms:.2f} ms",
                ]
            ) + " |"
        )
    return "\n".join(rows) + "\n"


def create_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return base_dir / timestamp


def _write_summary_csv(path: Path, benchmarks: list[SystemBenchmark], primary_k: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "system",
            f"recall@{primary_k}",
            "mrr",
            f"ndcg@{primary_k}",
            "faithfulness",
            "hallucination_rate",
            "retrieval_latency_ms",
            "rerank_latency_ms",
            "llm_latency_ms",
            "total_latency_ms",
            "improvements_vs_baseline",
        ])
        for benchmark in benchmarks:
            writer.writerow([
                benchmark.system,
                benchmark.recall_at_k.get(str(primary_k), 0.0),
                benchmark.mrr,
                benchmark.ndcg_at_k.get(str(primary_k), 0.0),
                benchmark.faithfulness,
                benchmark.hallucination_rate,
                benchmark.retrieval_latency_ms,
                benchmark.rerank_latency_ms,
                benchmark.llm_latency_ms,
                benchmark.total_latency_ms,
                json.dumps(benchmark.improvements_vs_baseline, ensure_ascii=True, sort_keys=True),
            ])


def _write_per_query_csv(path: Path, results: list[QueryEvaluationResult], primary_k: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "query_id",
            "system",
            "query",
            f"recall@{primary_k}",
            "mrr",
            f"ndcg@{primary_k}",
            "faithfulness",
            "hallucination_rate",
            "retrieval_latency_ms",
            "rerank_latency_ms",
            "llm_latency_ms",
            "total_latency_ms",
            "retrieved_ids",
        ])
        for result in results:
            writer.writerow([
                result.query_id,
                result.system,
                result.query,
                result.recall_at_k.get(str(primary_k), 0.0),
                result.mrr,
                result.ndcg_at_k.get(str(primary_k), 0.0),
                result.faithfulness,
                result.hallucination_rate,
                result.retrieval_latency_ms,
                result.rerank_latency_ms,
                result.llm_latency_ms,
                result.total_latency_ms,
                json.dumps(result.retrieved_ids, ensure_ascii=True),
            ])


def _build_svg_plot(benchmarks: list[SystemBenchmark], primary_k: int) -> str:
    width = 920
    height = 360
    padding = 48
    chart_height = 200
    bar_width = 42
    group_gap = 26
    metric_gap = 18
    metrics = [
        (f"Recall@{primary_k}", lambda item: item.recall_at_k.get(str(primary_k), 0.0), 1.0),
        ("MRR", lambda item: item.mrr, 1.0),
        (f"NDCG@{primary_k}", lambda item: item.ndcg_at_k.get(str(primary_k), 0.0), 1.0),
        ("Faithfulness", lambda item: item.faithfulness or 0.0, 1.0),
    ]
    colors = ["#7ff0cf", "#7ca7ff", "#ffbf69"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<rect width="100%" height="100%" fill="#07111f" rx="18"/>')
    svg.append('<text x="48" y="34" fill="#f3f6fb" font-size="20" font-family="Segoe UI, sans-serif">RAG Benchmark Overview</text>')

    x = padding
    baseline_y = padding + chart_height
    for metric_name, value_fn, scale in metrics:
        svg.append(f'<text x="{x}" y="{baseline_y + 24}" fill="#9eb0c7" font-size="12" font-family="Segoe UI, sans-serif">{metric_name}</text>')
        for index, benchmark in enumerate(benchmarks):
            value = max(0.0, min(scale, value_fn(benchmark)))
            bar_height = value * (chart_height - 30)
            bar_x = x + index * (bar_width + metric_gap)
            bar_y = baseline_y - bar_height
            color = colors[index % len(colors)]
            svg.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" rx="10" fill="{color}"/>')
            svg.append(f'<text x="{bar_x + 3}" y="{bar_y - 8}" fill="#f3f6fb" font-size="11" font-family="Segoe UI, sans-serif">{value:.2f}</text>')
            svg.append(f'<text x="{bar_x}" y="{baseline_y + 42}" fill="#dfe7f2" font-size="11" font-family="Segoe UI, sans-serif">{benchmark.system}</text>')
        x += (len(benchmarks) * (bar_width + metric_gap)) + group_gap

    svg.append('</svg>')
    return "\n".join(svg)


def _format_float(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"
