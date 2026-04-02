from __future__ import annotations

from rag_service.context.compression import compress_text, estimate_tokens
from rag_service.context.models import ContextBlock, ContextPackage
from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.reranking.models import RerankingResult
from rag_service.reranking.pipeline import rerank_candidates

logger = get_logger(__name__)


def build_context(
    query: str,
    settings: Settings,
    reranking: RerankingResult | None = None,
) -> ContextPackage:
    reranking_result = reranking or rerank_candidates(query, settings=settings)
    blocks: list[ContextBlock] = []
    omitted_chunk_ids: list[str] = []
    total_tokens = 0
    seen_document_ids: set[str] = set()
    seen_texts: set[str] = set()

    for hit in reranking_result.reranked_hits:
        if hit.rerank_score < settings.context.min_rerank_score:
            omitted_chunk_ids.append(hit.chunk_id)
            continue

        document_id = getattr(hit.metadata, "document_id", "")
        normalized_text = " ".join(hit.text.split()).lower()

        if settings.context.deduplicate_by_document and document_id in seen_document_ids:
            omitted_chunk_ids.append(hit.chunk_id)
            continue
        if settings.context.deduplicate_by_text and normalized_text in seen_texts:
            omitted_chunk_ids.append(hit.chunk_id)
            continue

        compressed_text = compress_text(
            query=reranking_result.retrieval.processed_query.rewritten_query,
            text=hit.text,
            token_limit=settings.context.per_chunk_token_limit,
            strategy=settings.context.compression_strategy,
        )
        token_count = estimate_tokens(compressed_text)
        if token_count == 0:
            omitted_chunk_ids.append(hit.chunk_id)
            continue
        if total_tokens + token_count > settings.context.max_context_tokens:
            omitted_chunk_ids.append(hit.chunk_id)
            continue
        if len(blocks) >= settings.context.max_chunks:
            omitted_chunk_ids.append(hit.chunk_id)
            continue

        block = ContextBlock(
            chunk_id=hit.chunk_id,
            title=getattr(hit.metadata, "title", ""),
            source_path=getattr(hit.metadata, "source_path", ""),
            section_title=getattr(hit.metadata, "section_title", ""),
            rerank_score=hit.rerank_score,
            token_count=token_count,
            compressed_text=compressed_text,
            original_text=hit.text,
        )
        blocks.append(block)
        total_tokens += token_count
        if document_id:
            seen_document_ids.add(document_id)
        seen_texts.add(normalized_text)

    context_text = _format_context(blocks, include_headers=settings.context.include_metadata_headers)

    logger.info(
        "context_built",
        selected_blocks=len(blocks),
        omitted_blocks=len(omitted_chunk_ids),
        total_tokens=total_tokens,
    )

    return ContextPackage(
        reranking=reranking_result,
        selected_blocks=blocks,
        total_tokens=total_tokens,
        omitted_chunk_ids=omitted_chunk_ids,
        context_text=context_text,
    )


def _format_context(blocks: list[ContextBlock], include_headers: bool) -> str:
    rendered: list[str] = []
    for index, block in enumerate(blocks, start=1):
        if include_headers:
            header = (
                f"[Context {index}] title={block.title}; "
                f"section={block.section_title}; source={block.source_path}"
            )
            rendered.append(header)
        rendered.append(block.compressed_text)
    return "\n\n".join(part for part in rendered if part).strip()
