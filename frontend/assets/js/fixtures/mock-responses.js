export const mockResponses = {
  default: {
    answer:
      "**Retrieval-Augmented Generation (RAG)** combines search and generation.\n\n- It retrieves relevant source passages before answering.\n- It injects those passages into the model context.\n- It improves grounding and reduces hallucinations.\n\nIn this website, every factual claim is paired with visible citations like [1] and [2].",
    citations: [
      {
        id: "doc-platform-1",
        text: "The ingestion layer extracts content from private corpora and prepares normalized text for indexing and retrieval.",
        source: "platform_overview.txt",
        score: 0.94,
      },
      {
        id: "doc-retrieval-2",
        text: "Combine lexical and dense retrieval for higher recall and use chunk identifiers for every segment.",
        source: "retrieval_notes.html",
        score: 0.91,
      },
    ],
    latency_ms: 142,
  },
  hybrid: {
    answer:
      "**Hybrid retrieval** improves recall because it blends sparse keyword matching with dense semantic search.\n\n```text\nBM25 -> keyword precision\nDense -> semantic recall\nRRF -> robust fusion\n```\n\nThat mix gives production RAG systems stronger retrieval coverage [1].",
    citations: [
      {
        id: "doc-hybrid-1",
        text: "Hybrid retrieval combines lexical and dense retrieval before reciprocal rank fusion to create a stronger candidate pool.",
        source: "retrieval_notes.html",
        score: 0.93,
      },
    ],
    latency_ms: 167,
  },
};
