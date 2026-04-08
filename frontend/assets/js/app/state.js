export function createState() {
  return {
    messages: [
      createMessage({
        role: "assistant",
        content:
          "Ask a question to explore your RAG system. This website can use mocked data or the live `/api/v1/query` and `/api/v1/query/stream` endpoints.",
        citations: [],
        warnings: [],
        latencyMs: 0,
        isStreaming: false,
      }),
    ],
    loading: false,
    error: null,
    debugMode: true,
    highlightedCitationId: null,
    pendingScrollCitationId: null,
  };
}

export function createMessage(partial) {
  return {
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    warnings: [],
    ...partial,
  };
}

export function getLatestAssistant(messages) {
  return [...messages].reverse().find((message) => message.role === "assistant" && message.content.trim());
}

export function findStreamingAssistantIndex(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.isStreaming) {
      return index;
    }
  }
  return -1;
}
