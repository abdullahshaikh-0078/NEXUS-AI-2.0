import { mockResponses } from "./mock-data.js";

const BASE_URL = "http://localhost:8000/api/v1";
const REQUEST_TIMEOUT_MS = 15000;

const API_CONFIG = {
  baseUrl: BASE_URL,
  useMock: true,
};

export async function queryRAG(query, options = {}) {
  const { onToken, timeoutMs = REQUEST_TIMEOUT_MS } = options;

  if (API_CONFIG.useMock) {
    return queryMockResponse(query, { onToken });
  }

  if (typeof onToken === "function") {
    return queryStreamingResponse(query, { onToken, timeoutMs });
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort("timeout"), timeoutMs);

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/query`, {
      method: "POST",
      headers: buildHeaders(),
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });

    const payload = await parseResponse(response);

    if (!response.ok) {
      throw new Error(payload?.detail || payload?.message || "The RAG service could not process your request.");
    }

    if (!payload || typeof payload.answer !== "string" || !payload.answer.trim()) {
      throw new Error("The RAG service returned an empty response.");
    }

    if (typeof onToken === "function") {
      onToken(payload.answer);
    }

    return normalizePayload(payload);
  } catch (error) {
    throw normalizeError(error);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export function getApiMode() {
  return API_CONFIG.useMock ? "mock" : "live";
}

export function setApiMode(mode) {
  API_CONFIG.useMock = mode !== "live";
}

export function toggleApiMode() {
  API_CONFIG.useMock = !API_CONFIG.useMock;
  return getApiMode();
}

async function queryStreamingResponse(query, options = {}) {
  const { onToken, timeoutMs } = options;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort("timeout"), timeoutMs);

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/query/stream`, {
      method: "POST",
      headers: buildHeaders(),
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const payload = await parseResponse(response);
      throw new Error(payload?.detail || payload?.message || "The streaming request failed.");
    }

    if (!response.body) {
      throw new Error("Streaming is not available in this browser session.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalPayload = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) {
          continue;
        }
        const event = JSON.parse(line);
        if (event.type === "token") {
          onToken(event.chunk || "");
        }
        if (event.type === "final") {
          finalPayload = event.response;
        }
      }
    }

    if (!finalPayload || typeof finalPayload.answer !== "string") {
      throw new Error("The RAG service returned an empty streamed response.");
    }

    return normalizePayload(finalPayload);
  } catch (error) {
    throw normalizeError(error);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function queryMockResponse(query, options = {}) {
  const { onToken } = options;
  const normalized = query.trim().toLowerCase();
  const payload = normalized.includes("hybrid") ? mockResponses.hybrid : mockResponses.default;

  if (typeof onToken === "function") {
    const chunks = chunkText(payload.answer, 48);
    for (const chunk of chunks) {
      await wait(45);
      onToken(chunk);
    }
  } else {
    await wait(700);
  }

  return normalizePayload(payload);
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();
  return text ? { message: text } : null;
}

function buildHeaders() {
  const headers = {
    "Content-Type": "application/json",
  };
  const apiKey = window.localStorage.getItem("rag-api-key");
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }
  return headers;
}

function normalizePayload(payload) {
  return {
    answer: payload.answer,
    citations: Array.isArray(payload.citations) ? payload.citations : [],
    latency_ms: Number.isFinite(payload.latency_ms) ? payload.latency_ms : 0,
    warnings: Array.isArray(payload.warnings) ? payload.warnings : [],
  };
}

function normalizeError(error) {
  if (error?.name === "AbortError" || error === "timeout") {
    return new Error("The request timed out. Please try again.");
  }

  if (error instanceof TypeError) {
    return new Error("The API is unreachable right now. Make sure the backend is running.");
  }

  return error;
}

function chunkText(value, chunkSize) {
  const chunks = [];
  for (let index = 0; index < value.length; index += chunkSize) {
    chunks.push(value.slice(index, index + chunkSize));
  }
  return chunks;
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}
