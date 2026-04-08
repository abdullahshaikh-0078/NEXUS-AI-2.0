import {
  clearStoredApiKey,
  getApiMode,
  getStoredApiKey,
  queryRAG,
  setStoredApiKey,
  toggleApiMode,
} from "../services/rag-api.js";
import {
  renderAnswer,
  renderChatHistory,
  renderCitations,
  renderDebugList,
  setError,
  setLoading,
  setResponseMetrics,
} from "../renderers/dom.js";
import {
  createMessage,
  createState,
  findStreamingAssistantIndex,
  getLatestAssistant,
} from "./state.js";

const state = createState();

const elements = {
  chatHistory: document.querySelector("#chat-history"),
  answerContent: document.querySelector("#answer-content"),
  answerMetrics: document.querySelector("#answer-metrics"),
  inlineCitations: document.querySelector("#inline-citations"),
  citationsList: document.querySelector("#citations-list"),
  debugList: document.querySelector("#debug-list"),
  debugPanel: document.querySelector("#debug-panel"),
  debugToggle: document.querySelector("#debug-toggle"),
  modeToggle: document.querySelector("#mode-toggle"),
  apiKeyInput: document.querySelector("#api-key-input"),
  apiKeySave: document.querySelector("#api-key-save"),
  apiKeyClear: document.querySelector("#api-key-clear"),
  composerForm: document.querySelector("#composer-form"),
  queryInput: document.querySelector("#query-input"),
  sendButton: document.querySelector("#send-button"),
  loadingRow: document.querySelector("#loading-row"),
  errorBanner: document.querySelector("#error-banner"),
  modePill: document.querySelector(".mode-pill"),
};

initialize();

function initialize() {
  elements.composerForm.addEventListener("submit", handleSubmit);
  elements.queryInput.addEventListener("keydown", handleComposerKeydown);
  elements.queryInput.addEventListener("input", syncComposerState);
  elements.debugToggle.addEventListener("click", toggleDebugMode);
  elements.modeToggle.addEventListener("click", handleModeToggle);
  elements.apiKeySave.addEventListener("click", saveApiKey);
  elements.apiKeyClear.addEventListener("click", clearApiKey);
  elements.apiKeyInput.value = getStoredApiKey();
  render();
}

async function handleSubmit(event) {
  event.preventDefault();

  const query = elements.queryInput.value.trim();
  if (!query || state.loading) {
    return;
  }

  state.error = null;
  state.highlightedCitationId = null;
  state.pendingScrollCitationId = null;
  state.loading = true;
  state.messages.push(createMessage({ role: "user", content: query, isStreaming: false }));
  state.messages.push(createMessage({ role: "assistant", content: "", citations: [], warnings: [], latencyMs: null, isStreaming: true }));
  elements.queryInput.value = "";
  render();

  try {
    const response = await queryRAG(query, {
      onToken(chunk) {
        appendToStreamingAnswer(chunk);
      },
    });

    finalizeStreamingAnswer(response);

    if (response.citations.length) {
      state.highlightedCitationId = response.citations[0].id;
    }
  } catch (error) {
    state.messages = state.messages.filter((message) => !message.isStreaming);
    state.error = error instanceof Error ? error.message : "Something went wrong.";
  } finally {
    state.loading = false;
    render();
  }
}

function appendToStreamingAnswer(chunk) {
  const assistantIndex = findStreamingAssistantIndex(state.messages);
  if (assistantIndex < 0) {
    return;
  }

  const currentMessage = state.messages[assistantIndex];
  state.messages[assistantIndex] = {
    ...currentMessage,
    content: `${currentMessage.content}${chunk}`,
  };
  render();
}

function finalizeStreamingAnswer(response) {
  const assistantIndex = findStreamingAssistantIndex(state.messages);
  if (assistantIndex < 0) {
    return;
  }

  state.messages[assistantIndex] = {
    ...state.messages[assistantIndex],
    content: response.answer,
    citations: response.citations,
    warnings: response.warnings,
    latencyMs: response.latency_ms,
    isStreaming: false,
  };
}

function handleComposerKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composerForm.requestSubmit();
  }
}

function syncComposerState() {
  elements.sendButton.disabled = state.loading || !elements.queryInput.value.trim();
}

function toggleDebugMode() {
  state.debugMode = !state.debugMode;
  render();
}

function handleModeToggle() {
  toggleApiMode();
  state.error = null;
  render();
}

function saveApiKey() {
  const savedKey = setStoredApiKey(elements.apiKeyInput.value);
  elements.apiKeyInput.value = savedKey;
  state.error = savedKey ? null : "Saved an empty key, so live requests will be sent without authentication.";
  render();
}

function clearApiKey() {
  clearStoredApiKey();
  elements.apiKeyInput.value = "";
  state.error = null;
  render();
}

function handleCitationSelect(citationId) {
  state.highlightedCitationId = citationId;
  state.pendingScrollCitationId = citationId;
  state.debugMode = true;
  render();
}

function render() {
  const latestAssistant = getLatestAssistant(state.messages);
  const citations = latestAssistant?.citations ?? [];

  renderChatHistory(elements.chatHistory, state.messages, state.highlightedCitationId, handleCitationSelect);
  renderAnswer(elements.answerContent, latestAssistant?.content ?? "", citations, state.highlightedCitationId, handleCitationSelect, elements.inlineCitations);
  renderCitations(elements.citationsList, citations, state.highlightedCitationId, handleCitationSelect);
  renderDebugList(elements.debugList, citations, state.highlightedCitationId, handleCitationSelect);
  setResponseMetrics(elements.answerMetrics, latestAssistant?.latencyMs, latestAssistant?.warnings ?? []);
  setError(elements.errorBanner, state.error);
  setLoading(elements.loadingRow, state.loading);

  elements.sendButton.disabled = state.loading || !elements.queryInput.value.trim();
  elements.queryInput.disabled = state.loading;
  elements.debugPanel.classList.toggle("is-hidden", !state.debugMode);
  elements.debugToggle.textContent = state.debugMode ? "Hide debug" : "Show debug";
  const apiKeySuffix = getStoredApiKey() ? " | API key saved" : "";
  elements.modePill.textContent = (getApiMode() === "mock" ? "Mock mode" : "Live mode") + apiKeySuffix;
  elements.modeToggle.textContent = getApiMode() === "mock" ? "Switch to live" : "Switch to mock";
  elements.chatHistory.parentElement.scrollTo({ top: elements.chatHistory.parentElement.scrollHeight, behavior: "smooth" });

  if (state.pendingScrollCitationId) {
    scrollToDebugCitation(state.pendingScrollCitationId);
    state.pendingScrollCitationId = null;
  }
}

function scrollToDebugCitation(citationId) {
  const target = elements.debugList.querySelector(`[data-citation-id="${citationId}"]`);
  if (!target) {
    return;
  }

  target.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
