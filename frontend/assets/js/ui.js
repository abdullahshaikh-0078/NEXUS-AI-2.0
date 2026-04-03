import { renderMarkdown } from "./markdown.js";

function createElement(tagName, className, textContent) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (typeof textContent === "string") {
    element.textContent = textContent;
  }
  return element;
}

export function renderChatHistory(container, messages, highlightedCitationId, onCitationSelect) {
  container.innerHTML = "";

  messages.forEach((message) => {
    const article = createElement(
      "article",
      message.role === "user" ? "message message-user" : "message message-assistant",
    );

    const header = createElement("header", "message-header");
    header.append(
      createElement("span", "message-role", message.role === "user" ? "You" : "NEXUS"),
      createElement("time", "message-time", formatTime(message.createdAt)),
    );

    const body = createElement("div", "message-body", message.content);
    article.append(header, body);

    if (message.isStreaming) {
      article.append(createElement("div", "streaming-indicator", "Streaming ready..."));
    }

    if (message.role === "assistant" && typeof message.latencyMs === "number") {
      article.append(createElement("div", "message-latency", `Response time: ${message.latencyMs} ms`));
    }

    if (message.citations?.length) {
      const strip = createElement("div", "message-citations");
      message.citations.forEach((citation, index) => {
        const button = createElement("button", "citation-chip", `[${index + 1}] ${citation.source}`);
        button.type = "button";
        if (citation.id === highlightedCitationId) {
          button.classList.add("is-active");
        }
        button.addEventListener("click", () => onCitationSelect(citation.id));
        strip.append(button);
      });
      article.append(strip);
    }

    container.append(article);
  });
}

export function renderAnswer(container, answer, citations, highlightedCitationId, onCitationSelect, inlineContainer) {
  const hasAnswer = answer && answer.trim();
  container.classList.toggle("empty-state", !hasAnswer);
  container.innerHTML = hasAnswer ? renderMarkdown(answer) : "Your latest answer will appear here.";

  inlineContainer.innerHTML = "";
  citations.forEach((citation, index) => {
    const button = createElement("button", "inline-citation", `[${index + 1}]`);
    button.type = "button";
    if (citation.id === highlightedCitationId) {
      button.classList.add("is-active");
    }
    button.addEventListener("click", () => onCitationSelect(citation.id));
    inlineContainer.append(button);
  });
}

export function renderCitations(container, citations, highlightedCitationId, onCitationSelect) {
  container.innerHTML = "";

  if (!citations.length) {
    container.className = "card-list empty-state";
    container.textContent = "Citations will appear with assistant responses.";
    return;
  }

  container.className = "card-list";
  citations.forEach((citation, index) => {
    const button = createElement("button", "citation-card");
    button.type = "button";
    if (citation.id === highlightedCitationId) {
      button.classList.add("is-active");
    }

    const indexNode = createElement("div", "citation-index", `[${index + 1}]`);
    const body = createElement("div", "citation-body");
    const title = createElement("strong", "", citation.source);
    const text = createElement("p", "", citation.text);
    const meta = createElement("span", "citation-meta", `Score ${citation.score.toFixed(2)}`);

    body.append(title, text, meta);
    button.append(indexNode, body);
    button.addEventListener("click", () => onCitationSelect(citation.id));
    container.append(button);
  });
}

export function renderDebugList(container, citations, highlightedCitationId, onCitationSelect) {
  container.innerHTML = "";

  if (!citations.length) {
    container.className = "card-list empty-state";
    container.textContent = "Submit a query to inspect the retrieved evidence.";
    return;
  }

  container.className = "card-list";
  citations.forEach((citation) => {
    const button = createElement("button", "debug-card");
    button.type = "button";
    button.dataset.citationId = citation.id;
    if (citation.id === highlightedCitationId) {
      button.classList.add("is-active");
    }

    button.append(
      createElement("strong", "debug-source", citation.source),
      createElement("p", "debug-text", citation.text),
      createElement("span", "debug-score", `score ${citation.score.toFixed(2)}`),
    );

    button.addEventListener("click", () => onCitationSelect(citation.id));
    container.append(button);
  });
}

export function setResponseMetrics(element, latencyMs) {
  if (typeof latencyMs === "number") {
    element.classList.remove("empty-state");
    element.textContent = `Response time: ${latencyMs} ms`;
    return;
  }

  element.classList.add("empty-state");
  element.textContent = "Response time: --";
}

export function setError(element, message) {
  if (!message) {
    element.textContent = "";
    element.classList.add("is-hidden");
    return;
  }
  element.textContent = message;
  element.classList.remove("is-hidden");
}

export function setLoading(element, isLoading) {
  element.classList.toggle("is-hidden", !isLoading);
}

function formatTime(isoString) {
  return new Date(isoString).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
