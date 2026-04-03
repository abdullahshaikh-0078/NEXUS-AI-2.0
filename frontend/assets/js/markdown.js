function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInline(value) {
  let rendered = escapeHtml(value);
  rendered = rendered.replace(/`([^`]+)`/g, "<code>$1</code>");
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return rendered;
}

export function renderMarkdown(markdown) {
  if (!markdown.trim()) {
    return "";
  }

  const codeBlocks = [];
  let content = markdown.replace(/```([\w-]*)\n([\s\S]*?)```/g, (_, language, code) => {
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    const languageClass = language ? ` class="language-${escapeHtml(language)}"` : "";
    codeBlocks.push(`<pre><code${languageClass}>${escapeHtml(code.trim())}</code></pre>`);
    return token;
  });

  const lines = content.split(/\n/);
  const chunks = [];
  let paragraph = [];
  let listItems = [];

  function flushParagraph() {
    if (!paragraph.length) {
      return;
    }
    chunks.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!listItems.length) {
      return;
    }
    chunks.push(`<ul>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ul>`);
    listItems = [];
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const codeMatch = trimmed.match(/^__CODE_BLOCK_(\d+)__$/);
    if (codeMatch) {
      flushParagraph();
      flushList();
      chunks.push(codeBlocks[Number(codeMatch[1])]);
      continue;
    }

    if (trimmed.startsWith("- ")) {
      flushParagraph();
      listItems.push(trimmed.slice(2));
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  return chunks.join("");
}
