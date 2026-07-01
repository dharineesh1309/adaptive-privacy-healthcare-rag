const form = document.getElementById("query-form");
const queryInput = document.getElementById("query");
const roleInput = document.getElementById("role");
const modeInput = document.getElementById("mode");
const submitButton = form.querySelector("button[type='submit']");
const errorBox = document.getElementById("form-error");
const resultBox = document.getElementById("result");
const emptyState = document.getElementById("empty-state");
const runtime = document.querySelector(".runtime");

function setText(id, value) {
  document.getElementById(id).textContent = String(value);
}

function formatScore(value) {
  return Number(value).toFixed(2);
}

function clearElement(element) {
  while (element.firstChild) element.removeChild(element.firstChild);
}

function renderResult(data) {
  const decision = data.query_decision;
  const riskLevel = document.getElementById("risk-level");
  riskLevel.textContent = decision.level;
  riskLevel.dataset.level = decision.level;
  setText("risk-score", formatScore(decision.query_score));
  setText("role-score", formatScore(decision.role_score));
  setText("policy-mode", data.mode.replace("_", " "));
  setText("latency", `${Number(data.latency_ms).toFixed(2)} ms · ${data.dataset}`);

  const reasons = document.getElementById("reasons");
  clearElement(reasons);
  decision.reasons.forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    reasons.appendChild(item);
  });

  const contexts = document.getElementById("contexts");
  clearElement(contexts);
  data.policy_decisions.forEach((policy, index) => {
    const article = document.createElement("article");
    article.className = "context";
    const meta = document.createElement("div");
    meta.className = "context-meta";
    const identity = document.createElement("span");
    identity.textContent = `${policy.document_id} · ${policy.selected_tier} tier`;
    const score = document.createElement("span");
    score.textContent = `fused risk ${formatScore(policy.final_score)}`;
    const text = document.createElement("p");
    text.textContent = data.protected_contexts[index] || "";
    meta.append(identity, score);
    article.append(meta, text);
    contexts.appendChild(article);
  });
  setText("context-count", `${data.policy_decisions.length} retrieved`);
  setText("answer", data.answer);
  emptyState.hidden = true;
  resultBox.hidden = false;
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Health check failed");
    const data = await response.json();
    runtime.classList.add("is-ready");
    setText("runtime-label", `${data.documents} protected documents · ${data.dataset}`);
  } catch (error) {
    setText("runtime-label", "Local pipeline unavailable");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  submitButton.disabled = true;
  submitButton.firstElementChild.textContent = "Evaluating policy…";
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryInput.value,
        role: roleInput.value,
        mode: modeInput.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed");
    renderResult(data);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.firstElementChild.textContent = "Run privacy policy";
  }
});

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    roleInput.value = button.dataset.role;
    queryInput.focus();
  });
});

loadHealth();
