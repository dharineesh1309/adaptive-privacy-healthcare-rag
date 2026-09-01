const form = document.getElementById("query-form");
const queryInput = document.getElementById("query");
const roleInput = document.getElementById("role");
const purposeInput = document.getElementById("purpose");
const principalInput = document.getElementById("principal");
const modeInput = document.getElementById("mode");
const submitButton = form.querySelector("button[type='submit']");
const compareButton = document.getElementById("compare-button");
const errorBox = document.getElementById("form-error");
const resultBox = document.getElementById("result");
const emptyState = document.getElementById("empty-state");
const runtime = document.querySelector(".runtime");
const comparisonBlock = document.getElementById("comparison-block");
const sessionToggle = document.getElementById("session-aware");
const sessionStatus = document.getElementById("session-status");
const resetSessionButton = document.getElementById("reset-session");
const resetLedgerButton = document.getElementById("reset-ledger");
const slowAttackButton = document.getElementById("slow-attack");
const ledgerAttackButton = document.getElementById("ledger-attack");
const roleVerifiedInput = document.getElementById("role-verified");
let sessionId = null;

function setText(id, value) {
  document.getElementById(id).textContent = String(value);
}

function formatScore(value) {
  return Number(value).toFixed(2);
}

function clearElement(element) {
  while (element.firstChild) element.removeChild(element.firstChild);
}

function appendReasons(id, values) {
  const list = document.getElementById(id);
  clearElement(list);
  values.forEach((value) => {
    const item = document.createElement("li");
    item.textContent = value;
    list.appendChild(item);
  });
}

function renderResult(data) {
  const decision = data.query_decision;
  const riskLevel = document.getElementById("risk-level");
  riskLevel.textContent = decision.level;
  riskLevel.dataset.level = decision.level;
  setText("risk-score", formatScore(decision.query_score));
  setText("role-score", formatScore(decision.role_score));
  const assurance = data.role_assurance || {};
  const assuranceLabel = assurance.downgraded
    ? `downgraded to ${assurance.effective_role}`
    : assurance.verified
      ? `verified ${assurance.effective_role}`
      : assurance.effective_role || "standard";
  setText("role-assurance", assuranceLabel);
  setText("purpose-label", data.purpose_decision?.purpose?.replaceAll("_", " ") || "stateless");
  setText("policy-mode", data.mode.replace("_", " "));
  setText("latency", `${Number(data.latency_ms).toFixed(2)} ms · ${data.dataset}`);
  appendReasons("reasons", decision.reasons);

  const contexts = document.getElementById("contexts");
  clearElement(contexts);
  data.policy_decisions.forEach((policy, index) => {
    const contextText = data.protected_contexts[index];
    if (typeof contextText !== "string") return;
    const article = document.createElement("article");
    article.className = "context";
    const meta = document.createElement("div");
    meta.className = "context-meta";
    const identity = document.createElement("span");
    identity.textContent = `${policy.document_id} · ${policy.selected_tier} tier`;
    const score = document.createElement("span");
    score.textContent = `fused risk ${formatScore(policy.final_score)}`;
    const text = document.createElement("p");
    text.textContent = contextText;
    meta.append(identity, score);
    article.append(meta, text);
    contexts.appendChild(article);
  });
  setText("context-count", `${data.protected_contexts.length} served / ${data.policy_decisions.length} retrieved`);
  setText("answer", data.answer);
  renderSession(data);
  renderLedger(data);
  renderSecurityEvents(data.security_events || []);
  emptyState.hidden = true;
  resultBox.hidden = false;
}

function renderSecurityEvents(events) {
  const block = document.getElementById("security-block");
  if (!events.length) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  setText("security-count", `${events.length} event${events.length === 1 ? "" : "s"}`);
  appendReasons("security-events", events.map((event) => (
    `${event.document_id}: removed ${event.removed_segments} segment(s) · ${event.categories.join(", ")}`
  )));
}

function renderSession(data) {
  const block = document.getElementById("session-block");
  if (!data.session || !data.session_decision) {
    block.hidden = true;
    return;
  }
  const session = data.session;
  const decision = data.session_decision;
  block.hidden = false;
  const action = document.getElementById("session-action");
  action.textContent = decision.action;
  action.dataset.action = decision.action;
  setText("session-turn", session.turn_count);
  setText("session-score", Number(session.cumulative_score).toFixed(2));
  setText("quasi-budget", session.remaining_budget.quasi_types);
  setText("sensitive-budget", session.remaining_budget.sensitive_types);
  sessionStatus.textContent = `Turn ${session.turn_count} · risk ${Number(session.cumulative_score).toFixed(2)}`;

  const meter = document.querySelector(".session-meter");
  const fill = document.getElementById("session-meter-fill");
  const percentage = Math.round(Number(session.cumulative_score) * 100);
  meter.setAttribute("aria-valuenow", String(percentage));
  fill.style.width = `${Math.min(100, percentage)}%`;
  fill.style.background = percentage >= 85 ? "var(--high)" : percentage >= 55 ? "var(--medium)" : "var(--low)";
  appendReasons("session-reasons", decision.reasons);

  const history = document.getElementById("session-history");
  clearElement(history);
  session.history.forEach((turn) => {
    const item = document.createElement("div");
    item.className = "history-turn";
    const label = document.createElement("strong");
    label.textContent = turn.action;
    const detail = document.createElement("span");
    detail.textContent = `T${turn.turn} · ${Number(turn.cumulative_score).toFixed(2)}`;
    item.append(label, detail);
    history.appendChild(item);
  });
}

function renderLedger(data) {
  const block = document.getElementById("ledger-block");
  if (!data.ledger || !data.ledger_decision || !data.purpose_decision) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  const purposeAction = document.getElementById("purpose-action");
  purposeAction.textContent = data.purpose_decision.allowed ? "purpose allowed" : "purpose denied";
  purposeAction.dataset.action = data.purpose_decision.allowed ? "allow" : "block";
  const ledgerAction = document.getElementById("ledger-action");
  ledgerAction.textContent = `ledger ${data.ledger_decision.action}`;
  ledgerAction.dataset.action = data.ledger_decision.action;
  setText("principal-ref", data.ledger.principal_ref);
  setText("release-count", data.ledger.release_count);
  setText("exposure-units", data.ledger.exposure_units);
  setText("document-count", data.ledger.document_count);
  setText("ledger-quasi", data.ledger.remaining_budget.quasi_attributes);
  setText("ledger-sensitive", data.ledger.remaining_budget.sensitive_attributes);
  setText("ledger-documents", data.ledger.remaining_budget.documents);
  const types = data.ledger.exposed_types.length ? data.ledger.exposed_types.join(", ") : "none";
  setText("exposed-types", `Accounted attribute types: ${types}`);
  appendReasons("purpose-reasons", data.purpose_decision.reasons);
  appendReasons("ledger-reasons", data.ledger_decision.reasons);
}

function strictestTier(decisions) {
  const order = { low: 0, medium: 1, high: 2 };
  return decisions.reduce((strictest, decision) => (
    order[decision.selected_tier] > order[strictest] ? decision.selected_tier : strictest
  ), "low");
}

function renderComparison(results) {
  const grid = document.getElementById("comparison-grid");
  clearElement(grid);
  results.forEach((data) => {
    const card = document.createElement("article");
    card.className = "comparison-card";
    card.dataset.mode = data.mode;
    const title = document.createElement("div");
    title.className = "comparison-title";
    const mode = document.createElement("strong");
    mode.textContent = data.mode.replace("_", " ");
    const tier = document.createElement("span");
    tier.className = "tier-tag";
    tier.textContent = `${strictestTier(data.policy_decisions)} tier`;
    const answer = document.createElement("p");
    answer.textContent = data.answer;
    title.append(mode, tier);
    card.append(title, answer);
    grid.appendChild(card);
  });
  comparisonBlock.hidden = false;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function requestPolicy(mode, useSession = sessionToggle.checked, topK = null) {
  return postJson("/api/query", {
    query: queryInput.value,
    role: roleInput.value,
    purpose: purposeInput.value,
    principal_id: principalInput.value,
    mode,
    role_verified: roleVerifiedInput.checked && !roleVerifiedInput.disabled,
    session_aware: Boolean(useSession && sessionId),
    session_id: useSession ? sessionId : null,
    top_k: topK,
  });
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Health check failed");
    const data = await response.json();
    runtime.classList.add("is-ready");
    setText("runtime-label", `${data.documents} protected documents · purpose + ledger active`);
  } catch (error) {
    setText("runtime-label", "Local pipeline unavailable");
  }
}

async function createSession(action = "create") {
  const payload = action === "reset" && sessionId
    ? { action: "reset", session_id: sessionId, principal_id: principalInput.value }
    : { principal_id: principalInput.value };
  const data = await postJson("/api/session", payload);
  sessionId = data.session_id;
  sessionStatus.textContent = "Turn 0 · risk 0.00";
  document.getElementById("session-block").hidden = true;
  return data;
}

async function resetLedger() {
  const data = await postJson("/api/session", {
    action: "reset_ledger",
    principal_id: principalInput.value,
  });
  sessionStatus.textContent = "Principal ledger reset · session unchanged";
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  submitButton.disabled = true;
  submitButton.firstElementChild.textContent = "Evaluating policy...";
  try {
    const data = await requestPolicy(modeInput.value);
    comparisonBlock.hidden = true;
    renderResult(data);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.firstElementChild.textContent = "Run privacy policy";
  }
});

compareButton.addEventListener("click", async () => {
  errorBox.hidden = true;
  compareButton.disabled = true;
  compareButton.textContent = "Comparing...";
  try {
    const comparison = await postJson("/api/compare", {
      query: queryInput.value,
      role: roleInput.value,
      role_verified: roleVerifiedInput.checked && !roleVerifiedInput.disabled,
    });
    const results = comparison.results;
    const adaptive = results.find((item) => item.mode === "adaptive");
    renderResult(adaptive);
    renderComparison(results);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    compareButton.disabled = false;
    compareButton.textContent = "Compare all three";
  }
});

resetSessionButton.addEventListener("click", async () => {
  resetSessionButton.disabled = true;
  try {
    await createSession("reset");
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    resetSessionButton.disabled = false;
  }
});

resetLedgerButton.addEventListener("click", async () => {
  resetLedgerButton.disabled = true;
  try {
    await resetLedger();
    document.getElementById("ledger-block").hidden = true;
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    resetLedgerButton.disabled = false;
  }
});

slowAttackButton.addEventListener("click", async () => {
  const sequence = [
    "What age group is the patient with diabetes in?",
    "Which city is the patient with diabetes from?",
    "When was the patient with diabetes diagnosed?",
    "What diagnosis does the patient have?",
    "Give the exact phone number for that patient.",
  ];
  slowAttackButton.disabled = true;
  sessionToggle.checked = true;
  roleInput.value = "patient";
  purposeInput.value = "treatment";
  updateRoleAssuranceControl();
  modeInput.value = "adaptive";
  errorBox.hidden = true;
  try {
    await resetLedger();
    await createSession("reset");
    for (let index = 0; index < sequence.length; index += 1) {
      queryInput.value = sequence[index];
      slowAttackButton.textContent = `Running turn ${index + 1}/${sequence.length}...`;
      const latest = await requestPolicy("adaptive", true);
      renderResult(latest);
      if (latest.blocked) break;
    }
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    slowAttackButton.disabled = false;
    slowAttackButton.textContent = "Run slow extraction sequence";
  }
});

ledgerAttackButton.addEventListener("click", async () => {
  ledgerAttackButton.disabled = true;
  sessionToggle.checked = true;
  principalInput.value = "ledger-demo-user";
  roleInput.value = "doctor";
  purposeInput.value = "treatment";
  roleVerifiedInput.disabled = false;
  roleVerifiedInput.checked = true;
  modeInput.value = "fixed_low";
  errorBox.hidden = true;
  try {
    ledgerAttackButton.textContent = "Resetting principal ledger...";
    await resetLedger();
    await createSession("create");
    queryInput.value = "What age group is the patient with diabetes in?";
    ledgerAttackButton.textContent = "Session 1: first disclosure...";
    renderResult(await requestPolicy("fixed_low", true, 1));
    await createSession("create");
    queryInput.value = "What age group is the patient with migraine in?";
    ledgerAttackButton.textContent = "Session 2: reset attack...";
    renderResult(await requestPolicy("fixed_low", true, 1));
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    ledgerAttackButton.disabled = false;
    ledgerAttackButton.textContent = "Run cross-session reset attack";
  }
});

document.querySelectorAll(".sample[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    roleInput.value = button.dataset.role;
    purposeInput.value = button.dataset.purpose;
    updateRoleAssuranceControl();
    queryInput.focus();
  });
});

function updateRoleAssuranceControl() {
  const privileged = roleInput.value === "doctor" || roleInput.value === "admin";
  roleVerifiedInput.disabled = !privileged;
  if (!privileged) roleVerifiedInput.checked = false;
}

roleInput.addEventListener("change", updateRoleAssuranceControl);
principalInput.addEventListener("change", () => {
  if (!principalInput.checkValidity()) return;
  createSession("create").catch((error) => {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  });
});
updateRoleAssuranceControl();

Promise.all([loadHealth(), createSession()]).catch((error) => {
  errorBox.textContent = error.message;
  errorBox.hidden = false;
});
