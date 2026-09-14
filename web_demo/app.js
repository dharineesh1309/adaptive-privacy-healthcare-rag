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
const themeToggle = document.getElementById("theme-toggle");
const queryCount = document.getElementById("query-count");
const labStatus = document.getElementById("lab-status");
const toast = document.getElementById("toast");

let sessionId = null;
let toastTimer = null;

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

function showToast(message) {
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 2600);
}

function activateView(name, moveFocus = false) {
  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    const active = panel.dataset.viewPanel === name;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  document.querySelectorAll(".nav-tab").forEach((button) => {
    const active = button.dataset.viewTarget === name;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (moveFocus) {
    const target = document.querySelector(`[data-view-panel="${name}"] h1`);
    if (target) {
      target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: true });
    }
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function activateResultPane(name) {
  document.querySelectorAll("[data-result-panel]").forEach((panel) => {
    const active = panel.dataset.resultPanel === name;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  document.querySelectorAll(".result-tab").forEach((button) => {
    const active = button.dataset.resultTarget === name;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
}

function updateQueryCount() {
  queryCount.textContent = `${queryInput.value.length} / 5000`;
}

function setLoading(loading, label = "Evaluating privacy policy...") {
  form.setAttribute("aria-busy", String(loading));
  submitButton.disabled = loading;
  compareButton.disabled = loading;
  submitButton.firstElementChild.textContent = loading ? label : "Run privacy-aware RAG";
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
  setText("policy-mode", data.mode.replaceAll("_", " "));
  setText("latency", `${Number(data.latency_ms).toFixed(1)} ms / ${data.dataset}`);
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
    identity.textContent = `${policy.document_id} / ${policy.selected_tier} tier`;
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
  renderPolicyList(data.policy_decisions);
  renderGeneration(data.generation || {});
  renderSession(data);
  renderLedger(data);
  renderSecurityEvents(data.security_events || []);
  emptyState.hidden = true;
  resultBox.hidden = false;
}

function renderPolicyList(decisions) {
  const list = document.getElementById("policy-list");
  clearElement(list);
  setText("policy-count", `${decisions.length} document${decisions.length === 1 ? "" : "s"}`);
  decisions.forEach((decision) => {
    const item = document.createElement("article");
    item.className = "policy-item";
    const identity = document.createElement("strong");
    identity.textContent = decision.document_id;
    const tier = document.createElement("span");
    tier.className = "policy-tier";
    tier.dataset.level = decision.selected_tier;
    tier.textContent = `${decision.selected_tier} tier`;
    const detail = document.createElement("small");
    const reasons = Array.isArray(decision.reasons) ? decision.reasons.join(" / ") : "Policy fusion applied";
    detail.textContent = `Score ${formatScore(decision.final_score)} / ${reasons}`;
    item.append(identity, tier, detail);
    list.appendChild(item);
  });
}

function renderSecurityEvents(events) {
  const block = document.getElementById("security-block");
  const clear = document.getElementById("security-clear");
  if (!events.length) {
    block.hidden = true;
    clear.hidden = false;
    return;
  }
  block.hidden = false;
  clear.hidden = true;
  setText("security-count", `${events.length} event${events.length === 1 ? "" : "s"}`);
  appendReasons("security-events", events.map((event) => (
    `${event.document_id}: removed ${event.removed_segments} segment(s) / ${event.categories.join(", ")}`
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
  sessionStatus.textContent = `Turn ${session.turn_count} / risk ${Number(session.cumulative_score).toFixed(2)}`;
  appendReasons("session-reasons", decision.reasons);

  const history = document.getElementById("session-history");
  clearElement(history);
  session.history.forEach((turn) => {
    const item = document.createElement("div");
    item.className = "history-turn";
    const label = document.createElement("strong");
    label.textContent = turn.action;
    const detail = document.createElement("span");
    detail.textContent = `Turn ${turn.turn} / ${Number(turn.cumulative_score).toFixed(2)}`;
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

function renderGeneration(generation) {
  const mode = generation.mode || "extractive";
  const badge = document.getElementById("generation-mode");
  const labels = {
    llm: "Groq LLM",
    llm_cache: "Groq cache",
    extractive_fallback: "local fallback",
    extractive_evaluation: "local comparison",
    blocked_before_generation: "blocked before LLM",
    no_context: "no generation",
    extractive: "local extractive",
  };
  badge.textContent = labels[mode] || mode.replaceAll("_", " ");
  badge.dataset.mode = mode;
  const details = [];
  if (generation.model && generation.provider === "groq") details.push(generation.model);
  if (Number.isFinite(Number(generation.calls_remaining))) {
    details.push(`${generation.calls_remaining}/${generation.daily_call_limit} LLM calls remaining today`);
  }
  if (generation.reason) details.push(generation.reason);
  setText("generation-status", details.join(" / ") || "No external provider call was made.");
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
    mode.textContent = data.mode.replaceAll("_", " ");
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
    const generation = data.generation || {};
    const generatorLabel = generation.configured
      ? `Groq ready / ${generation.calls_remaining} calls left`
      : "local fallback / Groq key missing";
    setText("runtime-label", `${data.documents} records / ${generatorLabel}`);
  } catch (error) {
    setText("runtime-label", "Local pipeline unavailable");
  }
}

async function createSession(action = "create", notify = false) {
  const payload = action === "reset" && sessionId
    ? { action: "reset", session_id: sessionId, principal_id: principalInput.value }
    : { principal_id: principalInput.value };
  const data = await postJson("/api/session", payload);
  sessionId = data.session_id;
  sessionStatus.textContent = "Turn 0 / risk 0.00";
  document.getElementById("session-block").hidden = true;
  if (notify) showToast(action === "reset" ? "Session memory reset." : "New session created.");
  return data;
}

async function resetLedger(notify = false) {
  const data = await postJson("/api/session", {
    action: "reset_ledger",
    principal_id: principalInput.value,
  });
  sessionStatus.textContent = "Principal ledger reset / session unchanged";
  if (notify) showToast("Principal ledger reset for this lab demo.");
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  setLoading(true);
  try {
    const data = await requestPolicy(modeInput.value);
    comparisonBlock.hidden = true;
    renderResult(data);
    activateResultPane("answer");
    activateView("results", true);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    setLoading(false);
  }
});

compareButton.addEventListener("click", async () => {
  errorBox.hidden = true;
  compareButton.disabled = true;
  compareButton.textContent = "Comparing policies...";
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
    activateResultPane("answer");
    activateView("results", true);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    compareButton.disabled = false;
    compareButton.textContent = "Compare policies";
  }
});

resetSessionButton.addEventListener("click", async () => {
  resetSessionButton.disabled = true;
  try {
    await createSession("reset", true);
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
    await resetLedger(true);
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
    labStatus.textContent = "Resetting session and principal ledger...";
    await resetLedger();
    await createSession("reset");
    let latest = null;
    for (let index = 0; index < sequence.length; index += 1) {
      queryInput.value = sequence[index];
      updateQueryCount();
      labStatus.textContent = `Running turn ${index + 1} of ${sequence.length}: ${sequence[index]}`;
      latest = await requestPolicy("adaptive", true);
      renderResult(latest);
      if (latest.blocked) break;
    }
    labStatus.textContent = latest?.blocked
      ? "Attack stopped. The cumulative privacy policy blocked further disclosure."
      : "Sequence complete. Inspect how the privacy budget changed.";
    activateResultPane("memory");
    activateView("results", true);
  } catch (error) {
    labStatus.textContent = `Attack could not complete: ${error.message}`;
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
    labStatus.textContent = "Resetting the controlled principal ledger...";
    await resetLedger();
    await createSession("create");
    queryInput.value = "What age group is the patient with diabetes in?";
    updateQueryCount();
    labStatus.textContent = "Session one: recording the first disclosure.";
    renderResult(await requestPolicy("fixed_low", true, 1));
    await createSession("create");
    queryInput.value = "What age group is the patient with migraine in?";
    updateQueryCount();
    labStatus.textContent = "Session two: attempting to evade the prior disclosure budget.";
    renderResult(await requestPolicy("fixed_low", true, 1));
    labStatus.textContent = "Reset attack complete. The principal ledger survived the new session.";
    activateResultPane("memory");
    activateView("results", true);
  } catch (error) {
    labStatus.textContent = `Attack could not complete: ${error.message}`;
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
    updateQueryCount();
    updateRoleAssuranceControl();
    activateView("query");
    queryInput.focus();
    showToast("Scenario loaded. Review the access context, then run it.");
  });
});

document.querySelectorAll("[data-view-target]").forEach((button) => {
  button.addEventListener("click", () => activateView(button.dataset.viewTarget));
});

document.querySelectorAll("[data-result-target]").forEach((button) => {
  button.addEventListener("click", () => activateResultPane(button.dataset.resultTarget));
});

function updateRoleAssuranceControl() {
  const privileged = roleInput.value === "doctor" || roleInput.value === "admin";
  roleVerifiedInput.disabled = !privileged;
  if (!privileged) roleVerifiedInput.checked = false;
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeToggle.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  themeToggle.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
  try {
    window.localStorage.setItem("adaptive-privacy-theme", theme);
  } catch (error) {
    // Theme persistence is optional in privacy-restricted browser modes.
  }
}

function initializeTheme() {
  let saved = null;
  try {
    saved = window.localStorage.getItem("adaptive-privacy-theme");
  } catch (error) {
    saved = null;
  }
  const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(saved === "dark" || saved === "light" ? saved : preferredDark ? "dark" : "light");
}

themeToggle.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

roleInput.addEventListener("change", updateRoleAssuranceControl);
queryInput.addEventListener("input", updateQueryCount);
queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});

principalInput.addEventListener("change", () => {
  if (!principalInput.checkValidity()) return;
  createSession("create", true).catch((error) => {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  });
});

initializeTheme();
updateQueryCount();
updateRoleAssuranceControl();

Promise.all([loadHealth(), createSession()]).catch((error) => {
  errorBox.textContent = error.message;
  errorBox.hidden = false;
});
