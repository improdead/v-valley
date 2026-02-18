const checklistRoot = document.getElementById("setupChecklist");
const progressBar = document.getElementById("setupProgressBar");
const progressText = document.getElementById("setupProgressText");
const templateCountEl = document.getElementById("templateCount");
const apiStateEl = document.getElementById("apiState");
const heroArt = document.getElementById("heroArt");
const apiBaseInput = document.getElementById("apiBase");
const refreshSetupInfoBtn = document.getElementById("refreshSetupInfoBtn");
const applyLowCostPresetBtn = document.getElementById("applyLowCostPresetBtn");
const setupInfoOutput = document.getElementById("setupInfoOutput");
const lowCostPresetOutput = document.getElementById("lowCostPresetOutput");
const registerForm = document.getElementById("registerForm");
const registerOutput = document.getElementById("registerOutput");
const claimForm = document.getElementById("claimForm");
const claimOutput = document.getElementById("claimOutput");
const meForm = document.getElementById("meForm");
const meOutput = document.getElementById("meOutput");
const copyKeyBtn = document.getElementById("copyKeyBtn");
const meApiKeyInput = document.getElementById("meApiKey");
const joinTownForm = document.getElementById("joinTownForm");
const joinTownOutput = document.getElementById("joinTownOutput");
const joinTownIdInput = document.getElementById("joinTownId");
const leaveTownBtn = document.getElementById("leaveTownBtn");
const simStateBtn = document.getElementById("simStateBtn");
const simTickBtn = document.getElementById("simTickBtn");
const simTownIdInput = document.getElementById("simTownId");
const simTickStepsInput = document.getElementById("simTickSteps");
const simPlanningScopeInput = document.getElementById("simPlanningScope");
const simOutput = document.getElementById("simOutput");
const claimTokenInput = document.getElementById("claimToken");
const claimCodeInput = document.getElementById("claimCode");
const claimOwnerInput = document.getElementById("claimOwner");
const refreshTownsBtn = document.getElementById("refreshTownsBtn");
const townsOutput = document.getElementById("townsOutput");
const townCardsContainer = document.getElementById("townCardsContainer");
const liveAgentsEl = document.getElementById("liveAgents");
const autoJoinBtn = document.getElementById("autoJoinBtn");
const simControlModeInput = document.getElementById("simControlMode");
const refreshScenariosBtn = document.getElementById("refreshScenariosBtn");
const scenarioCardsContainer = document.getElementById("scenarioCardsContainer");
const scenarioServerList = document.getElementById("scenarioServerList");
const scenarioLeaderboardList = document.getElementById("scenarioLeaderboardList");
const scenariosOutput = document.getElementById("scenariosOutput");
const myScenarioQueueBtn = document.getElementById("myScenarioQueueBtn");
const myScenarioMatchBtn = document.getElementById("myScenarioMatchBtn");
const myScenarioForfeitBtn = document.getElementById("myScenarioForfeitBtn");
const myScenarioOutput = document.getElementById("myScenarioOutput");
const legacyListBtn = document.getElementById("legacyListBtn");
const legacyReplayBtn = document.getElementById("legacyReplayBtn");
const legacyImportBtn = document.getElementById("legacyImportBtn");
const legacySimulationIdInput = document.getElementById("legacySimulationId");
const legacyStartStepInput = document.getElementById("legacyStartStep");
const legacyMaxStepsInput = document.getElementById("legacyMaxSteps");
const legacyTownIdInput = document.getElementById("legacyTownId");
const legacyOutput = document.getElementById("legacyOutput");
const legacyImportOutput = document.getElementById("legacyImportOutput");

const CHECKLIST_KEY = "vvalley_setup_checklist_v1";
const API_BASE_KEY = "vvalley_api_base_v1";

function toPrettyJson(v) {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function setStatLoading(el, loading) {
  if (!el) return;
  if (loading) {
    el.classList.add("stat-loading");
    el.textContent = "";
  } else {
    el.classList.remove("stat-loading");
  }
}

function ratingTier(rating) {
  const value = Number(rating || 0);
  if (value >= 2100) return { label: "Diamond", cls: "tier-diamond", icon: "assets/scenarios/processed/shared/frames/badge_diamond.png" };
  if (value >= 1900) return { label: "Platinum", cls: "tier-platinum", icon: "assets/scenarios/processed/shared/frames/badge_platinum.png" };
  if (value >= 1700) return { label: "Gold", cls: "tier-gold", icon: "assets/scenarios/processed/shared/frames/badge_gold.png" };
  if (value >= 1500) return { label: "Silver", cls: "tier-silver", icon: "assets/scenarios/processed/shared/frames/badge_silver.png" };
  if (value >= 1300) return { label: "Bronze", cls: "tier-bronze", icon: "assets/scenarios/processed/shared/frames/badge_bronze.png" };
  return { label: "Starter", cls: "tier-starter", icon: "assets/scenarios/processed/shared/frames/badge_starter.png" };
}

function scenarioRules(scenario) {
  return scenario && typeof scenario.rules_json === "object" && scenario.rules_json ? scenario.rules_json : {};
}

function scenarioKindFromMeta(scenarioLike) {
  const rules = scenarioRules(scenarioLike);
  const configured = String(rules.engine || scenarioLike?.scenario_kind || "").trim().toLowerCase();
  if (configured) return configured;
  const key = String(scenarioLike?.scenario_key || "").toLowerCase();
  if (key.includes("werewolf")) return "werewolf";
  if (key.includes("blackjack")) return "blackjack";
  if (key.includes("holdem") || key.includes("hold'em")) return "holdem";
  if (key.includes("anaconda")) return "anaconda";
  return "scenario";
}

function scenarioUiGroupFromMeta(scenarioLike) {
  const rules = scenarioRules(scenarioLike);
  const configured = String(rules.ui_group || scenarioLike?.ui_group || "").trim().toLowerCase();
  if (configured) return configured;
  return scenarioKindFromMeta(scenarioLike) === "werewolf" ? "social" : "casino";
}

function scenarioLabelFromKind(kind) {
  const normalized = String(kind || "").toLowerCase();
  if (normalized === "werewolf") return "Werewolf";
  if (normalized === "anaconda") return "Anaconda";
  if (normalized === "blackjack") return "Blackjack";
  if (normalized === "holdem") return "Hold'em";
  return "Scenario";
}

function guessDefaultApiBase() {
  if (window.location.port === "8080") {
    return `${window.location.protocol}//${window.location.host}`;
  }
  return `${window.location.protocol}//${window.location.hostname}:8080`;
}

function normalizeBase(raw) {
  return (raw || "").trim().replace(/\/+$/, "");
}

function getApiBase() {
  const fromInput = normalizeBase(apiBaseInput?.value);
  if (fromInput) return fromInput;
  const fromStorage = normalizeBase(localStorage.getItem(API_BASE_KEY));
  if (fromStorage) return fromStorage;
  return normalizeBase(guessDefaultApiBase());
}

function persistApiBase() {
  const base = getApiBase();
  if (apiBaseInput) apiBaseInput.value = base;
  localStorage.setItem(API_BASE_KEY, base);
}

function preserveApiParamOnTownLinks() {
  const api = new URLSearchParams(window.location.search).get("api");
  if (!api) return;
  document.querySelectorAll('a[href*="town.html"]').forEach((link) => {
    const href = link.getAttribute("href");
    if (!href) return;
    try {
      const url = new URL(href, window.location.href);
      url.searchParams.set("api", api);
      link.setAttribute("href", `${url.pathname}${url.search}${url.hash}`);
    } catch {
      // Keep original href when parsing fails.
    }
  });
}

async function apiJson(path, opts = {}) {
  const base = getApiBase();
  const url = `${base}${path}`;
  const headers = { ...(opts.headers || {}) };
  const method = opts.method || "GET";
  const hasBody = opts.body !== undefined;
  let body = opts.body;

  if (hasBody && typeof body !== "string") {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }

  const resp = await fetch(url, { method, headers, body });
  let payload;
  try {
    payload = await resp.json();
  } catch {
    payload = { raw: await resp.text() };
  }
  if (!resp.ok) {
    const err = new Error(payload.detail || `HTTP ${resp.status}`);
    err.payload = payload;
    err.status = resp.status;
    throw err;
  }
  return payload;
}

function loadChecklistState() {
  try {
    return JSON.parse(localStorage.getItem(CHECKLIST_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveChecklistState(state) {
  localStorage.setItem(CHECKLIST_KEY, JSON.stringify(state));
}

function updateChecklistProgress() {
  const boxes = [...checklistRoot.querySelectorAll('input[type="checkbox"]')];
  const done = boxes.filter((box) => box.checked).length;
  const total = boxes.length;
  const pct = Math.round((done / total) * 100);
  progressBar.style.width = `${pct}%`;
  progressText.textContent = `${done}/${total} complete`;
}

function initChecklist() {
  const state = loadChecklistState();
  const boxes = checklistRoot.querySelectorAll('input[type="checkbox"]');
  boxes.forEach((box) => {
    const key = box.dataset.key;
    box.checked = Boolean(state[key]);
    box.addEventListener("change", () => {
      state[key] = box.checked;
      saveChecklistState(state);
      updateChecklistProgress();
    });
  });
  updateChecklistProgress();
}

async function loadApiStatus() {
  setStatLoading(apiStateEl, true);
  setStatLoading(templateCountEl, true);
  try {
    const healthResp = await fetch(`${getApiBase()}/healthz`);
    if (!healthResp.ok) throw new Error("health request failed");
    setStatLoading(apiStateEl, false);
    apiStateEl.textContent = "online";
  } catch {
    setStatLoading(apiStateEl, false);
    apiStateEl.textContent = "offline";
  }

  try {
    const tplResp = await fetch(`${getApiBase()}/api/v1/maps/templates`);
    if (!tplResp.ok) throw new Error("templates request failed");
    const payload = await tplResp.json();
    setStatLoading(templateCountEl, false);
    templateCountEl.textContent = String(payload.count);
  } catch {
    setStatLoading(templateCountEl, false);
    templateCountEl.textContent = "n/a";
  }
}

function initHeroParallax() {
  if (!heroArt) return;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const maxShift = 6;
  window.addEventListener("pointermove", (event) => {
    const x = (event.clientX / window.innerWidth - 0.5) * 2;
    const y = (event.clientY / window.innerHeight - 0.5) * 2;
    heroArt.style.transform = `translate(${x * maxShift}px, ${y * maxShift}px)`;
  });
}

function markChecklistStep(step, checked = true) {
  const el = checklistRoot?.querySelector(`input[data-key="${step}"]`);
  if (!el) return;
  el.checked = checked;
  const state = loadChecklistState();
  state[step] = checked;
  saveChecklistState(state);
  updateChecklistProgress();
}

function syncTownFields(townId) {
  const clean = (townId || "").trim();
  if (!clean) return;
  if (joinTownIdInput && !joinTownIdInput.value.trim()) joinTownIdInput.value = clean;
  if (simTownIdInput && !simTownIdInput.value.trim()) simTownIdInput.value = clean;
}

async function refreshSetupInfo() {
  try {
    const payload = await apiJson("/api/v1/agents/setup-info");
    setupInfoOutput.textContent = toPrettyJson(payload);
  } catch (err) {
    setupInfoOutput.textContent = `ERROR: ${err.message}`;
  }
}

async function applyLowCostPreset() {
  if (!lowCostPresetOutput) return;
  lowCostPresetOutput.textContent = "Applying situational token preset...";
  persistApiBase();
  try {
    const payload = await apiJson("/api/v1/llm/presets/situational-default", {
      method: "POST",
    });
    lowCostPresetOutput.textContent = toPrettyJson(payload);
  } catch (err) {
    lowCostPresetOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function refreshTowns() {
  if (!townsOutput) return;
  townsOutput.textContent = "Loading towns...";
  setStatLoading(liveAgentsEl, true);
  try {
    const payload = await apiJson("/api/v1/towns");
    const towns = Array.isArray(payload.towns) ? payload.towns : [];
    const firstTown = towns[0] || null;
    if (firstTown?.town_id && joinTownIdInput && !joinTownIdInput.value.trim()) {
      joinTownIdInput.value = firstTown.town_id;
    }
    if (firstTown?.town_id && simTownIdInput && !simTownIdInput.value.trim()) {
      simTownIdInput.value = firstTown.town_id;
    }

    // Render live town cards
    if (townCardsContainer) {
      townCardsContainer.innerHTML = "";
      let totalAgents = 0;
      towns.forEach((town) => {
        const pop = town.population || 0;
        const maxPop = town.max_agents || 25;
        const slots = town.available_slots != null ? town.available_slots : (maxPop - pop);
        const isFull = town.is_full || pop >= maxPop;
        totalAgents += pop;

        const card = document.createElement("article");
        card.className = "town-card town-card-live";
        card.addEventListener("click", () => {
          const base = getApiBase();
          const viewerUrl = `./town.html?api=${encodeURIComponent(base)}&town=${encodeURIComponent(town.town_id)}`;
          window.location.href = viewerUrl;
        });

        const badgeClass = isFull ? "badge badge-full" : "badge badge-open";
        const badgeText = isFull ? "Full" : `${slots} slots open`;

        card.innerHTML = `
          <h3>${town.town_id || "unknown"}</h3>
          <p>${pop}/${maxPop} agents</p>
          <div class="town-card-meta">
            <span class="${badgeClass}">${badgeText}</span>
          </div>
        `;
        townCardsContainer.appendChild(card);
      });

      if (liveAgentsEl) {
        setStatLoading(liveAgentsEl, false);
        liveAgentsEl.textContent = String(totalAgents);
      }
    }

    townsOutput.textContent = toPrettyJson(payload);
  } catch (err) {
    townsOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
    if (liveAgentsEl) {
      setStatLoading(liveAgentsEl, false);
      liveAgentsEl.textContent = "n/a";
    }
  }
}

function getCurrentApiKey() {
  return (meApiKeyInput?.value || "").trim();
}

async function joinScenarioQueue(scenarioKey) {
  if (!myScenarioOutput) return;
  const key = getCurrentApiKey();
  if (!key) {
    myScenarioOutput.textContent = "ERROR: missing API key in /me section";
    return;
  }
  myScenarioOutput.textContent = `Joining queue: ${scenarioKey} ...`;
  try {
    const payload = await apiJson(`/api/v1/scenarios/${encodeURIComponent(scenarioKey)}/queue/join`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}` },
    });
    myScenarioOutput.textContent = toPrettyJson(payload);
    await refreshScenarios();
  } catch (err) {
    myScenarioOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function leaveScenarioQueue(scenarioKey) {
  if (!myScenarioOutput) return;
  const key = getCurrentApiKey();
  if (!key) {
    myScenarioOutput.textContent = "ERROR: missing API key in /me section";
    return;
  }
  myScenarioOutput.textContent = `Leaving queue: ${scenarioKey} ...`;
  try {
    const payload = await apiJson(`/api/v1/scenarios/${encodeURIComponent(scenarioKey)}/queue/leave`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}` },
    });
    myScenarioOutput.textContent = toPrettyJson(payload);
    await refreshScenarios();
  } catch (err) {
    myScenarioOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function refreshScenarios() {
  if (!scenariosOutput || !scenarioCardsContainer) return;
  scenariosOutput.textContent = "Loading scenarios...";
  try {
    const payload = await apiJson("/api/v1/scenarios");
    const scenarios = Array.isArray(payload.scenarios) ? payload.scenarios : [];
    scenarioCardsContainer.innerHTML = "";

    const grouped = {
      social: [],
      casino: [],
      other: [],
    };
    scenarios.forEach((scenario) => {
      const group = scenarioUiGroupFromMeta(scenario);
      if (group === "social") grouped.social.push(scenario);
      else if (group === "casino") grouped.casino.push(scenario);
      else grouped.other.push(scenario);
    });

    const groupSpecs = [
      { key: "social", title: "Social Deduction", subtitle: "Hidden roles and social strategy." },
      { key: "casino", title: "Mini-Games / Casino", subtitle: "Fake-money betting and ranked competition." },
      { key: "other", title: "Other Scenarios", subtitle: "Additional game modes." },
    ];

    for (const group of groupSpecs) {
      const entries = grouped[group.key] || [];
      if (!entries.length) continue;
      const wrapper = document.createElement("section");
      wrapper.className = "scenario-group";
      wrapper.innerHTML = `
        <h4 class="scenario-group-title">${group.title}</h4>
        <p class="scenario-group-subtitle">${group.subtitle}</p>
      `;
      const grid = document.createElement("div");
      grid.className = "town-grid";

      for (const scenario of entries) {
        let queueStatus = null;
        try {
          queueStatus = await apiJson(`/api/v1/scenarios/${encodeURIComponent(scenario.scenario_key)}/queue/status`);
        } catch {
          queueStatus = null;
        }

        const kind = scenarioKindFromMeta(scenario);
        const kindLabel = scenarioLabelFromKind(kind);
        const card = document.createElement("article");
        card.className = "town-card town-card-live";
        const minPlayers = Number(scenario.min_players || 0);
        const maxPlayers = Number(scenario.max_players || 0);
        const queued = Number(queueStatus?.queued || 0);
        const buyIn = Number(scenario.buy_in || 0);
        const category = String(scenario.category || "").replace(/_/g, " ");

        card.innerHTML = `
          <h3>${scenario.name || scenario.scenario_key}</h3>
          <p>${kindLabel}${category ? ` · ${category}` : ""}</p>
          <p>${minPlayers}-${maxPlayers} players · ${queued} queued</p>
          <div class="town-card-meta">
            <span class="badge ${queued >= minPlayers ? "badge-open" : "badge-full"}">${queued >= minPlayers ? "Ready Soon" : "Waiting"}</span>
            <span class="badge">${buyIn > 0 ? `${buyIn} buy-in` : "No buy-in"}</span>
          </div>
          <div class="row-inline">
            <button class="btn btn-small" data-join="${scenario.scenario_key}">Join Queue</button>
            <button class="btn btn-small btn-ghost" data-leave="${scenario.scenario_key}">Leave Queue</button>
          </div>
        `;

        card.querySelector("[data-join]")?.addEventListener("click", (event) => {
          event.stopPropagation();
          joinScenarioQueue(scenario.scenario_key);
        });
        card.querySelector("[data-leave]")?.addEventListener("click", (event) => {
          event.stopPropagation();
          leaveScenarioQueue(scenario.scenario_key);
        });
        grid.appendChild(card);
      }

      wrapper.appendChild(grid);
      scenarioCardsContainer.appendChild(wrapper);
    }

    await refreshScenarioServers();
    await refreshScenarioLeaderboards(scenarios);
    scenariosOutput.textContent = toPrettyJson(payload);
  } catch (err) {
    scenariosOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function refreshScenarioServers() {
  if (!scenarioServerList) return;
  scenarioServerList.innerHTML = "";
  try {
    const payload = await apiJson("/api/v1/scenarios/servers");
    const servers = Array.isArray(payload.servers) ? payload.servers : [];
    if (servers.length === 0) {
      const empty = document.createElement("article");
      empty.className = "town-card";
      empty.innerHTML = `
        <h3>No live servers</h3>
        <p>Queue for a scenario to spin up a new match server in your town.</p>
      `;
      scenarioServerList.appendChild(empty);
      return;
    }

    const base = getApiBase();
    servers.forEach((server) => {
      const card = document.createElement("article");
      card.className = "town-card town-card-live";
      const kind = scenarioKindFromMeta(server);
      const scenarioLabel = scenarioLabelFromKind(kind);
      const scenarioName = String(server.scenario_name || scenarioLabel || server.scenario_key || "Scenario");
      const townId = String(server.town_id || "");
      const watchUrl = `./town.html?api=${encodeURIComponent(base)}&town=${encodeURIComponent(townId)}`;
      card.innerHTML = `
        <h3>${scenarioName} · ${townId || "town"}</h3>
        <p>${Number(server.participant_count || 0)} players · ${server.phase || "warmup"}</p>
        <div class="town-card-meta">
          <span class="badge">${scenarioLabel}</span>
          <span class="badge">${server.match_id || ""}</span>
          <span class="badge">${Number(server.pot || 0)} pot</span>
        </div>
        <div class="row-inline">
          <button class="btn btn-small" data-watch="1">Watch</button>
          <button class="btn btn-small btn-ghost" data-queue="1">Queue Same Game</button>
        </div>
      `;
      card.querySelector("[data-watch]")?.addEventListener("click", (event) => {
        event.stopPropagation();
        window.location.href = watchUrl;
      });
      card.querySelector("[data-queue]")?.addEventListener("click", (event) => {
        event.stopPropagation();
        joinScenarioQueue(String(server.scenario_key || ""));
      });
      scenarioServerList.appendChild(card);
    });
  } catch (err) {
    const failed = document.createElement("article");
    failed.className = "town-card";
    failed.innerHTML = `
      <h3>Server list unavailable</h3>
      <p>${String(err?.message || "Unknown error")}</p>
    `;
    scenarioServerList.appendChild(failed);
  }
}

async function refreshScenarioLeaderboards(scenarios) {
  if (!scenarioLeaderboardList) return;
  scenarioLeaderboardList.innerHTML = "";
  const items = Array.isArray(scenarios) ? scenarios : [];
  if (items.length === 0) return;

  for (const scenario of items) {
    const scenarioKey = String(scenario.scenario_key || "");
    if (!scenarioKey) continue;
    let payload;
    try {
      payload = await apiJson(`/api/v1/scenarios/ratings/${encodeURIComponent(scenarioKey)}?limit=5`);
    } catch {
      payload = { leaderboard: [] };
    }
    const leaderboard = Array.isArray(payload.leaderboard) ? payload.leaderboard : [];
    const card = document.createElement("article");
    card.className = "town-card";
    const rowsHtml = leaderboard.length
      ? leaderboard
          .slice(0, 5)
          .map((entry, idx) => {
            const tier = ratingTier(entry.rating);
            const agentId = String(entry.agent_id || "");
            return `
              <div class="leaderboard-row">
                <span>#${idx + 1} ${agentId.slice(0, 10)}</span>
                <span class="tier-badge ${tier.cls}"><img class="tier-badge-icon" src="${tier.icon}" alt="${tier.label}" />${tier.label}</span>
                <span>${Number(entry.rating || 0)}</span>
              </div>
            `;
          })
          .join("")
      : '<div class="leaderboard-empty">No ranked matches yet.</div>';
    card.innerHTML = `
      <h3>${scenario.name || scenarioKey}</h3>
      ${rowsHtml}
    `;
    scenarioLeaderboardList.appendChild(card);
  }
}

async function loadMyScenarioQueue() {
  if (!myScenarioOutput) return;
  const key = getCurrentApiKey();
  if (!key) {
    myScenarioOutput.textContent = "ERROR: missing API key in /me section";
    return;
  }
  myScenarioOutput.textContent = "Loading queue...";
  try {
    const payload = await apiJson("/api/v1/scenarios/me/queue", {
      headers: { Authorization: `Bearer ${key}` },
    });
    myScenarioOutput.textContent = toPrettyJson(payload);
  } catch (err) {
    myScenarioOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function loadMyScenarioMatch() {
  if (!myScenarioOutput) return;
  const key = getCurrentApiKey();
  if (!key) {
    myScenarioOutput.textContent = "ERROR: missing API key in /me section";
    return;
  }
  myScenarioOutput.textContent = "Loading active match...";
  try {
    const payload = await apiJson("/api/v1/scenarios/me/match", {
      headers: { Authorization: `Bearer ${key}` },
    });
    myScenarioOutput.textContent = toPrettyJson(payload);
    const matchId = payload?.match?.match_id;
    if (matchId && payload?.match?.town_id) {
      const base = getApiBase();
      const watchUrl = `./town.html?api=${encodeURIComponent(base)}&town=${encodeURIComponent(payload.match.town_id)}`;
      myScenarioOutput.textContent += `\n\nWatch in town viewer: ${watchUrl}`;
    }
  } catch (err) {
    myScenarioOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function forfeitMyScenarioMatch() {
  if (!myScenarioOutput) return;
  const key = getCurrentApiKey();
  if (!key) {
    myScenarioOutput.textContent = "ERROR: missing API key in /me section";
    return;
  }
  myScenarioOutput.textContent = "Checking active match...";
  try {
    const me = await apiJson("/api/v1/scenarios/me/match", {
      headers: { Authorization: `Bearer ${key}` },
    });
    const matchId = String(me?.match?.match_id || "");
    if (!me?.active || !matchId) {
      myScenarioOutput.textContent = "No active match to forfeit.";
      return;
    }
    myScenarioOutput.textContent = `Forfeiting ${matchId}...`;
    const payload = await apiJson(`/api/v1/scenarios/matches/${encodeURIComponent(matchId)}/forfeit`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}` },
    });
    myScenarioOutput.textContent = toPrettyJson(payload);
    await refreshScenarios();
  } catch (err) {
    myScenarioOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onRegisterSubmit(event) {
  event.preventDefault();
  registerOutput.textContent = "Registering...";
  persistApiBase();
  const generatePersona = document.getElementById("registerGeneratePersona")?.checked || false;
  const spriteSel = document.getElementById("registerSprite");
  const spriteName = spriteSel?.value || null;
  if (generatePersona) {
    registerOutput.textContent = "Generating persona via AI — this may take a few seconds...";
  }
  try {
    const payload = await apiJson("/api/v1/agents/register", {
      method: "POST",
      body: {
        name: document.getElementById("registerName").value.trim() || "",
        owner_handle: document.getElementById("registerOwner").value.trim() || null,
        description: document.getElementById("registerDescription").value.trim() || null,
        auto_claim: document.getElementById("registerAutoClaim").checked,
        generate_persona: generatePersona,
        sprite_name: spriteName || null,
      },
    });
    registerOutput.textContent = toPrettyJson(payload);
    const agent = payload.agent || {};
    if (agent.api_key) {
      meApiKeyInput.value = agent.api_key;
      markChecklistStep("register", true);
    }
    if (agent.claimed) markChecklistStep("claim", true);
    if (agent.claim_token) claimTokenInput.value = agent.claim_token;
    if (agent.verification_code) claimCodeInput.value = agent.verification_code;
    if (agent.owner_handle) claimOwnerInput.value = agent.owner_handle;
  } catch (err) {
    registerOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onClaimSubmit(event) {
  event.preventDefault();
  claimOutput.textContent = "Claiming...";
  persistApiBase();
  try {
    const payload = await apiJson("/api/v1/agents/claim", {
      method: "POST",
      body: {
        claim_token: claimTokenInput.value.trim(),
        verification_code: claimCodeInput.value.trim(),
        owner_handle: claimOwnerInput.value.trim() || null,
      },
    });
    claimOutput.textContent = toPrettyJson(payload);
    if (payload.ok) markChecklistStep("claim", true);
  } catch (err) {
    claimOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onMeSubmit(event) {
  event.preventDefault();
  meOutput.textContent = "Fetching /me...";
  persistApiBase();
  const key = meApiKeyInput.value.trim();
  if (!key) {
    meOutput.textContent = "ERROR: missing API key";
    return;
  }
  try {
    const payload = await apiJson("/api/v1/agents/me", {
      headers: {
        Authorization: `Bearer ${key}`,
      },
    });
    meOutput.textContent = toPrettyJson(payload);
    syncTownFields(payload?.agent?.current_town?.town_id || "");
    markChecklistStep("join", Boolean(payload?.agent?.current_town?.town_id));
    markChecklistStep("watch", true);
  } catch (err) {
    meOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function copyKey() {
  const value = meApiKeyInput.value.trim();
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    meOutput.textContent = "Copied API key to clipboard.";
  } catch {
    meOutput.textContent = "Clipboard write failed. Copy key manually from the field.";
  }
}

async function onJoinTownSubmit(event) {
  event.preventDefault();
  if (!joinTownOutput) return;
  joinTownOutput.textContent = "Joining town...";
  persistApiBase();
  const key = meApiKeyInput.value.trim();
  const townId = joinTownIdInput?.value.trim() || "";
  if (!key) {
    joinTownOutput.textContent = "ERROR: missing API key";
    return;
  }
  if (!townId) {
    joinTownOutput.textContent = "ERROR: missing town id";
    return;
  }

  try {
    const payload = await apiJson("/api/v1/agents/me/join-town", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
      },
      body: {
        town_id: townId,
      },
    });
    joinTownOutput.textContent = toPrettyJson(payload);
    syncTownFields(payload?.membership?.town_id || townId);
    markChecklistStep("join", true);
    await refreshTowns();
  } catch (err) {
    joinTownOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onAutoJoinClick() {
  if (!joinTownOutput) return;
  joinTownOutput.textContent = "Auto-joining best town...";
  persistApiBase();
  const key = meApiKeyInput.value.trim();
  if (!key) {
    joinTownOutput.textContent = "ERROR: missing API key";
    return;
  }
  try {
    const payload = await apiJson("/api/v1/agents/me/auto-join", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
      },
    });
    joinTownOutput.textContent = toPrettyJson(payload);
    syncTownFields(payload?.membership?.town_id || "");
    markChecklistStep("join", true);
    await refreshTowns();
  } catch (err) {
    joinTownOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onLeaveTownClick() {
  if (!joinTownOutput) return;
  joinTownOutput.textContent = "Leaving town...";
  persistApiBase();
  const key = meApiKeyInput.value.trim();
  if (!key) {
    joinTownOutput.textContent = "ERROR: missing API key";
    return;
  }
  try {
    const payload = await apiJson("/api/v1/agents/me/leave-town", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
      },
    });
    joinTownOutput.textContent = toPrettyJson(payload);
    markChecklistStep("join", false);
    await refreshTowns();
  } catch (err) {
    joinTownOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onSimStateClick() {
  if (!simOutput) return;
  simOutput.textContent = "Loading simulation state...";
  persistApiBase();
  const townId = simTownIdInput?.value.trim() || "";
  if (!townId) {
    simOutput.textContent = "ERROR: missing town id";
    return;
  }
  try {
    const payload = await apiJson(`/api/v1/sim/towns/${encodeURIComponent(townId)}/state`);
    simOutput.textContent = toPrettyJson(payload);
    markChecklistStep("watch", true);
  } catch (err) {
    simOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onSimTickClick() {
  if (!simOutput) return;
  simOutput.textContent = "Running simulation tick...";
  persistApiBase();
  const townId = simTownIdInput?.value.trim() || "";
  if (!townId) {
    simOutput.textContent = "ERROR: missing town id";
    return;
  }
  const parsedSteps = Number.parseInt(simTickStepsInput?.value || "1", 10);
  const steps = Number.isFinite(parsedSteps) && parsedSteps > 0 ? parsedSteps : 1;
  const planningScope = simPlanningScopeInput?.value || "short_action";
  const controlMode = simControlModeInput?.value || "hybrid";
  try {
    const payload = await apiJson(`/api/v1/sim/towns/${encodeURIComponent(townId)}/tick`, {
      method: "POST",
      body: { steps, planning_scope: planningScope, control_mode: controlMode },
    });
    simOutput.textContent = toPrettyJson(payload);
    markChecklistStep("watch", true);
  } catch (err) {
    simOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onLegacyListClick() {
  if (!legacyOutput) return;
  legacyOutput.textContent = "Loading simulations...";
  persistApiBase();
  try {
    const payload = await apiJson("/api/v1/legacy/simulations");
    legacyOutput.textContent = toPrettyJson(payload);
    const first = Array.isArray(payload.simulations) ? payload.simulations[0] : null;
    if (first?.simulation_id && legacySimulationIdInput && !legacySimulationIdInput.value.trim()) {
      legacySimulationIdInput.value = first.simulation_id;
    }
  } catch (err) {
    legacyOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onLegacyReplayClick() {
  if (!legacyOutput) return;
  legacyOutput.textContent = "Replaying legacy window...";
  persistApiBase();
  const simulationId = legacySimulationIdInput?.value.trim() || "";
  if (!simulationId) {
    legacyOutput.textContent = "ERROR: missing simulation id";
    return;
  }

  const startStep = Number.parseInt(legacyStartStepInput?.value || "0", 10);
  const maxSteps = Number.parseInt(legacyMaxStepsInput?.value || "120", 10);

  try {
    const payload = await apiJson(`/api/v1/legacy/simulations/${encodeURIComponent(simulationId)}/events`, {
      method: "POST",
      body: {
        start_step: Number.isFinite(startStep) && startStep >= 0 ? startStep : 0,
        max_steps: Number.isFinite(maxSteps) && maxSteps > 0 ? maxSteps : 120,
      },
    });
    legacyOutput.textContent = toPrettyJson(payload);
  } catch (err) {
    legacyOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

async function onLegacyImportClick() {
  if (!legacyImportOutput) return;
  legacyImportOutput.textContent = "Importing The Ville...";
  persistApiBase();
  const townId = legacyTownIdInput?.value.trim() || "the_ville_legacy";
  try {
    const payload = await apiJson("/api/v1/legacy/import-the-ville", {
      method: "POST",
      body: {
        town_id: townId,
        publish_version_record: true,
      },
    });
    legacyImportOutput.textContent = toPrettyJson(payload);
    await refreshTowns();
  } catch (err) {
    legacyImportOutput.textContent = `ERROR: ${err.message}\n${toPrettyJson(err.payload || {})}`;
  }
}

function initOnboardingConsole() {
  if (!apiBaseInput) return;
  const apiFromQuery = normalizeBase(new URLSearchParams(window.location.search).get("api"));
  apiBaseInput.value = apiFromQuery || normalizeBase(localStorage.getItem(API_BASE_KEY)) || guessDefaultApiBase();
  apiBaseInput.addEventListener("change", () => {
    persistApiBase();
    loadApiStatus();
    refreshSetupInfo();
  });
  refreshSetupInfoBtn?.addEventListener("click", () => {
    persistApiBase();
    loadApiStatus();
    refreshSetupInfo();
  });
  applyLowCostPresetBtn?.addEventListener("click", applyLowCostPreset);
  registerForm?.addEventListener("submit", onRegisterSubmit);
  claimForm?.addEventListener("submit", onClaimSubmit);
  meForm?.addEventListener("submit", onMeSubmit);
  copyKeyBtn?.addEventListener("click", copyKey);
  joinTownForm?.addEventListener("submit", onJoinTownSubmit);
  leaveTownBtn?.addEventListener("click", onLeaveTownClick);
  autoJoinBtn?.addEventListener("click", onAutoJoinClick);
  refreshTownsBtn?.addEventListener("click", () => {
    persistApiBase();
    refreshTowns();
  });
  refreshScenariosBtn?.addEventListener("click", () => {
    persistApiBase();
    refreshScenarios();
  });
  myScenarioQueueBtn?.addEventListener("click", loadMyScenarioQueue);
  myScenarioMatchBtn?.addEventListener("click", loadMyScenarioMatch);
  myScenarioForfeitBtn?.addEventListener("click", forfeitMyScenarioMatch);
  legacyListBtn?.addEventListener("click", onLegacyListClick);
  legacyReplayBtn?.addEventListener("click", onLegacyReplayClick);
  legacyImportBtn?.addEventListener("click", onLegacyImportClick);
  simStateBtn?.addEventListener("click", onSimStateClick);
  simTickBtn?.addEventListener("click", onSimTickClick);

  // Populate sprite picker from API (fallback to static list)
  const spriteSelect = document.getElementById("registerSprite");
  if (spriteSelect) {
    apiJson("/api/v1/agents/characters").then((data) => {
      (data.characters || []).forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name.replace(/_/g, " ");
        spriteSelect.appendChild(opt);
      });
    }).catch(() => {
      // Fallback: hardcoded list
      const fallback = [
        "Abigail_Chen", "Adam_Smith", "Arthur_Burton", "Ayesha_Khan",
        "Carlos_Gomez", "Carmen_Ortiz", "Eddy_Lin", "Francisco_Lopez",
        "Giorgio_Rossi", "Hailey_Johnson", "Isabella_Rodriguez", "Jane_Moreno",
        "Jennifer_Moore", "John_Lin", "Klaus_Mueller", "Latoya_Williams",
        "Maria_Lopez", "Mei_Lin", "Rajiv_Patel", "Ryan_Park",
        "Sam_Moore", "Tamara_Taylor", "Tom_Moreno", "Wolfgang_Schulz",
        "Yuriko_Yamamoto",
      ];
      fallback.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name.replace(/_/g, " ");
        spriteSelect.appendChild(opt);
      });
    });
  }
}

initChecklist();
initOnboardingConsole();
persistApiBase();
preserveApiParamOnTownLinks();
loadApiStatus();
refreshSetupInfo();
refreshTowns();
refreshScenarios();
initHeroParallax();
