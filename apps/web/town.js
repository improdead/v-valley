/* V-Valley Town Viewer — Phaser 3 game + API integration */
(function () {
  "use strict";

  const API_BASE = (function () {
    const params = new URLSearchParams(window.location.search);
    if (params.get("api")) return params.get("api").replace(/\/+$/, "");
    return window.location.origin;
  })();
  const API = `${API_BASE}/api/v1`;
  const TILE_SIZE = 32;
  const POLL_INTERVAL_MS = 3000;
  const SPECTATE_POLL_MS = 1000;
  const ANIM_DURATION_MS = 800;
  const MAX_EVENTS = 100;
  const MAX_BUBBLES = 3;

  const CHARACTER_SPRITES = [
    "Abigail_Chen", "Adam_Smith", "Arthur_Burton", "Ayesha_Khan",
    "Carlos_Gomez", "Carmen_Ortiz", "Eddy_Lin", "Francisco_Lopez",
    "Giorgio_Rossi", "Hailey_Johnson", "Isabella_Rodriguez", "Jane_Moreno",
    "Jennifer_Moore", "John_Lin", "Klaus_Mueller", "Latoya_Williams",
    "Maria_Lopez", "Mei_Lin", "Rajiv_Patel", "Ryan_Park",
    "Sam_Moore", "Tamara_Taylor", "Tom_Moreno", "Wolfgang_Schulz",
    "Yuriko_Yamamoto",
  ];

  let currentTownId = null;
  let phaserGame = null;
  let gameScene = null;
  let autoTickTimer = null;
  let pollTimer = null;
  let pollInFlight = false;
  let spectateTimer = null;
  let activeSpectateMatchId = null;

  let agentSpriteMap = {}; // agent_id -> { sprite, label, lastX, lastY, ... }
  let agentCharMap = {};
  let latestState = null;
  let latestMatches = [];

  let prevNpcs = {};
  let prevConvos = {};
  let prevPronunciatio = {};
  let eventLog = [];
  let activeBubbles = [];
  let lastAgentListHash = "";
  let knownMatchIds = new Set();

  const $townSelect = document.getElementById("townSelect");
  const $btnRefreshTowns = document.getElementById("btnRefreshTowns");
  const $btnTick = document.getElementById("btnTick");
  const $btnAutoTick = document.getElementById("btnAutoTick");
  const $tickScope = document.getElementById("tickScope");
  const $tickMode = document.getElementById("tickMode");
  const $gameTime = document.getElementById("gameTime");
  const $stepCount = document.getElementById("stepCount");
  const $agentList = document.getElementById("agentList");
  const $convoList = document.getElementById("convoList");
  const $loadingOverlay = document.getElementById("loading-overlay");
  const $eventFeed = document.getElementById("eventFeed");
  const $eventFeedNew = document.getElementById("eventFeedNew");
  const $connDot = document.getElementById("connDot");
  const $connLabel = document.getElementById("connLabel");
  const $matchList = document.getElementById("matchList");
  const $zoomIn = document.getElementById("zoomIn");
  const $zoomOut = document.getElementById("zoomOut");

  const $matchModal = document.getElementById("matchModal");
  const $matchModalClose = document.getElementById("matchModalClose");
  const $matchModalTitle = document.getElementById("matchModalTitle");
  const $matchModalPhase = document.getElementById("matchModalPhase");
  const $matchModalPlayers = document.getElementById("matchModalPlayers");
  const $matchModalEvents = document.getElementById("matchModalEvents");

  function hashStr(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    return Math.abs(h);
  }

  function assignCharacter(agentId, serverSpriteName) {
    if (!agentCharMap[agentId]) {
      if (serverSpriteName && CHARACTER_SPRITES.includes(serverSpriteName)) {
        agentCharMap[agentId] = serverSpriteName;
      } else {
        agentCharMap[agentId] = CHARACTER_SPRITES[hashStr(agentId) % CHARACTER_SPRITES.length];
      }
    }
    return agentCharMap[agentId];
  }

  function getInitials(name) {
    const rgx = /(\p{L}{1})\p{L}+/gu;
    const matches = [...(name || "").matchAll(rgx)];
    return ((matches.shift()?.[1] || "") + (matches.pop()?.[1] || "")).toUpperCase() || "??";
  }

  function formatClock(clock) {
    if (!clock) return "--:--";
    const h = String(clock.hour).padStart(2, "0");
    const m = String(clock.minute).padStart(2, "0");
    return `${h}:${m}`;
  }

  async function apiFetch(path, opts) {
    const res = await fetch(`${API}${path}`, opts);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
    }
    return res.json();
  }

  function setConnectionStatus(state, detail = "") {
    if (!$connDot || !$connLabel) return;
    $connDot.classList.remove("ok", "warn", "err");
    if (state === "inflight") {
      $connDot.classList.add("warn");
      $connLabel.textContent = "Polling...";
    } else if (state === "error") {
      $connDot.classList.add("err");
      $connLabel.textContent = "Disconnected";
    } else {
      $connDot.classList.add("ok");
      $connLabel.textContent = "Connected";
    }
    $connLabel.title = detail || "";
  }

  async function loadTowns() {
    try {
      const data = await apiFetch("/towns");
      $townSelect.innerHTML = '<option value="">-- Select Town --</option>';
      (data.towns || []).forEach((town) => {
        const opt = document.createElement("option");
        opt.value = town.town_id;
        opt.textContent = `${town.name || town.town_id} (${town.population}/${town.max_agents})`;
        $townSelect.appendChild(opt);
      });
      if (currentTownId) $townSelect.value = currentTownId;
    } catch (e) {
      console.error("Failed to load towns:", e);
    }
  }

  async function runTick() {
    if (!currentTownId) return;
    const townId = currentTownId;
    $btnTick.textContent = "...";
    setConnectionStatus("inflight", "Manual tick in progress");
    try {
      const data = await apiFetch(`/sim/towns/${townId}/tick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steps: 1,
          planning_scope: $tickScope.value,
          control_mode: $tickMode.value,
        }),
      });
      if (townId !== currentTownId) return;
      if (data.tick && data.tick.state) {
        updateFromState(data.tick.state, data.tick?.scenario?.active_matches || []);
      }
      setConnectionStatus("connected", "Tick completed");
    } catch (e) {
      console.error("Tick error:", e);
      setConnectionStatus("error", String(e?.message || e));
    } finally {
      $btnTick.textContent = "Tick";
    }
  }

  function toggleAutoTick() {
    if (autoTickTimer) {
      clearInterval(autoTickTimer);
      autoTickTimer = null;
      $btnAutoTick.classList.remove("active");
      $btnAutoTick.textContent = "Auto";
    } else {
      autoTickTimer = setInterval(runTick, 5000);
      $btnAutoTick.classList.add("active");
      $btnAutoTick.textContent = "Stop";
      runTick();
    }
  }

  async function pollState() {
    if (!currentTownId || pollInFlight) return;
    const townId = currentTownId;
    pollInFlight = true;
    setConnectionStatus("inflight", "Polling town state");
    try {
      const data = await apiFetch(`/sim/towns/${townId}/state`);
      let matches = [];
      try {
        const activeMatches = await apiFetch(`/scenarios/towns/${townId}/active`);
        matches = activeMatches.matches || [];
      } catch (matchErr) {
        console.warn("Active matches fetch failed:", matchErr);
      }
      if (townId !== currentTownId) return;
      if (data.state) updateFromState(data.state, matches);
      setConnectionStatus("connected", `Step ${data?.state?.step ?? "--"}`);
    } catch (e) {
      console.error("Poll error:", e);
      setConnectionStatus("error", String(e?.message || e));
    } finally {
      pollInFlight = false;
    }
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollState, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    pollInFlight = false;
  }

  function updateFromState(state, scenarioMatches) {
    latestState = state;
    latestMatches = Array.isArray(scenarioMatches) ? scenarioMatches : [];

    if (state.clock) {
      $gameTime.textContent = formatClock(state.clock);
      $stepCount.textContent = `step ${state.step || 0}`;
      if (gameScene) updateDayNight(gameScene, Number(state.clock.hour || 0));
    }

    const npcs = state.npcs || [];
    const convos = state.conversations || [];

    const events = diffState({ state, npcs, convos, matches: latestMatches });
    if (events.length > 0) appendEvents(events);

    updateAgentCards(npcs);
    if (gameScene) updateAgentSprites(npcs);
    updateConversations(convos);
    renderActiveMatches(latestMatches);
  }

  function diffState({ state, npcs, convos, matches }) {
    const events = [];

    const nextNpcMap = {};
    npcs.forEach((npc) => {
      nextNpcMap[npc.agent_id] = {
        x: npc.x,
        y: npc.y,
        current_location: npc.current_location,
        goal_reason: npc.goal?.reason || "",
        status: npc.status,
        pronunciatio: npc.pronunciatio || "",
      };
      const prev = prevNpcs[npc.agent_id];
      if (!prev) {
        events.push({
          type: "arrival",
          emoji: "🎉",
          title: `${npc.name} joined the town`,
          detail: npc.current_location ? `@ ${npc.current_location}` : "new resident",
          step: state.step,
          agentId: npc.agent_id,
        });
      } else {
        if (prev.current_location !== npc.current_location && npc.current_location) {
          events.push({
            type: "movement",
            emoji: "🚶",
            title: `${npc.name} moved`,
            detail: `to ${npc.current_location}`,
            step: state.step,
            agentId: npc.agent_id,
          });
        }
        const goalReason = npc.goal?.reason || "";
        if (goalReason && prev.goal_reason !== goalReason) {
          events.push({
            type: "goal",
            emoji: "🎯",
            title: `${npc.name} updated goal`,
            detail: goalReason,
            step: state.step,
            agentId: npc.agent_id,
          });
        }
      }

      if ((prevPronunciatio[npc.agent_id] || "") !== (npc.pronunciatio || "") && npc.pronunciatio) {
        showSpeechBubbleForAgent(npc.agent_id, npc.pronunciatio, 2200);
      }
      prevPronunciatio[npc.agent_id] = npc.pronunciatio || "";
    });

    Object.keys(prevNpcs).forEach((agentId) => {
      if (!nextNpcMap[agentId]) {
        events.push({
          type: "arrival",
          emoji: "👋",
          title: `${agentId} left the town`,
          detail: "departed",
          step: state.step,
          agentId,
        });
      }
    });

    const nextConvos = {};
    convos.forEach((convo) => {
      const key = convo.session_id;
      nextConvos[key] = {
        turns: Number(convo.turns || 0),
        last_message: convo.last_message || "",
      };
      const prev = prevConvos[key];
      const aName = npcName(convo.agent_a);
      const bName = npcName(convo.agent_b);
      if (!prev) {
        events.push({
          type: "social",
          emoji: "💬",
          title: `${aName} and ${bName} started talking`,
          detail: `session ${convo.session_id.slice(0, 6)}`,
          step: state.step,
          agentId: convo.agent_a,
        });
      } else if ((prev.last_message || "") !== (convo.last_message || "") && convo.last_message) {
        const speakerId = Number(convo.turns || 0) % 2 === 1 ? convo.agent_a : convo.agent_b;
        showSpeechBubbleForAgent(speakerId, convo.last_message, 7000);
      }
    });

    const nextMatchIds = new Set((matches || []).map((m) => String(m.match_id || "")).filter(Boolean));
    nextMatchIds.forEach((matchId) => {
      if (!knownMatchIds.has(matchId)) {
        const m = (matches || []).find((item) => String(item.match_id) === matchId);
        const label = m?.scenario_key?.includes("werewolf") ? "Werewolf" : "Anaconda";
        events.push({
          type: "goal",
          emoji: "🎮",
          title: `New ${label} match formed`,
          detail: `${m?.phase || "warmup"} · ${m?.participant_count || 0} players`,
          step: state.step,
          agentId: null,
        });
        showMatchToast(`${label} match formed · ${m?.participant_count || 0} players`);
      }
    });
    knownMatchIds = nextMatchIds;

    prevNpcs = nextNpcMap;
    prevConvos = nextConvos;
    return events;
  }

  function npcName(agentId) {
    const npc = (latestState?.npcs || []).find((item) => String(item.agent_id) === String(agentId));
    return npc?.name || agentId;
  }

  function appendEvents(events) {
    let newCount = 0;
    const shouldStick = isEventFeedAtBottom();
    events.forEach((evt) => {
      eventLog.push(evt);
      newCount += 1;
    });
    if (eventLog.length > MAX_EVENTS) {
      eventLog = eventLog.slice(eventLog.length - MAX_EVENTS);
    }
    renderEventFeed();
    if (shouldStick) {
      scrollEventFeedBottom();
      if ($eventFeedNew) $eventFeedNew.style.display = "none";
    } else if (newCount > 0 && $eventFeedNew) {
      $eventFeedNew.textContent = `↓ ${newCount} new events`;
      $eventFeedNew.style.display = "inline-flex";
    }
  }

  function renderEventFeed() {
    if (!$eventFeed) return;
    if (eventLog.length === 0) {
      $eventFeed.innerHTML = '<div class="empty-state"><p>No events yet</p></div>';
      return;
    }

    $eventFeed.innerHTML = eventLog
      .map((evt, idx) => {
        const cls = evt.type || "goal";
        const agentAttr = evt.agentId ? `data-agent-id="${evt.agentId}"` : "";
        return `
          <button class="event-item ${cls}" data-event-idx="${idx}" ${agentAttr}>
            <div class="event-title">${evt.emoji || "•"} ${escapeHtml(evt.title || "")}</div>
            <div class="event-detail">${escapeHtml(evt.detail || "")} · step ${evt.step ?? "--"}</div>
          </button>
        `;
      })
      .join("");

    $eventFeed.querySelectorAll(".event-item").forEach((el) => {
      el.addEventListener("click", () => {
        const aid = el.getAttribute("data-agent-id");
        if (aid) window._focusAgent(aid);
      });
    });
  }

  function isEventFeedAtBottom() {
    if (!$eventFeed) return true;
    const threshold = 16;
    return $eventFeed.scrollTop + $eventFeed.clientHeight >= $eventFeed.scrollHeight - threshold;
  }

  function scrollEventFeedBottom() {
    if (!$eventFeed) return;
    $eventFeed.scrollTop = $eventFeed.scrollHeight;
  }

  function updateAgentCards(npcs) {
    const hash = npcs
      .map((npc) => {
        const scenarioId = npc.scenario?.match_id || "";
        return `${npc.agent_id}:${npc.x}:${npc.y}:${npc.status}:${npc.pronunciatio || ""}:${scenarioId}`;
      })
      .join("|");

    if (hash === lastAgentListHash) return;
    lastAgentListHash = hash;

    const selectedId = document.querySelector(".agent-card.focused")?.dataset.agentId || "";

    if (npcs.length === 0) {
      $agentList.innerHTML = '<div class="empty-state"><p>No agents in this town</p></div>';
      return;
    }

    $agentList.innerHTML = npcs
      .map((npc) => {
        const initials = getInitials(npc.name);
        const emoji = npc.pronunciatio || "";
        const action = npc.memory_summary?.active_action || "idle";
        const goalReason = npc.goal?.reason || "";
        const scenarioBadge = npc.scenario
          ? `<span class="agent-scenario-badge">🎮 ${escapeHtml(npc.scenario.scenario_key || "match")}</span>`
          : "";
        return `
          <div class="agent-card" data-agent-id="${npc.agent_id}">
            <span class="agent-pronunciatio">${emoji || initials}</span>
            <div class="agent-name">${escapeHtml(npc.name)}</div>
            <div class="agent-owner">${npc.owner_handle ? "@" + escapeHtml(npc.owner_handle) : "unclaimed"}</div>
            ${scenarioBadge}
            <div class="agent-action">${escapeHtml(action)}${goalReason ? " — " + escapeHtml(goalReason) : ""}</div>
            <div class="agent-pos">(${npc.x}, ${npc.y})${npc.current_location ? " @ " + escapeHtml(npc.current_location) : ""}</div>
          </div>
        `;
      })
      .join("");

    $agentList.querySelectorAll(".agent-card").forEach((card) => {
      card.addEventListener("click", () => window._focusAgent(card.dataset.agentId));
    });

    if (selectedId) {
      const selected = document.querySelector(`.agent-card[data-agent-id="${selectedId}"]`);
      if (selected) selected.classList.add("focused");
    }
  }

  function updateConversations(convos) {
    if (!latestState) return;
    const nameMap = {};
    (latestState.npcs || []).forEach((npc) => {
      nameMap[npc.agent_id] = npc.name;
    });

    if (convos.length === 0) {
      $convoList.innerHTML = '<div class="empty-state"><p>No active conversations</p></div>';
      return;
    }

    $convoList.innerHTML = convos
      .map((convo) => {
        const nameA = nameMap[convo.agent_a] || convo.agent_a;
        const nameB = nameMap[convo.agent_b] || convo.agent_b;
        return `
          <div class="convo-card">
            <div class="convo-agents">${escapeHtml(nameA)} ↔ ${escapeHtml(nameB)}</div>
            <div class="convo-msg">${escapeHtml(convo.last_message || "...")}</div>
            <div class="convo-meta">${Number(convo.turns || 0)} turns · started step ${Number(convo.started_step || 0)}</div>
          </div>
        `;
      })
      .join("");
  }

  function renderActiveMatches(matches) {
    if (!$matchList) return;
    if (!matches || matches.length === 0) {
      $matchList.innerHTML = '<div class="empty-state"><p>No active games</p></div>';
      return;
    }

    $matchList.innerHTML = matches
      .map((match) => {
        const scenarioLabel = String(match.scenario_key || "").includes("werewolf") ? "🐺 Werewolf" : "🃏 Anaconda";
        const meta = String(match.scenario_key || "").includes("anaconda")
          ? `Pot ${Number(match.pot || 0)} · ${Number(match.participant_count || 0)} players`
          : `${Number(match.participant_count || 0)} players · round ${Number(match.round_number || 0)}`;
        return `
          <div class="match-card">
            <div class="match-title">${scenarioLabel} · ${escapeHtml(match.phase || "warmup")}</div>
            <div class="match-meta">${escapeHtml(meta)}</div>
            <button class="match-watch-btn" data-match-id="${escapeHtml(String(match.match_id || ""))}">Watch Live</button>
          </div>
        `;
      })
      .join("");

    $matchList.querySelectorAll(".match-watch-btn").forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        const matchId = btn.getAttribute("data-match-id") || "";
        if (matchId) openMatchModal(matchId);
      });
    });
  }

  function showMatchToast(text) {
    const toast = document.createElement("div");
    toast.className = "match-toast";
    toast.textContent = `🎮 ${text}`;
    document.body.appendChild(toast);
    window.setTimeout(() => toast.remove(), 5000);
  }

  function openMatchModal(matchId) {
    activeSpectateMatchId = matchId;
    if (!$matchModal) return;
    $matchModal.style.display = "flex";
    pollMatchModal();
    if (spectateTimer) clearInterval(spectateTimer);
    spectateTimer = setInterval(pollMatchModal, SPECTATE_POLL_MS);
  }

  function closeMatchModal() {
    activeSpectateMatchId = null;
    if (spectateTimer) clearInterval(spectateTimer);
    spectateTimer = null;
    if ($matchModal) $matchModal.style.display = "none";
  }

  async function pollMatchModal() {
    if (!activeSpectateMatchId) return;
    try {
      const payload = await apiFetch(`/scenarios/matches/${activeSpectateMatchId}/spectate`);
      if (!activeSpectateMatchId) return;
      const state = payload.public_state || {};
      const events = payload.recent_events || [];
      if ($matchModalTitle) {
        $matchModalTitle.textContent = `${state.scenario_key || "match"} · ${state.match_id || ""}`;
      }
      if ($matchModalPhase) {
        $matchModalPhase.textContent = `Status ${state.status || "--"} · Phase ${state.phase || "--"} · Round ${state.round_number ?? "--"}`;
      }
      if ($matchModalPlayers) {
        const participants = state.participants || [];
        $matchModalPlayers.innerHTML = participants
          .map((p) => {
            const role = p.role ? ` (${p.role})` : "";
            const chips = p.chips_end != null ? ` · chips ${p.chips_end}` : "";
            return `<div class="match-player-row">${escapeHtml(npcName(p.agent_id))}${role} · ${escapeHtml(p.status || "")}${chips}</div>`;
          })
          .join("");
      }
      if ($matchModalEvents) {
        const rows = events.slice(-40);
        $matchModalEvents.innerHTML = rows
          .map((evt) => {
            const data = evt.data_json || {};
            const detail = typeof data === "object" ? JSON.stringify(data) : String(data || "");
            return `<div class="match-event-row">s${evt.step} · ${escapeHtml(evt.phase || "")} · ${escapeHtml(evt.event_type || "event")} · ${escapeHtml(detail)}</div>`;
          })
          .join("");
      }
    } catch (e) {
      if ($matchModalPhase) $matchModalPhase.textContent = `Failed to load: ${String(e?.message || e)}`;
    }
  }

  function createPhaserGame() {
    if (phaserGame) {
      phaserGame.destroy(true);
      phaserGame = null;
      gameScene = null;
    }
    agentSpriteMap = {};

    const container = document.getElementById("game-container");
    const config = {
      type: Phaser.AUTO,
      width: container.clientWidth,
      height: container.clientHeight,
      parent: "game-container",
      pixelArt: true,
      physics: { default: "arcade", arcade: { gravity: { y: 0 } } },
      scene: { preload, create, update },
      scale: {
        mode: Phaser.Scale.RESIZE,
        autoCenter: Phaser.Scale.CENTER_BOTH,
      },
    };
    phaserGame = new Phaser.Game(config);
  }

  function preload() {
    $loadingOverlay.textContent = "Loading map assets...";

    this.load.image("blocks_1", "assets/map/map_assets/blocks/blocks_1.png");
    this.load.image("blocks_2", "assets/map/map_assets/blocks/blocks_2.png");
    this.load.image("blocks_3", "assets/map/map_assets/blocks/blocks_3.png");
    this.load.image("walls", "assets/map/map_assets/v1/Room_Builder_32x32.png");
    this.load.image("interiors_pt1", "assets/map/map_assets/v1/interiors_pt1.png");
    this.load.image("interiors_pt2", "assets/map/map_assets/v1/interiors_pt2.png");
    this.load.image("interiors_pt3", "assets/map/map_assets/v1/interiors_pt3.png");
    this.load.image("interiors_pt4", "assets/map/map_assets/v1/interiors_pt4.png");
    this.load.image("interiors_pt5", "assets/map/map_assets/v1/interiors_pt5.png");
    this.load.image("CuteRPG_Field_B", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Field_B.png");
    this.load.image("CuteRPG_Field_C", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Field_C.png");
    this.load.image("CuteRPG_Harbor_C", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Harbor_C.png");
    this.load.image("CuteRPG_Village_B", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Village_B.png");
    this.load.image("CuteRPG_Forest_B", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Forest_B.png");
    this.load.image("CuteRPG_Desert_C", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Desert_C.png");
    this.load.image("CuteRPG_Mountains_B", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Mountains_B.png");
    this.load.image("CuteRPG_Desert_B", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Desert_B.png");
    this.load.image("CuteRPG_Forest_C", "assets/map/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Forest_C.png");
    this.load.tilemapTiledJSON("map", "assets/map/the_ville.json");

    CHARACTER_SPRITES.forEach((name) => {
      this.load.atlas(`char_${name}`, `assets/characters/${name}.png`, "assets/characters/atlas.json");
    });
  }

  function create() {
    gameScene = this;
    $loadingOverlay.textContent = "Building world...";

    const map = this.make.tilemap({ key: "map" });

    const collisions = map.addTilesetImage("blocks", "blocks_1");
    const blocks2 = map.addTilesetImage("blocks_2", "blocks_2");
    const blocks3 = map.addTilesetImage("blocks_3", "blocks_3");
    const walls = map.addTilesetImage("Room_Builder_32x32", "walls");
    const interiors_pt1 = map.addTilesetImage("interiors_pt1", "interiors_pt1");
    const interiors_pt2 = map.addTilesetImage("interiors_pt2", "interiors_pt2");
    const interiors_pt3 = map.addTilesetImage("interiors_pt3", "interiors_pt3");
    const interiors_pt4 = map.addTilesetImage("interiors_pt4", "interiors_pt4");
    const interiors_pt5 = map.addTilesetImage("interiors_pt5", "interiors_pt5");
    const CuteRPG_Field_B = map.addTilesetImage("CuteRPG_Field_B", "CuteRPG_Field_B");
    const CuteRPG_Field_C = map.addTilesetImage("CuteRPG_Field_C", "CuteRPG_Field_C");
    const CuteRPG_Harbor_C = map.addTilesetImage("CuteRPG_Harbor_C", "CuteRPG_Harbor_C");
    const CuteRPG_Village_B = map.addTilesetImage("CuteRPG_Village_B", "CuteRPG_Village_B");
    const CuteRPG_Forest_B = map.addTilesetImage("CuteRPG_Forest_B", "CuteRPG_Forest_B");
    const CuteRPG_Desert_C = map.addTilesetImage("CuteRPG_Desert_C", "CuteRPG_Desert_C");
    const CuteRPG_Mountains_B = map.addTilesetImage("CuteRPG_Mountains_B", "CuteRPG_Mountains_B");
    const CuteRPG_Desert_B = map.addTilesetImage("CuteRPG_Desert_B", "CuteRPG_Desert_B");
    const CuteRPG_Forest_C = map.addTilesetImage("CuteRPG_Forest_C", "CuteRPG_Forest_C");

    const allTilesets = [
      CuteRPG_Field_B, CuteRPG_Field_C, CuteRPG_Harbor_C,
      CuteRPG_Village_B, CuteRPG_Forest_B, CuteRPG_Desert_C,
      CuteRPG_Mountains_B, CuteRPG_Desert_B, CuteRPG_Forest_C,
      interiors_pt1, interiors_pt2, interiors_pt3, interiors_pt4, interiors_pt5,
      walls,
    ];

    map.createLayer("Bottom Ground", allTilesets, 0, 0);
    map.createLayer("Exterior Ground", allTilesets, 0, 0);
    map.createLayer("Exterior Decoration L1", allTilesets, 0, 0);
    map.createLayer("Exterior Decoration L2", allTilesets, 0, 0);
    map.createLayer("Interior Ground", allTilesets, 0, 0);
    map.createLayer("Wall", [CuteRPG_Field_C, walls], 0, 0);
    map.createLayer("Interior Furniture L1", allTilesets, 0, 0);
    map.createLayer("Interior Furniture L2 ", allTilesets, 0, 0);

    const fg1 = map.createLayer("Foreground L1", allTilesets, 0, 0);
    const fg2 = map.createLayer("Foreground L2", allTilesets, 0, 0);
    if (fg1) fg1.setDepth(10);
    if (fg2) fg2.setDepth(10);

    const collLayer = map.createLayer("Collisions", [collisions, blocks2, blocks3], 0, 0);
    if (collLayer) collLayer.setDepth(-1);

    this._cameraTarget = this.physics.add
      .sprite(map.widthInPixels / 2, map.heightInPixels / 2, `char_${CHARACTER_SPRITES[0]}`, "down")
      .setSize(1, 1)
      .setAlpha(0);

    const camera = this.cameras.main;
    camera.startFollow(this._cameraTarget, true, 0.1, 0.1);
    camera.setBounds(0, 0, map.widthInPixels, map.heightInPixels);
    camera.setZoom(1);

    this._dayNightOverlay = this.add
      .rectangle(0, 0, camera.width * 4, camera.height * 4, 0x000000, 0)
      .setScrollFactor(0)
      .setDepth(998)
      .setOrigin(0, 0);

    this._cursors = this.input.keyboard.createCursorKeys();
    this._keys = this.input.keyboard.addKeys("W,A,S,D,PLUS,MINUS");

    this.input.on("wheel", (pointer, gameObjects, deltaX, deltaY) => {
      const newZoom = Phaser.Math.Clamp(camera.zoom - deltaY * 0.001, 0.3, 3);
      camera.setZoom(newZoom);
    });

    CHARACTER_SPRITES.forEach((name) => {
      const key = `char_${name}`;
      ["down", "left", "right", "up"].forEach((dir) => {
        const animKey = `${key}-${dir}-walk`;
        if (!this.anims.exists(animKey)) {
          this.anims.create({
            key: animKey,
            frames: this.anims.generateFrameNames(key, {
              prefix: `${dir}-walk.`,
              start: 0,
              end: 3,
              zeroPad: 3,
            }),
            frameRate: 6,
            repeat: -1,
          });
        }
      });
    });

    $loadingOverlay.style.display = "none";
    pollState();
    startPolling();
  }

  function update(time, delta) {
    if (!this._cameraTarget || !this._cursors) return;
    const speed = 500;
    this._cameraTarget.body.setVelocity(0);

    if (this._cursors.left.isDown || this._keys.A.isDown) this._cameraTarget.body.setVelocityX(-speed);
    if (this._cursors.right.isDown || this._keys.D.isDown) this._cameraTarget.body.setVelocityX(speed);
    if (this._cursors.up.isDown || this._keys.W.isDown) this._cameraTarget.body.setVelocityY(-speed);
    if (this._cursors.down.isDown || this._keys.S.isDown) this._cameraTarget.body.setVelocityY(speed);
  }

  function updateDayNight(scene, hour) {
    const overlay = scene?._dayNightOverlay;
    if (!overlay) return;

    let color = 0x000000;
    let alpha = 0;
    let icon = "☀️";

    if (hour >= 22 || hour < 6) {
      color = 0x0a1a3a;
      alpha = 0.35;
      icon = "🌙";
    } else if (hour >= 20 && hour < 22) {
      color = 0x1a2a4a;
      alpha = 0.2;
      icon = "🌙";
    } else if (hour >= 17 && hour < 20) {
      color = 0xffaa55;
      alpha = 0.12;
      icon = "🌅";
    } else if (hour >= 6 && hour < 8) {
      color = 0xff9944;
      alpha = 0.1;
      icon = "🌄";
    }

    overlay.setFillStyle(color, alpha);
    scene.tweens.add({
      targets: overlay,
      alpha,
      duration: 1200,
      ease: "Sine.easeInOut",
    });

    if ($gameTime) $gameTime.textContent = `${icon} ${$gameTime.textContent.replace(/^[^\d]*/, "")}`;
  }

  function updateAgentSprites(npcs) {
    if (!gameScene) return;
    const seenIds = new Set();

    npcs.forEach((npc) => {
      seenIds.add(npc.agent_id);
      const charName = assignCharacter(npc.agent_id, npc.sprite_name);
      const atlasKey = `char_${charName}`;
      const targetX = npc.x * TILE_SIZE + TILE_SIZE / 2;
      const targetY = npc.y * TILE_SIZE + TILE_SIZE;

      if (!agentSpriteMap[npc.agent_id]) {
        const sprite = gameScene.physics.add
          .sprite(targetX, targetY, atlasKey, "down")
          .setSize(30, 40)
          .setDepth(5)
          .setInteractive({ useHandCursor: true });

        sprite.on("pointerdown", () => window._focusAgent(npc.agent_id));
        sprite.on("pointerover", () => sprite.setTint(0xe94560));
        sprite.on("pointerout", () => sprite.clearTint());

        const label = gameScene.add
          .text(targetX, targetY - 48, getInitials(npc.name), {
            font: "bold 11px monospace",
            color: "#ffffff",
            backgroundColor: "#00000099",
            padding: { x: 4, y: 2 },
          })
          .setOrigin(0.5, 1)
          .setDepth(11);

        agentSpriteMap[npc.agent_id] = {
          sprite,
          label,
          atlasKey,
          lastX: npc.x,
          lastY: npc.y,
          tween: null,
          labelTween: null,
        };
      } else {
        const entry = agentSpriteMap[npc.agent_id];
        const sprite = entry.sprite;
        const label = entry.label;
        const dx = npc.x - entry.lastX;
        const dy = npc.y - entry.lastY;

        if (dx !== 0 || dy !== 0) {
          let dir = "down";
          if (Math.abs(dx) >= Math.abs(dy)) dir = dx > 0 ? "right" : "left";
          else dir = dy > 0 ? "down" : "up";

          sprite.anims.play(`${atlasKey}-${dir}-walk`, true);
          if (entry.tween) entry.tween.stop();
          if (entry.labelTween) entry.labelTween.stop();

          entry.tween = gameScene.tweens.add({
            targets: sprite,
            x: targetX,
            y: targetY,
            duration: ANIM_DURATION_MS,
            ease: "Linear",
            onComplete: () => {
              sprite.anims.stop();
              sprite.setTexture(atlasKey, dir);
            },
          });

          entry.labelTween = gameScene.tweens.add({
            targets: label,
            x: targetX,
            y: targetY - 48,
            duration: ANIM_DURATION_MS,
            ease: "Linear",
          });
        }

        const initials = getInitials(npc.name);
        const emoji = npc.pronunciatio || "";
        const scenarioIcon = npc.scenario ? "🎮 " : "";
        label.setText(emoji ? `${scenarioIcon}${emoji} ${initials}` : `${scenarioIcon}${initials}`);

        entry.lastX = npc.x;
        entry.lastY = npc.y;
      }
    });

    Object.keys(agentSpriteMap).forEach((id) => {
      if (!seenIds.has(id)) {
        agentSpriteMap[id].sprite.destroy();
        agentSpriteMap[id].label.destroy();
        delete agentSpriteMap[id];
      }
    });
  }

  function showSpeechBubbleForAgent(agentId, text, duration = 8000) {
    if (!gameScene || !agentSpriteMap[agentId] || !text) return;

    if (activeBubbles.length >= MAX_BUBBLES) {
      const oldest = activeBubbles.shift();
      if (oldest && oldest.destroy) oldest.destroy();
    }

    const sprite = agentSpriteMap[agentId].sprite;
    const bubble = gameScene.add
      .text(sprite.x, sprite.y - 64, String(text).slice(0, 80), {
        font: "13px VT323",
        color: "#1a2d48",
        backgroundColor: "#faf6e8",
        padding: { x: 6, y: 3 },
        wordWrap: { width: 190 },
        align: "left",
      })
      .setOrigin(0.5, 1)
      .setDepth(24)
      .setAlpha(0.96);

    activeBubbles.push(bubble);
    gameScene.tweens.add({
      targets: bubble,
      alpha: 0,
      y: bubble.y - 6,
      delay: Math.max(400, duration - 900),
      duration: 900,
      ease: "Sine.easeInOut",
      onComplete: () => {
        const idx = activeBubbles.indexOf(bubble);
        if (idx >= 0) activeBubbles.splice(idx, 1);
        bubble.destroy();
      },
    });
  }

  function selectTown(townId) {
    if (!townId) return;

    stopPolling();
    closeMatchModal();
    if (autoTickTimer) {
      clearInterval(autoTickTimer);
      autoTickTimer = null;
      $btnAutoTick.classList.remove("active");
      $btnAutoTick.textContent = "Auto";
    }

    if (phaserGame) {
      phaserGame.destroy(true);
      phaserGame = null;
      gameScene = null;
    }

    agentSpriteMap = {};
    agentCharMap = {};
    prevNpcs = {};
    prevConvos = {};
    prevPronunciatio = {};
    activeBubbles.forEach((bubble) => bubble?.destroy?.());
    activeBubbles = [];

    currentTownId = townId;
    knownMatchIds = new Set();

    $loadingOverlay.style.display = "flex";
    $loadingOverlay.textContent = "Loading town...";

    createPhaserGame();
  }

  function changeZoom(delta) {
    if (!gameScene) return;
    const cam = gameScene.cameras.main;
    cam.setZoom(Phaser.Math.Clamp(cam.zoom + delta, 0.3, 3));
  }

  function escapeHtml(input) {
    return String(input)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  window._focusAgent = function (agentId) {
    if (!gameScene || !agentSpriteMap[agentId]) return;
    const sprite = agentSpriteMap[agentId].sprite;
    gameScene.cameras.main.pan(sprite.x, sprite.y, 500, "Sine.easeInOut");
    document.querySelectorAll(".agent-card").forEach((el) => el.classList.remove("focused"));
    const card = document.querySelector(`.agent-card[data-agent-id="${agentId}"]`);
    if (card) card.classList.add("focused");
  };

  $townSelect.addEventListener("change", (e) => selectTown(e.target.value));
  $btnRefreshTowns.addEventListener("click", loadTowns);
  $btnTick.addEventListener("click", runTick);
  $btnAutoTick.addEventListener("click", toggleAutoTick);
  $zoomIn?.addEventListener("click", () => changeZoom(0.15));
  $zoomOut?.addEventListener("click", () => changeZoom(-0.15));
  $eventFeed?.addEventListener("scroll", () => {
    if (isEventFeedAtBottom() && $eventFeedNew) $eventFeedNew.style.display = "none";
  });
  $eventFeedNew?.addEventListener("click", () => {
    scrollEventFeedBottom();
    $eventFeedNew.style.display = "none";
  });
  $matchModalClose?.addEventListener("click", closeMatchModal);
  $matchModal?.addEventListener("click", (event) => {
    if (event.target === $matchModal) closeMatchModal();
  });

  const urlTown = new URLSearchParams(window.location.search).get("town");

  loadTowns().then(() => {
    if (urlTown) {
      $townSelect.value = urlTown;
      selectTown(urlTown);
    }
  });
})();
