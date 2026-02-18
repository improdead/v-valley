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
  const STREAM_INTERVAL_MS = 1000;
  const STREAM_RETRY_MS = 2500;
  const ANIM_DURATION_MS = 800;
  const MAX_EVENTS = 100;
  const MAX_BUBBLES = 3;
  const FORCE_POLLING = new URLSearchParams(window.location.search).get("transport") === "poll";

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
  let stateStream = null;
  let streamRetryTimer = null;
  let spectateTimer = null;
  let activeSpectateMatchId = null;
  let activeMatchFrames = [];
  let matchReplayMode = false;
  let matchReplayIndex = 0;
  let matchReplayTimer = null;
  let matchShowRoles = false;
  let lastSpectatePhase = "";

  let agentSpriteMap = {}; // agent_id -> { sprite, label, lastX, lastY, ... }
  let agentCharMap = {};
  let latestState = null;
  let latestMatches = [];
  let latestLiveSnapshot = null;
  let timelineSnapshots = [];
  let replayMode = false;
  let replayTimer = null;
  let replayIndex = 0;

  let prevNpcs = {};
  let prevConvos = {};
  let prevPronunciatio = {};
  let conversationHistory = {}; // session_id -> { agentA, agentB, startedStep, turns[] }
  let activeConversationId = null;
  let eventLog = [];
  let activeBubbles = [];
  let lastAgentListHash = "";
  let knownMatchIds = new Set();
  let focusedAgentId = null;
  let activeLocationName = "";
  let relationGraphNodes = [];

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
  const $btnReplay = document.getElementById("btnReplay");
  const $btnRelations = document.getElementById("btnRelations");

  const $matchModal = document.getElementById("matchModal");
  const $matchModalClose = document.getElementById("matchModalClose");
  const $matchModalTitle = document.getElementById("matchModalTitle");
  const $matchModalPhase = document.getElementById("matchModalPhase");
  const $matchModalPlayers = document.getElementById("matchModalPlayers");
  const $matchModalEvents = document.getElementById("matchModalEvents");
  const $matchModalBoard = document.getElementById("matchModalBoard");
  const $matchRolesToggle = document.getElementById("matchRolesToggle");
  const $matchReplayToggle = document.getElementById("matchReplayToggle");
  const $matchReplayControls = document.getElementById("matchReplayControls");
  const $matchReplayPlay = document.getElementById("matchReplayPlay");
  const $matchReplaySlider = document.getElementById("matchReplaySlider");
  const $matchReplayLabel = document.getElementById("matchReplayLabel");
  const $matchReplayLive = document.getElementById("matchReplayLive");
  const $matchVoteTracker = document.getElementById("matchVoteTracker");
  const $matchSceneBanner = document.getElementById("matchSceneBanner");
  const $matchTranscriptBtn = document.getElementById("matchTranscriptBtn");
  const $matchTranscriptModal = document.getElementById("matchTranscriptModal");
  const $matchTranscriptClose = document.getElementById("matchTranscriptClose");
  const $matchTranscriptBody = document.getElementById("matchTranscriptBody");
  const $convoModal = document.getElementById("convoModal");
  const $convoModalClose = document.getElementById("convoModalClose");
  const $convoModalTitle = document.getElementById("convoModalTitle");
  const $convoModalMeta = document.getElementById("convoModalMeta");
  const $convoModalBody = document.getElementById("convoModalBody");
  const $convoModalJump = document.getElementById("convoModalJump");
  const $locationTooltip = document.getElementById("locationTooltip");
  const $locationPanel = document.getElementById("locationPanel");
  const $locationPanelTitle = document.getElementById("locationPanelTitle");
  const $locationPanelBody = document.getElementById("locationPanelBody");
  const $locationPanelClose = document.getElementById("locationPanelClose");
  const $replayModal = document.getElementById("replayModal");
  const $replayModalClose = document.getElementById("replayModalClose");
  const $replaySlider = document.getElementById("replaySlider");
  const $replayPlay = document.getElementById("replayPlay");
  const $replayLive = document.getElementById("replayLive");
  const $replayMeta = document.getElementById("replayMeta");
  const $relationModal = document.getElementById("relationModal");
  const $relationModalClose = document.getElementById("relationModalClose");
  const $relationCanvas = document.getElementById("relationCanvas");
  const $agentDrawer = document.getElementById("agentDrawer");
  const $agentDrawerTitle = document.getElementById("agentDrawerTitle");
  const $agentDrawerBody = document.getElementById("agentDrawerBody");
  const $agentDrawerClose = document.getElementById("agentDrawerClose");

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

  function ensureConversationRecord(convo) {
    const sessionId = String(convo?.session_id || "").trim();
    if (!sessionId) return null;
    if (!conversationHistory[sessionId]) {
      conversationHistory[sessionId] = {
        sessionId,
        agentA: String(convo?.agent_a || ""),
        agentB: String(convo?.agent_b || ""),
        startedStep: Number(convo?.started_step || 0),
        turns: [],
      };
    }
    const rec = conversationHistory[sessionId];
    rec.agentA = String(convo?.agent_a || rec.agentA || "");
    rec.agentB = String(convo?.agent_b || rec.agentB || "");
    rec.startedStep = Number(convo?.started_step || rec.startedStep || 0);
    return rec;
  }

  function appendConversationTurn(convo, speakerId, text, step) {
    const rec = ensureConversationRecord(convo);
    if (!rec) return;
    const line = String(text || "").trim();
    if (!line) return;
    const speaker = String(speakerId || "").trim();
    const last = rec.turns[rec.turns.length - 1];
    if (last && last.speakerId === speaker && last.text === line) return;
    rec.turns.push({
      turn: Number(convo?.turns || rec.turns.length + 1),
      speakerId: speaker,
      text: line,
      step: Number(step || latestState?.step || 0),
      clock: formatClock(latestState?.clock),
    });
    if (rec.turns.length > 60) {
      rec.turns = rec.turns.slice(rec.turns.length - 60);
    }
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
      if (townId !== currentTownId) return;
      if (data.state) updateFromState(data.state, data.state?.scenario_matches || []);
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
    pollState();
  }

  function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    pollInFlight = false;
  }

  function stopStateStream() {
    if (stateStream) {
      stateStream.close();
      stateStream = null;
    }
    if (streamRetryTimer) {
      clearTimeout(streamRetryTimer);
      streamRetryTimer = null;
    }
  }

  function scheduleStateStreamRetry(townId) {
    if (streamRetryTimer) clearTimeout(streamRetryTimer);
    streamRetryTimer = window.setTimeout(() => {
      streamRetryTimer = null;
      if (currentTownId === townId && !stateStream && !FORCE_POLLING) {
        startStateStream();
      }
    }, STREAM_RETRY_MS);
  }

  function startStateStream() {
    if (!currentTownId || FORCE_POLLING || typeof window.EventSource === "undefined") {
      startPolling();
      return;
    }
    const townId = currentTownId;
    stopStateStream();
    stopPolling();
    const streamUrl = `${API}/sim/towns/${encodeURIComponent(townId)}/events/stream?interval_ms=${STREAM_INTERVAL_MS}`;
    try {
      stateStream = new EventSource(streamUrl);
    } catch (err) {
      console.warn("EventSource init failed, fallback to polling:", err);
      setConnectionStatus("error", "Stream init failed, using polling");
      startPolling();
      return;
    }

    stateStream.addEventListener("open", () => {
      if (townId !== currentTownId) return;
      setConnectionStatus("connected", "Live stream connected");
    });

    stateStream.addEventListener("state", (event) => {
      if (townId !== currentTownId) return;
      try {
        const payload = JSON.parse(String(event?.data || "{}"));
        if (payload?.state) {
          updateFromState(payload.state, payload.state?.scenario_matches || []);
          setConnectionStatus("connected", `Live · step ${payload?.state?.step ?? "--"}`);
        }
      } catch (err) {
        console.warn("Stream parse error:", err);
      }
    });

    stateStream.addEventListener("error", (event) => {
      if (townId !== currentTownId) return;
      const payloadText = String(event?.data || "");
      if (payloadText) {
        try {
          const payload = JSON.parse(payloadText);
          console.warn("Stream event error:", payload);
        } catch {
          console.warn("Stream event error payload:", payloadText);
        }
      }
      setConnectionStatus("error", "Stream interrupted, retrying via polling");
      stopStateStream();
      startPolling();
      scheduleStateStreamRetry(townId);
    });
  }

  function startTownDataFlow() {
    if (FORCE_POLLING) {
      startPolling();
      return;
    }
    if (typeof window.EventSource === "undefined") {
      startPolling();
      return;
    }
    startStateStream();
  }

  function stopTownDataFlow() {
    stopPolling();
    stopStateStream();
  }

  function cloneJsonSafe(value) {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch {
      return value;
    }
  }

  function recordTimelineSnapshot(state, scenarioMatches) {
    const matches = Array.isArray(scenarioMatches) ? scenarioMatches : [];
    const step = Number(state?.step || 0);
    const snapshot = {
      step,
      capturedAt: Date.now(),
      state: cloneJsonSafe(state),
      matches: cloneJsonSafe(matches),
    };
    const last = timelineSnapshots[timelineSnapshots.length - 1];
    if (last && Number(last.step) === step) timelineSnapshots[timelineSnapshots.length - 1] = snapshot;
    else timelineSnapshots.push(snapshot);
    if (timelineSnapshots.length > 120) {
      timelineSnapshots = timelineSnapshots.slice(timelineSnapshots.length - 120);
    }
    latestLiveSnapshot = snapshot;
  }

  function syncLivePrevState({ state, npcs, convos, matches }) {
    const nextNpcMap = {};
    (npcs || []).forEach((npc) => {
      nextNpcMap[npc.agent_id] = {
        x: npc.x,
        y: npc.y,
        current_location: npc.current_location,
        goal_reason: npc.goal?.reason || "",
        status: npc.status,
        pronunciatio: npc.pronunciatio || "",
      };
      prevPronunciatio[npc.agent_id] = npc.pronunciatio || "";
    });
    prevNpcs = nextNpcMap;

    const nextConvos = {};
    (convos || []).forEach((convo) => {
      ensureConversationRecord(convo);
      const key = convo.session_id;
      nextConvos[key] = {
        turns: Number(convo.turns || 0),
        last_message: convo.last_message || "",
      };
      const prev = prevConvos[key];
      if (!prev && convo.last_message) {
        const speakerId = Number(convo.turns || 0) % 2 === 1 ? convo.agent_a : convo.agent_b;
        appendConversationTurn(convo, speakerId, convo.last_message, state?.step || 0);
      } else if ((prev?.last_message || "") !== (convo.last_message || "") && convo.last_message) {
        const speakerId = Number(convo.turns || 0) % 2 === 1 ? convo.agent_a : convo.agent_b;
        appendConversationTurn(convo, speakerId, convo.last_message, state?.step || 0);
      }
    });
    prevConvos = nextConvos;
    knownMatchIds = new Set((matches || []).map((m) => String(m.match_id || "")).filter(Boolean));
  }

  function renderStateView(state, scenarioMatches, opts = {}) {
    const emitDiff = opts.emitDiff !== false;
    latestState = state;
    latestMatches = Array.isArray(scenarioMatches) ? scenarioMatches : [];

    if (state.clock) {
      $gameTime.textContent = formatClock(state.clock);
      $stepCount.textContent = `step ${state.step || 0}`;
      if (gameScene) updateDayNight(gameScene, Number(state.clock.hour || 0));
    }

    const npcs = state.npcs || [];
    const convos = state.conversations || [];

    if (emitDiff) {
      const events = diffState({ state, npcs, convos, matches: latestMatches });
      if (events.length > 0) appendEvents(events);
    }

    updateAgentCards(npcs);
    if (gameScene) updateAgentSprites(npcs);
    updateConversations(convos);
    renderActiveMatches(latestMatches);
    renderConversationModal();
    renderLocationPanel();
    renderAgentDrawer();
    if ($relationModal && $relationModal.style.display === "flex") {
      renderRelationshipGraph();
    }
  }

  function updateFromState(state, scenarioMatches) {
    const matches = Array.isArray(scenarioMatches) ? scenarioMatches : [];
    recordTimelineSnapshot(state, matches);
    if (replayMode) {
      syncLivePrevState({
        state,
        npcs: state?.npcs || [],
        convos: state?.conversations || [],
        matches,
      });
      return;
    }
    renderStateView(state, matches, { emitDiff: true });
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
      ensureConversationRecord(convo);
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
        if (convo.last_message) {
          const speakerId = Number(convo.turns || 0) % 2 === 1 ? convo.agent_a : convo.agent_b;
          appendConversationTurn(convo, speakerId, convo.last_message, state.step);
        }
      } else if ((prev.last_message || "") !== (convo.last_message || "") && convo.last_message) {
        const speakerId = Number(convo.turns || 0) % 2 === 1 ? convo.agent_a : convo.agent_b;
        appendConversationTurn(convo, speakerId, convo.last_message, state.step);
        showSpeechBubbleForAgent(speakerId, convo.last_message, 7000);
      }
    });

    Object.keys(prevConvos).forEach((sessionId) => {
      if (nextConvos[sessionId]) return;
      const rec = conversationHistory[sessionId];
      const aName = npcName(rec?.agentA || "");
      const bName = npcName(rec?.agentB || "");
      events.push({
        type: "social",
        emoji: "📝",
        title: `${aName} and ${bName} ended conversation`,
        detail: `session ${sessionId.slice(0, 6)}`,
        step: state.step,
        agentId: rec?.agentA || null,
      });
    });

    const nextMatchIds = new Set((matches || []).map((m) => String(m.match_id || "")).filter(Boolean));
    nextMatchIds.forEach((matchId) => {
      if (!knownMatchIds.has(matchId)) {
        const m = (matches || []).find((item) => String(item.match_id) === matchId);
        const label = scenarioDisplayName(m);
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
            <div class="convo-actions">
              <button class="convo-detail-btn" data-convo-id="${escapeHtml(String(convo.session_id || ""))}">Transcript</button>
            </div>
          </div>
        `;
      })
      .join("");

    $convoList.querySelectorAll(".convo-detail-btn").forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        const sessionId = btn.getAttribute("data-convo-id") || "";
        if (sessionId) openConversationModal(sessionId);
      });
    });
  }

  function openConversationModal(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    activeConversationId = sid;
    if ($convoModal) $convoModal.style.display = "flex";
    renderConversationModal();
  }

  function closeConversationModal() {
    activeConversationId = null;
    if ($convoModal) $convoModal.style.display = "none";
  }

  function renderConversationModal() {
    if (!$convoModal || !$convoModalTitle || !$convoModalMeta || !$convoModalBody) return;
    if (!activeConversationId) {
      $convoModal.style.display = "none";
      return;
    }
    const rec = conversationHistory[activeConversationId];
    if (!rec) {
      $convoModalTitle.textContent = "Conversation";
      $convoModalMeta.textContent = `session ${activeConversationId.slice(0, 8)} unavailable`;
      $convoModalBody.innerHTML = '<div class="empty-state"><p>No transcript captured yet.</p></div>';
      return;
    }
    const nameA = npcName(rec.agentA);
    const nameB = npcName(rec.agentB);
    const turnCount = rec.turns.length;
    $convoModalTitle.textContent = `${nameA} ↔ ${nameB}`;
    $convoModalMeta.textContent = `session ${activeConversationId.slice(0, 8)} · ${turnCount} turns · started s${rec.startedStep || 0}`;
    if (turnCount === 0) {
      $convoModalBody.innerHTML = '<div class="empty-state"><p>No transcript captured yet.</p></div>';
      return;
    }
    $convoModalBody.innerHTML = rec.turns
      .map((line, idx) => {
        const speakerName = npcName(line.speakerId || "");
        const label = speakerName || line.speakerId || "speaker";
        const step = Number(line.step || 0);
        return `
          <div class="convo-line">
            <div class="convo-line-head">
              <span>#${idx + 1}</span>
              <span>${escapeHtml(label)}</span>
              <span>step ${step}</span>
              <span>${escapeHtml(line.clock || "--:--")}</span>
            </div>
            <div class="convo-line-text">${escapeHtml(line.text || "")}</div>
          </div>
        `;
      })
      .join("");
  }

  function jumpToConversationAgents() {
    if (!activeConversationId || !gameScene) return;
    const rec = conversationHistory[activeConversationId];
    if (!rec) return;
    const sprites = [rec.agentA, rec.agentB]
      .map((aid) => agentSpriteMap[aid]?.sprite)
      .filter(Boolean);
    if (sprites.length === 0) return;
    const cx = sprites.reduce((sum, sprite) => sum + sprite.x, 0) / sprites.length;
    const cy = sprites.reduce((sum, sprite) => sum + sprite.y, 0) / sprites.length;
    gameScene.cameras.main.pan(cx, cy, 650, "Sine.easeInOut");
    if (rec.agentA) window._focusAgent(rec.agentA);
  }

  function stopReplayPlayback() {
    if (replayTimer) {
      clearInterval(replayTimer);
      replayTimer = null;
    }
    if ($replayPlay) $replayPlay.textContent = "Play";
  }

  function replayToLiveState() {
    replayMode = false;
    stopReplayPlayback();
    if ($replayModal) $replayModal.style.display = "none";
    if (latestLiveSnapshot?.state) {
      renderStateView(latestLiveSnapshot.state, latestLiveSnapshot.matches || [], { emitDiff: false });
      syncLivePrevState({
        state: latestLiveSnapshot.state,
        npcs: latestLiveSnapshot.state?.npcs || [],
        convos: latestLiveSnapshot.state?.conversations || [],
        matches: latestLiveSnapshot.matches || [],
      });
    }
  }

  function renderReplayFrame(index) {
    if (!timelineSnapshots.length) return;
    replayIndex = Math.max(0, Math.min(timelineSnapshots.length - 1, Number(index || 0)));
    const frame = timelineSnapshots[replayIndex];
    if (!frame) return;
    if ($replaySlider) {
      $replaySlider.max = String(Math.max(0, timelineSnapshots.length - 1));
      $replaySlider.value = String(replayIndex);
    }
    if ($replayMeta) {
      const clock = frame.state?.clock ? formatClock(frame.state.clock) : "--:--";
      $replayMeta.textContent = `Frame ${replayIndex + 1}/${timelineSnapshots.length} · step ${frame.step} · ${clock}`;
    }
    renderStateView(frame.state, frame.matches || [], { emitDiff: false });
  }

  function startReplayPlayback() {
    stopReplayPlayback();
    if (!timelineSnapshots.length) return;
    if (replayIndex >= timelineSnapshots.length - 1) {
      renderReplayFrame(0);
    }
    if ($replayPlay) $replayPlay.textContent = "Pause";
    replayTimer = window.setInterval(() => {
      if (replayIndex >= timelineSnapshots.length - 1) {
        stopReplayPlayback();
        return;
      }
      renderReplayFrame(replayIndex + 1);
    }, 700);
  }

  function toggleReplayPlayback() {
    if (!replayMode) return;
    if (replayTimer) {
      stopReplayPlayback();
    } else {
      startReplayPlayback();
    }
  }

  function openReplayModal() {
    if (!$replayModal) return;
    if (!timelineSnapshots.length) {
      if ($replayMeta) $replayMeta.textContent = "No snapshots captured yet.";
      return;
    }
    replayMode = true;
    $replayModal.style.display = "flex";
    if ($replaySlider) {
      $replaySlider.min = "0";
      $replaySlider.max = String(Math.max(0, timelineSnapshots.length - 1));
    }
    renderReplayFrame(timelineSnapshots.length - 1);
  }

  function closeReplayModal() {
    replayToLiveState();
  }

  function renderAgentDrawer() {
    if (!$agentDrawerBody || !$agentDrawerTitle || !$agentDrawer) return;
    if (!focusedAgentId || !latestState) {
      $agentDrawer.style.display = "none";
      return;
    }
    const npc = (latestState.npcs || []).find((item) => String(item.agent_id) === String(focusedAgentId));
    if (!npc) {
      $agentDrawer.style.display = "none";
      return;
    }
    const memory = npc.memory_summary || {};
    const scenario = npc.scenario || null;
    $agentDrawer.style.display = "block";
    $agentDrawerTitle.textContent = npc.name || npc.agent_id;
    const rows = [
      ["Agent ID", npc.agent_id || ""],
      ["Owner", npc.owner_handle ? `@${npc.owner_handle}` : "unclaimed"],
      ["Status", npc.status || "idle"],
      ["Location", npc.current_location || `${npc.x},${npc.y}`],
      ["Goal", npc.goal?.reason || "none"],
      ["Pronunciatio", npc.pronunciatio || "-"],
      ["Active Action", memory.active_action || "idle"],
      ["Recent Reflection", memory.last_reflection || "-"],
    ];
    if (scenario) {
      rows.push(["Scenario", `${scenario.scenario_key || "match"} · ${scenario.phase || "phase"}`]);
      rows.push(["Match Status", scenario.participant_status || scenario.status || "active"]);
    }
    $agentDrawerBody.innerHTML = rows
      .map(([label, value]) => (
        `<div class="agent-drawer-row"><span class="agent-drawer-label">${escapeHtml(label)}:</span>${escapeHtml(String(value || "-"))}</div>`
      ))
      .join("");
  }

  function closeAgentDrawer() {
    focusedAgentId = null;
    if ($agentDrawer) $agentDrawer.style.display = "none";
  }

  function parseSpatialLookups(map) {
    const props = Array.isArray(map?.properties) ? map.properties : [];
    const rawValue = props.find((item) => item?.name === "spatial_lookups")?.value;
    if (!rawValue) return {};
    if (typeof rawValue === "object") return rawValue || {};
    try {
      return JSON.parse(String(rawValue));
    } catch {
      return {};
    }
  }

  function resolveLocationAtWorld(scene, worldX, worldY) {
    if (!scene || !scene._sectorLayer) return "";
    const tile = scene._sectorLayer.getTileAtWorldXY(worldX, worldY, false, scene.cameras.main);
    if (!tile || tile.index == null || Number(tile.index) <= 0) return "";
    const lookup = scene._sectorLookup || {};
    return String(lookup[String(tile.index)] || "").trim();
  }

  function showLocationTooltip(scene, pointer) {
    if (!$locationTooltip) return;
    const name = resolveLocationAtWorld(scene, pointer.worldX, pointer.worldY);
    if (!name) {
      $locationTooltip.style.display = "none";
      return;
    }
    const container = document.getElementById("game-container");
    const rect = container?.getBoundingClientRect();
    if (!rect) return;
    $locationTooltip.textContent = `📍 ${name}`;
    $locationTooltip.style.display = "block";
    const left = Math.max(8, Math.min(rect.width - 280, Number(pointer.x || 0) + 14));
    const top = Math.max(8, Math.min(rect.height - 28, Number(pointer.y || 0) + 10));
    $locationTooltip.style.left = `${left}px`;
    $locationTooltip.style.top = `${top}px`;
  }

  function occupantsForLocation(locationName) {
    const name = String(locationName || "").trim().toLowerCase();
    if (!name || !latestState) return [];
    return (latestState.npcs || []).filter((npc) => {
      const current = String(npc.current_location || "").trim().toLowerCase();
      if (!current) return false;
      return current === name || current.startsWith(`${name},`) || current.includes(name);
    });
  }

  function openLocationPanel(locationName) {
    const name = String(locationName || "").trim();
    if (!name) return;
    activeLocationName = name;
    renderLocationPanel();
  }

  function closeLocationPanel() {
    activeLocationName = "";
    if ($locationPanel) $locationPanel.style.display = "none";
  }

  function renderLocationPanel() {
    if (!$locationPanel || !$locationPanelTitle || !$locationPanelBody) return;
    if (!activeLocationName) {
      $locationPanel.style.display = "none";
      return;
    }
    const occupants = occupantsForLocation(activeLocationName);
    $locationPanel.style.display = "block";
    $locationPanelTitle.textContent = activeLocationName;
    if (occupants.length === 0) {
      $locationPanelBody.innerHTML = '<div class="empty-state"><p>No agents currently visible in this location.</p></div>';
      return;
    }
    $locationPanelBody.innerHTML = occupants
      .map((npc) => {
        const scenario = npc.scenario ? " · 🎮 in game" : "";
        return `
          <div class="location-agent-row">
            <strong>${escapeHtml(npc.name || npc.agent_id)}</strong>
            <span> · ${escapeHtml(npc.status || "idle")}${scenario}</span>
          </div>
        `;
      })
      .join("");
  }

  function handleMapPointerDown(scene, pointer) {
    const name = resolveLocationAtWorld(scene, pointer.worldX, pointer.worldY);
    if (!name) return;
    openLocationPanel(name);
  }

  function openRelationModal() {
    if (!$relationModal) return;
    $relationModal.style.display = "flex";
    renderRelationshipGraph();
  }

  function closeRelationModal() {
    if ($relationModal) $relationModal.style.display = "none";
  }

  function relationshipEdges(npcs) {
    const ids = new Set((npcs || []).map((npc) => String(npc.agent_id || "")));
    const edges = [];
    (npcs || []).forEach((npc) => {
      const sourceId = String(npc.agent_id || "");
      const rel = npc.memory_summary?.top_relationships || [];
      rel.forEach((item) => {
        const targetId = String(item?.agent_id || "");
        const score = Number(item?.score || 0);
        if (!sourceId || !targetId || !ids.has(targetId) || sourceId === targetId) return;
        edges.push({ sourceId, targetId, score });
      });
    });
    return edges;
  }

  function renderRelationshipGraph() {
    if (!$relationCanvas) return;
    const npcs = latestState?.npcs || [];
    const canvas = $relationCanvas;
    const width = Math.max(320, canvas.clientWidth || 720);
    const height = Math.max(240, canvas.clientHeight || 420);
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0d1a2c";
    ctx.fillRect(0, 0, width, height);
    if (!npcs.length) {
      ctx.fillStyle = "#7a95af";
      ctx.font = "20px VT323";
      ctx.fillText("No relationship data yet.", 18, 34);
      relationGraphNodes = [];
      return;
    }

    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.max(90, Math.min(width, height) * 0.34);
    const nodes = npcs.map((npc, idx) => {
      const angle = (idx / Math.max(1, npcs.length)) * Math.PI * 2 - Math.PI / 2;
      return {
        agentId: String(npc.agent_id || ""),
        name: String(npc.name || npc.agent_id || "Agent"),
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
        emoji: String(npc.pronunciatio || "🙂"),
      };
    });
    relationGraphNodes = nodes;
    const byId = {};
    nodes.forEach((node) => { byId[node.agentId] = node; });

    const edges = relationshipEdges(npcs);
    edges.forEach((edge) => {
      const a = byId[edge.sourceId];
      const b = byId[edge.targetId];
      if (!a || !b) return;
      const score = Math.max(0, Math.min(10, Number(edge.score || 0)));
      const alpha = 0.14 + (score / 10) * 0.5;
      ctx.strokeStyle = `rgba(83,216,251,${alpha.toFixed(3)})`;
      ctx.lineWidth = 1 + (score / 10) * 3;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });

    nodes.forEach((node) => {
      ctx.fillStyle = "#1a3350";
      ctx.strokeStyle = "#53d8fb";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 16, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffd56b";
      ctx.font = "14px VT323";
      ctx.textAlign = "center";
      ctx.fillText(node.emoji, node.x, node.y + 5);
      ctx.fillStyle = "#d8e4ef";
      ctx.font = "12px 'Press Start 2P'";
      ctx.fillText(node.name.slice(0, 14), node.x, node.y + 30);
    });
  }

  function handleRelationCanvasClick(event) {
    if (!$relationCanvas || !relationGraphNodes.length) return;
    const rect = $relationCanvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const sx = $relationCanvas.width / rect.width;
    const sy = $relationCanvas.height / rect.height;
    const x = (event.clientX - rect.left) * sx;
    const y = (event.clientY - rect.top) * sy;
    let nearest = null;
    let bestDist = Infinity;
    relationGraphNodes.forEach((node) => {
      const dx = x - node.x;
      const dy = y - node.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < bestDist) {
        bestDist = d;
        nearest = node;
      }
    });
    if (nearest && bestDist <= 22) {
      closeRelationModal();
      window._focusAgent(nearest.agentId);
    }
  }

  function renderActiveMatches(matches) {
    if (!$matchList) return;
    if (!matches || matches.length === 0) {
      $matchList.innerHTML = '<div class="empty-state"><p>No active games</p></div>';
      return;
    }

    $matchList.innerHTML = matches
      .map((match) => {
        const kind = scenarioKindFromState(match);
        const scenarioLabel = `🎮 ${scenarioDisplayName(match)}`;
        const meta = (kind === "anaconda" || kind === "holdem" || kind === "blackjack")
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
    activeMatchFrames = [];
    matchReplayMode = false;
    matchReplayIndex = 0;
    matchShowRoles = false;
    lastSpectatePhase = "";
    if (matchReplayTimer) {
      clearInterval(matchReplayTimer);
      matchReplayTimer = null;
    }
    if ($matchReplayControls) $matchReplayControls.style.display = "none";
    if ($matchRolesToggle) $matchRolesToggle.textContent = "Roles: Hidden";
    if ($matchReplayToggle) $matchReplayToggle.textContent = "Replay";
    if ($matchReplayLabel) $matchReplayLabel.textContent = "Replay 1/1";
    if ($matchReplaySlider) {
      $matchReplaySlider.min = "0";
      $matchReplaySlider.max = "0";
      $matchReplaySlider.value = "0";
    }
    if ($matchModalBoard) {
      $matchModalBoard.innerHTML = '<div class="empty-state"><p>Loading spectator scene...</p></div>';
    }
    closeMatchTranscriptModal();
    if (!$matchModal) return;
    $matchModal.style.display = "flex";
    pollMatchModal();
    if (spectateTimer) clearInterval(spectateTimer);
    spectateTimer = setInterval(pollMatchModal, SPECTATE_POLL_MS);
  }

  function closeMatchModal() {
    activeSpectateMatchId = null;
    activeMatchFrames = [];
    matchReplayMode = false;
    matchReplayIndex = 0;
    lastSpectatePhase = "";
    if (matchReplayTimer) clearInterval(matchReplayTimer);
    matchReplayTimer = null;
    if (spectateTimer) clearInterval(spectateTimer);
    spectateTimer = null;
    if ($matchTranscriptModal) $matchTranscriptModal.style.display = "none";
    if ($matchModal) $matchModal.style.display = "none";
  }

  function scenarioKindFromState(value) {
    const explicit = String(value?.scenario_kind || "").toLowerCase();
    if (explicit) return explicit;
    const raw = String(typeof value === "string" ? value : (value?.scenario_key || "")).toLowerCase();
    if (raw.includes("werewolf")) return "werewolf";
    if (raw.includes("blackjack")) return "blackjack";
    if (raw.includes("holdem") || raw.includes("hold'em")) return "holdem";
    if (raw.includes("anaconda")) return "anaconda";
    return "generic";
  }

  function scenarioDisplayName(value) {
    const explicit = String(value?.scenario_name || "").trim();
    if (explicit) return explicit;
    const kind = scenarioKindFromState(value);
    if (kind === "werewolf") return "Werewolf";
    if (kind === "anaconda") return "Anaconda";
    if (kind === "blackjack") return "Blackjack";
    if (kind === "holdem") return "Hold'em";
    return "Scenario";
  }

  function currentMatchFrame() {
    if (!activeMatchFrames.length) return null;
    if (matchReplayMode) {
      const idx = Math.max(0, Math.min(activeMatchFrames.length - 1, Number(matchReplayIndex || 0)));
      return activeMatchFrames[idx];
    }
    return activeMatchFrames[activeMatchFrames.length - 1];
  }

  function renderMatchEventRows(events) {
    if (!$matchModalEvents) return;
    const rows = (events || []).slice(-80);
    $matchModalEvents.innerHTML = rows
      .map((evt) => {
        const data = evt.data_json || {};
        const detail = typeof data === "object" ? JSON.stringify(data) : String(data || "");
        return `<div class="match-event-row">s${evt.step} · ${escapeHtml(evt.phase || "")} · ${escapeHtml(evt.event_type || "event")} · ${escapeHtml(detail)}</div>`;
      })
      .join("");
  }

  function latestWerewolfTally(events) {
    const rows = Array.isArray(events) ? events : [];
    for (let i = rows.length - 1; i >= 0; i -= 1) {
      const evt = rows[i];
      if (String(evt?.event_type || "") !== "vote") continue;
      const tally = evt?.data_json?.tally;
      if (tally && typeof tally === "object") return tally;
    }
    return {};
  }

  const scenarioArtRoot = "assets/scenarios/processed";

  function renderMatchVoteTracker(state, events) {
    if (!$matchVoteTracker) return;
    const kind = scenarioKindFromState(state);
    if (kind === "blackjack") {
      const bj = state?.blackjack || {};
      const dealerUp = String(bj.dealer_upcard || (bj.dealer_cards || [])[0] || "--");
      $matchVoteTracker.innerHTML = `
        <div class="match-vote-row">Hand: ${Number(bj.current_hand_index || 1)}/${Number(bj.max_hands || 1)}</div>
        <div class="match-vote-row">Dealer upcard: ${escapeHtml(dealerUp)}</div>
        <div class="match-vote-row"><img class="match-vote-icon" src="${anacondaChipAsset(state?.pot || 0)}" alt="chips" />Pot: ${Number(state?.pot || 0)}</div>
      `;
      return;
    }
    if (kind === "holdem") {
      const hold = state?.holdem || {};
      $matchVoteTracker.innerHTML = `
        <div class="match-vote-row">Board cards: ${Number((hold.board_cards || []).length || 0)}</div>
        <div class="match-vote-row">Current bet: ${Number(hold.current_bet || 0)}</div>
        <div class="match-vote-row"><img class="match-vote-icon" src="${anacondaChipAsset(state?.pot || 0)}" alt="chips" />Pot: ${Number(state?.pot || 0)}</div>
      `;
      return;
    }
    if (kind !== "werewolf") {
      const pot = Number(state?.pot || 0);
      const folded = (state?.folded_agent_ids || []).length;
      const allIn = (state?.all_in_agent_ids || []).length;
      $matchVoteTracker.innerHTML = `
        <div class="match-vote-row"><img class="match-vote-icon" src="${anacondaChipAsset(pot)}" alt="chips" />Pot: ${pot}</div>
        <div class="match-vote-row">Folded: ${folded}</div>
        <div class="match-vote-row">All-in: ${allIn}</div>
      `;
      return;
    }
    const tally = latestWerewolfTally(events);
    const rows = Object.entries(tally).sort((a, b) => Number(b[1]) - Number(a[1]));
    if (!rows.length) {
      $matchVoteTracker.innerHTML = '<div class="match-vote-row">No vote tally yet.</div>';
      return;
    }
    $matchVoteTracker.innerHTML = rows
      .map(([agentId, votes]) => `<div class="match-vote-row"><img class="match-vote-icon" src="${scenarioArtRoot}/werewolf/ui/token_accuse.png" alt="vote token" />${escapeHtml(npcName(agentId))}: ${Number(votes || 0)}</div>`)
      .join("");
  }

  function seatCirclePositions(count) {
    const total = Math.max(1, Number(count || 1));
    const out = [];
    for (let i = 0; i < total; i += 1) {
      const angle = (i / total) * Math.PI * 2 - Math.PI / 2;
      out.push({
        left: 50 + Math.cos(angle) * 36,
        top: 50 + Math.sin(angle) * 34,
      });
    }
    return out;
  }

  function werewolfRoleAsset(role) {
    const normalized = String(role || "").trim().toLowerCase();
    if (!normalized) return "";
    if (normalized.includes("alpha")) return `${scenarioArtRoot}/werewolf/cards/role_alpha.png`;
    if (normalized.includes("werewolf")) return `${scenarioArtRoot}/werewolf/cards/role_werewolf.png`;
    if (normalized.includes("seer")) return `${scenarioArtRoot}/werewolf/cards/role_seer.png`;
    if (normalized.includes("doctor")) return `${scenarioArtRoot}/werewolf/cards/role_doctor.png`;
    if (normalized.includes("hunter")) return `${scenarioArtRoot}/werewolf/cards/role_hunter.png`;
    if (normalized.includes("witch")) return `${scenarioArtRoot}/werewolf/cards/role_witch.png`;
    if (normalized.includes("villager")) return `${scenarioArtRoot}/werewolf/cards/role_villager.png`;
    return "";
  }

  function werewolfPhaseAsset(phase) {
    const normalized = String(phase || "").toLowerCase();
    if (normalized.includes("night")) return `${scenarioArtRoot}/werewolf/ui/phase_night.png`;
    return `${scenarioArtRoot}/werewolf/ui/phase_day.png`;
  }

  function anacondaActionAsset(phase) {
    const normalized = String(phase || "").toLowerCase();
    if (normalized.includes("fold")) return `${scenarioArtRoot}/anaconda/ui/button_fold.png`;
    if (normalized.includes("all_in") || normalized.includes("all-in")) return `${scenarioArtRoot}/anaconda/ui/button_all_in.png`;
    if (normalized.includes("raise")) return `${scenarioArtRoot}/anaconda/ui/button_raise.png`;
    if (normalized.includes("pass")) return `${scenarioArtRoot}/anaconda/ui/button_check.png`;
    if (normalized.includes("discard")) return `${scenarioArtRoot}/anaconda/ui/button_fold.png`;
    if (normalized.includes("showdown")) return `${scenarioArtRoot}/anaconda/ui/button_all_in.png`;
    if (normalized.includes("check")) return `${scenarioArtRoot}/anaconda/ui/button_check.png`;
    if (normalized.includes("deal")) return `${scenarioArtRoot}/anaconda/ui/button_call.png`;
    return `${scenarioArtRoot}/anaconda/ui/button_call.png`;
  }

  function anacondaChipAsset(amount) {
    const value = Number(amount || 0);
    if (value >= 100) return `${scenarioArtRoot}/anaconda/chips/chip_100.png`;
    if (value >= 25) return `${scenarioArtRoot}/anaconda/chips/chip_25.png`;
    if (value >= 10) return `${scenarioArtRoot}/anaconda/chips/chip_10.png`;
    if (value >= 5) return `${scenarioArtRoot}/anaconda/chips/chip_5.png`;
    return `${scenarioArtRoot}/anaconda/chips/chip_1.png`;
  }

  const anacondaRankByCategory = {
    1: "rank_high_card",
    2: "rank_one_pair",
    3: "rank_two_pair",
    4: "rank_three_kind",
    5: "rank_straight",
    6: "rank_flush",
    7: "rank_full_house",
    8: "rank_four_kind",
    9: "rank_straight_flush",
    10: "rank_royal_flush",
  };

  function anacondaRankAsset(state) {
    const showdownRows = state?.showdown?.rows || {};
    const bestCategory = Object.values(showdownRows).reduce((max, row) => {
      const category = Number(row?.category || 0);
      return Number.isFinite(category) ? Math.max(max, category) : max;
    }, 0);
    if (bestCategory > 0) {
      const key = anacondaRankByCategory[bestCategory] || "rank_high_card";
      return `${scenarioArtRoot}/anaconda/ui/${key}.png`;
    }
    const phase = String(state?.phase || "").toLowerCase();
    if (phase.includes("showdown")) return `${scenarioArtRoot}/anaconda/ui/rank_royal_flush.png`;
    if (phase.includes("reveal5")) return `${scenarioArtRoot}/anaconda/ui/rank_flush.png`;
    if (phase.includes("reveal4")) return `${scenarioArtRoot}/anaconda/ui/rank_straight.png`;
    if (phase.includes("reveal3")) return `${scenarioArtRoot}/anaconda/ui/rank_two_pair.png`;
    if (phase.includes("reveal2")) return `${scenarioArtRoot}/anaconda/ui/rank_one_pair.png`;
    return `${scenarioArtRoot}/anaconda/ui/rank_high_card.png`;
  }

  function anacondaCardAsset(card) {
    const raw = String(card || "").trim().toUpperCase();
    if (raw.length !== 2) return "";
    const rank = raw[0];
    const suit = raw[1];
    if (!"23456789TJQKA".includes(rank) || !"CDHS".includes(suit)) return "";
    return `${scenarioArtRoot}/anaconda/cards/faces/${rank}${suit}.png`;
  }

  function anacondaCardToken(card) {
    const raw = String(card || "").trim().toUpperCase();
    const rank = raw.length === 2 ? raw[0] : "?";
    const suit = raw.length === 2 ? raw[1] : "?";
    const suitMap = {
      S: { symbol: "♠", cls: "spade" },
      H: { symbol: "♥", cls: "heart" },
      D: { symbol: "♦", cls: "diamond" },
      C: { symbol: "♣", cls: "club" },
    };
    const icon = suitMap[suit] || { symbol: "?", cls: "unknown" };
    const rankLabel = rank === "T" ? "10" : rank;
    const asset = anacondaCardAsset(raw);
    if (!asset) {
      return `<span class="ana-card-face ana-suit-${icon.cls}">${escapeHtml(rankLabel)}${icon.symbol}</span>`;
    }
    return `<span class="ana-card-face-wrap"><img class="ana-card-face-image" src="${asset}" alt="${escapeHtml(raw)}" loading="lazy" decoding="async" onerror="this.style.display='none';this.nextElementSibling.style.display='inline-flex';" /><span class="ana-card-face ana-suit-${icon.cls}" style="display:none">${escapeHtml(rankLabel)}${icon.symbol}</span></span>`;
  }

  function renderWerewolfBoard(state, events) {
    if (!$matchModalBoard) return;
    const participants = state?.participants || [];
    const alive = new Set((state?.alive_agent_ids || []).map((v) => String(v)));
    const tally = latestWerewolfTally(events);
    const seats = seatCirclePositions(participants.length);
    const cards = participants
      .map((p, idx) => {
        const aid = String(p.agent_id || "");
        const pos = seats[idx] || { left: 50, top: 50 };
        const eliminated = alive.size > 0 && !alive.has(aid);
        const roleVisible = (matchShowRoles && p.role) || String(state?.status || "") === "resolved";
        const roleValue = roleVisible && p.role ? String(p.role) : "?";
        const roleAsset = roleVisible ? werewolfRoleAsset(roleValue) : "";
        const votes = Number(tally[aid] || 0);
        return `
          <div class="ww-seat ${eliminated ? "eliminated" : ""}" style="left:${pos.left}%;top:${pos.top}%;">
            <div class="ww-avatar">${escapeHtml(String(npcName(aid)).slice(0, 1).toUpperCase() || "?")}</div>
            <div class="ww-name">${escapeHtml(npcName(aid))}</div>
            <div class="ww-role-wrap">
              ${roleAsset
                ? `<img class="ww-role-card" src="${roleAsset}" alt="${escapeHtml(roleValue)}" />`
                : `<div class="ww-role">${escapeHtml(roleValue || "?")}</div>`}
            </div>
            <div class="ww-votes">votes ${votes}</div>
            ${eliminated ? `<img class="ww-tombstone" src="${scenarioArtRoot}/werewolf/ui/tombstone.png" alt="eliminated" />` : ""}
          </div>
        `;
      })
      .join("");
    const phaseLabel = String(state?.phase || "").toUpperCase();
    $matchModalBoard.innerHTML = `
      <div class="scenario-board werewolf-board">
        <div class="ww-overlay"></div>
        <img class="ww-phase-art" src="${werewolfPhaseAsset(state?.phase)}" alt="phase token" />
        <img class="ww-mode-icon" src="${scenarioArtRoot}/werewolf/icon.png" alt="werewolf icon" />
        <div class="ww-campfire"><img src="${scenarioArtRoot}/werewolf/board/campfire_circle.png" alt="campfire" /></div>
        <div class="ww-phase">${escapeHtml(phaseLabel || "WEREWOLF")}</div>
        ${cards}
      </div>
    `;
  }

  function renderAnacondaBoard(state) {
    if (!$matchModalBoard) return;
    const participants = state?.participants || [];
    const revealed = state?.revealed_cards || {};
    const seats = seatCirclePositions(participants.length);
    const cards = participants
      .map((p, idx) => {
        const aid = String(p.agent_id || "");
        const pos = seats[idx] || { left: 50, top: 50 };
        const chips = p.chips_end != null ? Number(p.chips_end) : Number(p.chips || 0);
        const hand = Array.isArray(revealed[aid]) ? revealed[aid] : [];
        const displayCards = hand.length
          ? `<span class="ana-revealed-cards">${hand.slice(0, 5).map((c) => anacondaCardToken(c)).join("")}</span>`
          : `<span class="ana-hidden-cards"><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" /></span>`;
        return `
          <div class="ana-seat ${String(p.status || "").toLowerCase().includes("fold") ? "folded" : ""}" style="left:${pos.left}%;top:${pos.top}%;">
            <div class="ana-name">${escapeHtml(npcName(aid))}</div>
            <div class="ana-cards">${displayCards}</div>
            <div class="ana-chips"><img class="ana-chip-icon" src="${anacondaChipAsset(chips)}" alt="chips" />${chips}</div>
          </div>
        `;
      })
      .join("");
    const pot = Number(state?.pot || 0);
    $matchModalBoard.innerHTML = `
      <div class="scenario-board anaconda-board">
        <img class="ana-table-art" src="${scenarioArtRoot}/anaconda/board/poker_table.png" alt="poker table" />
        <div class="ana-table"></div>
        <img class="ana-phase-art" src="${anacondaActionAsset(state?.phase)}" alt="betting action" />
        <img class="ana-ranks-art" src="${anacondaRankAsset(state)}" alt="rank badge" />
        <div class="ana-pot"><img src="${anacondaChipAsset(pot)}" alt="chips" />Pot ${pot}</div>
        <div class="ana-phase">${escapeHtml(String(state?.phase || "").toUpperCase())}</div>
        ${cards}
      </div>
    `;
  }

  function holdemHoleToken(card) {
    const raw = String(card || "").trim().toUpperCase();
    if (!raw || raw === "??") {
      return `<img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" />`;
    }
    return anacondaCardToken(raw);
  }

  function renderBlackjackBoard(state) {
    if (!$matchModalBoard) return;
    const participants = state?.participants || [];
    const seats = seatCirclePositions(participants.length);
    const bj = state?.blackjack || {};
    const hands = bj.player_hands || {};
    const bets = bj.player_bets || {};
    const dealerCards = Array.isArray(bj.dealer_cards) ? bj.dealer_cards : [];
    const dealerTotal = dealerCards.filter((c) => c !== "??").length
      ? dealerCards.filter((c) => c !== "??").map((c) => String(c || "")).length
      : 0;
    const cardsHtml = participants
      .map((p, idx) => {
        const aid = String(p.agent_id || "");
        const pos = seats[idx] || { left: 50, top: 50 };
        const chips = p.chips_end != null ? Number(p.chips_end) : Number(p.chips || 0);
        const hand = Array.isArray(hands[aid]) ? hands[aid] : [];
        const rendered = hand.length
          ? hand.map((card) => (String(card || "").toUpperCase() === "??"
            ? `<img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" />`
            : anacondaCardToken(card))).join("")
          : `<img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" />`;
        return `
          <div class="ana-seat" style="left:${pos.left}%;top:${pos.top}%;">
            <div class="ana-name">${escapeHtml(npcName(aid))}</div>
            <div class="ana-cards"><span class="ana-revealed-cards">${rendered}</span></div>
            <div class="ana-chips"><img class="ana-chip-icon" src="${anacondaChipAsset(chips)}" alt="chips" />${chips} · bet ${Number(bets[aid] || 0)}</div>
          </div>
        `;
      })
      .join("");

    const dealerRendered = dealerCards.length
      ? dealerCards.map((card) => (String(card || "").toUpperCase() === "??"
        ? `<img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" />`
        : anacondaCardToken(card))).join("")
      : `<img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden card" />`;

    $matchModalBoard.innerHTML = `
      <div class="scenario-board anaconda-board">
        <img class="ana-table-art" src="${scenarioArtRoot}/anaconda/board/poker_table.png" alt="table" />
        <img class="ana-phase-art" src="${scenarioArtRoot}/blackjack/ui/actions_sheet.png" alt="blackjack actions" onerror="this.onerror=null;this.src='${scenarioArtRoot}/anaconda/ui/button_call.png';" />
        <img class="ana-ranks-art" src="${scenarioArtRoot}/blackjack/icon.png" alt="blackjack icon" onerror="this.onerror=null;this.src='${scenarioArtRoot}/anaconda/icon.png';" />
        <div class="ana-pot"><img src="${anacondaChipAsset(state?.pot || 0)}" alt="chips" />Pot ${Number(state?.pot || 0)}</div>
        <div class="ana-phase">${escapeHtml(`HAND ${Number(bj.current_hand_index || 1)}/${Number(bj.max_hands || 1)}`)}</div>
        <div class="bj-dealer">
          <div class="ana-name">Dealer</div>
          <div class="ana-cards"><span class="ana-revealed-cards">${dealerRendered}</span></div>
          <div class="ana-chips">cards ${dealerCards.length}${dealerTotal ? ` · showing` : ""}</div>
        </div>
        ${cardsHtml}
      </div>
    `;
  }

  function renderHoldemBoard(state) {
    if (!$matchModalBoard) return;
    const participants = state?.participants || [];
    const seats = seatCirclePositions(participants.length);
    const holdem = state?.holdem || {};
    const holeCards = holdem.hole_cards || {};
    const boardCards = Array.isArray(holdem.board_cards) ? holdem.board_cards : [];
    const dealerId = String(holdem.button_agent_id || "");
    const smallBlindId = String(holdem.small_blind_agent_id || "");
    const bigBlindId = String(holdem.big_blind_agent_id || "");

    const cardsHtml = participants
      .map((p, idx) => {
        const aid = String(p.agent_id || "");
        const pos = seats[idx] || { left: 50, top: 50 };
        const chips = p.chips_end != null ? Number(p.chips_end) : Number(p.chips || 0);
        const hand = Array.isArray(holeCards[aid]) ? holeCards[aid] : ["??", "??"];
        const marker = aid === dealerId ? "D" : (aid === smallBlindId ? "SB" : (aid === bigBlindId ? "BB" : ""));
        return `
          <div class="ana-seat ${String(p.status || "").toLowerCase().includes("fold") ? "folded" : ""}" style="left:${pos.left}%;top:${pos.top}%;">
            <div class="ana-name">${escapeHtml(npcName(aid))} ${marker ? `<span class="holdem-marker">${marker}</span>` : ""}</div>
            <div class="ana-cards"><span class="ana-revealed-cards">${hand.slice(0, 2).map((card) => holdemHoleToken(card)).join("")}</span></div>
            <div class="ana-chips"><img class="ana-chip-icon" src="${anacondaChipAsset(chips)}" alt="chips" />${chips}</div>
          </div>
        `;
      })
      .join("");
    const boardTokens = boardCards.length
      ? boardCards.map((card) => anacondaCardToken(card)).join("")
      : `<img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden board" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden board" /><img class="ana-card-back" src="${scenarioArtRoot}/anaconda/cards/card_back.png" alt="hidden board" />`;

    $matchModalBoard.innerHTML = `
      <div class="scenario-board anaconda-board">
        <img class="ana-table-art" src="${scenarioArtRoot}/anaconda/board/poker_table.png" alt="table" />
        <img class="ana-phase-art" src="${scenarioArtRoot}/holdem/ui/markers_sheet.png" alt="holdem markers" onerror="this.onerror=null;this.src='${scenarioArtRoot}/shared/frames/badge_silver.png';" />
        <img class="ana-ranks-art" src="${scenarioArtRoot}/holdem/icon.png" alt="holdem icon" onerror="this.onerror=null;this.src='${scenarioArtRoot}/anaconda/icon.png';" />
        <div class="ana-pot"><img src="${anacondaChipAsset(state?.pot || 0)}" alt="chips" />Pot ${Number(state?.pot || 0)}</div>
        <div class="ana-phase">${escapeHtml(String(state?.phase || "").toUpperCase())}</div>
        <div class="holdem-board-cards">${boardTokens}</div>
        ${cardsHtml}
      </div>
    `;
  }

  function showMatchSceneBanner(text) {
    if (!$matchSceneBanner) return;
    $matchSceneBanner.textContent = text;
    $matchSceneBanner.classList.remove("show");
    window.requestAnimationFrame(() => {
      $matchSceneBanner.classList.add("show");
      window.setTimeout(() => $matchSceneBanner.classList.remove("show"), 1800);
    });
  }

  function renderMatchBoard(state, events) {
    const kind = scenarioKindFromState(state);
    if (kind === "werewolf") renderWerewolfBoard(state, events);
    else if (kind === "anaconda") renderAnacondaBoard(state);
    else if (kind === "blackjack") renderBlackjackBoard(state);
    else if (kind === "holdem") renderHoldemBoard(state);
    else if ($matchModalBoard) {
      $matchModalBoard.innerHTML = '<div class="empty-state"><p>No dedicated scene for this match type.</p></div>';
    }
  }

  function renderMatchPlayers(state) {
    if (!$matchModalPlayers) return;
    const winnerSet = new Set((state?.result?.winners || []).map((v) => String(v)));
    const participants = state?.participants || [];
    $matchModalPlayers.innerHTML = participants
      .map((p) => {
        const role = (matchShowRoles || String(state?.status || "") === "resolved") && p.role ? ` (${p.role})` : "";
        const chips = p.chips_end != null ? ` · chips ${p.chips_end}` : ` · chips ${p.chips ?? 0}`;
        const delta = Number(p.rating_delta || 0);
        const deltaText = Number.isFinite(delta) && delta !== 0 ? ` · rating ${delta > 0 ? "+" : ""}${delta}` : "";
        const trophy = winnerSet.has(String(p.agent_id || "")) ? "🏆 " : "";
        return `<div class="match-player-row">${trophy}${escapeHtml(npcName(p.agent_id))}${role} · ${escapeHtml(p.status || "")}${chips}${deltaText}</div>`;
      })
      .join("");
  }

  function pushMatchFrame(payload) {
    const frame = {
      public_state: cloneJsonSafe(payload?.public_state || {}),
      recent_events: cloneJsonSafe(payload?.recent_events || []),
      capturedAt: Date.now(),
    };
    const prev = activeMatchFrames[activeMatchFrames.length - 1];
    const prevState = prev?.public_state || {};
    const state = frame.public_state || {};
    const sameFrame = (
      String(prevState.phase || "") === String(state.phase || "")
      && Number(prevState.round_number || 0) === Number(state.round_number || 0)
      && Number((prev?.recent_events || []).length) === Number((frame.recent_events || []).length)
      && Number(prevState.pot || 0) === Number(state.pot || 0)
    );
    if (!sameFrame) activeMatchFrames.push(frame);
    else activeMatchFrames[activeMatchFrames.length - 1] = frame;
    if (activeMatchFrames.length > 240) {
      activeMatchFrames = activeMatchFrames.slice(activeMatchFrames.length - 240);
    }
    if ($matchReplaySlider) {
      $matchReplaySlider.max = String(Math.max(0, activeMatchFrames.length - 1));
      if (!matchReplayMode) $matchReplaySlider.value = String(Math.max(0, activeMatchFrames.length - 1));
    }
  }

  function renderMatchReplayFrame(index) {
    if (!activeMatchFrames.length) return;
    matchReplayIndex = Math.max(0, Math.min(activeMatchFrames.length - 1, Number(index || 0)));
    if ($matchReplaySlider) $matchReplaySlider.value = String(matchReplayIndex);
    if ($matchReplayLabel) $matchReplayLabel.textContent = `Replay ${matchReplayIndex + 1}/${activeMatchFrames.length}`;
    const frame = currentMatchFrame();
    if (frame) renderSpectatorFrame(frame);
  }

  function stopMatchReplayPlayback() {
    if (matchReplayTimer) clearInterval(matchReplayTimer);
    matchReplayTimer = null;
    if ($matchReplayPlay) $matchReplayPlay.textContent = "Play";
  }

  function toggleMatchReplayPlayback() {
    if (!matchReplayMode) return;
    if (matchReplayTimer) {
      stopMatchReplayPlayback();
      return;
    }
    if (matchReplayIndex >= activeMatchFrames.length - 1) {
      renderMatchReplayFrame(0);
    }
    if ($matchReplayPlay) $matchReplayPlay.textContent = "Pause";
    matchReplayTimer = window.setInterval(() => {
      if (matchReplayIndex >= activeMatchFrames.length - 1) {
        stopMatchReplayPlayback();
        return;
      }
      renderMatchReplayFrame(matchReplayIndex + 1);
    }, 800);
  }

  function setMatchReplayMode(enabled) {
    matchReplayMode = Boolean(enabled);
    stopMatchReplayPlayback();
    if ($matchReplayControls) $matchReplayControls.style.display = matchReplayMode ? "grid" : "none";
    if ($matchReplayToggle) $matchReplayToggle.textContent = matchReplayMode ? "Replay: On" : "Replay";
    if (matchReplayMode) {
      renderMatchReplayFrame(Math.max(0, activeMatchFrames.length - 1));
    } else {
      matchReplayIndex = Math.max(0, activeMatchFrames.length - 1);
      if ($matchReplaySlider) $matchReplaySlider.value = String(matchReplayIndex);
      const frame = currentMatchFrame();
      if (frame) renderSpectatorFrame(frame);
    }
  }

  function openMatchTranscriptModal() {
    if (!$matchTranscriptModal || !$matchTranscriptBody) return;
    const frame = currentMatchFrame();
    const events = frame?.recent_events || [];
    if (!events.length) {
      $matchTranscriptBody.innerHTML = '<div class="empty-state"><p>No transcript yet.</p></div>';
      $matchTranscriptModal.style.display = "flex";
      return;
    }
    $matchTranscriptBody.innerHTML = events
      .map((evt) => {
        const data = evt.data_json || {};
        const detail = typeof data === "object" ? JSON.stringify(data) : String(data || "");
        return `<div class="match-event-row">s${evt.step} · ${escapeHtml(evt.phase || "")} · ${escapeHtml(evt.event_type || "")} · ${escapeHtml(detail)}</div>`;
      })
      .join("");
    $matchTranscriptModal.style.display = "flex";
  }

  function closeMatchTranscriptModal() {
    if ($matchTranscriptModal) $matchTranscriptModal.style.display = "none";
  }

  function renderSpectatorFrame(frame) {
    const state = frame?.public_state || {};
    const events = frame?.recent_events || [];
    const scenarioLabel = scenarioDisplayName(state);
    if ($matchModalTitle) {
      $matchModalTitle.textContent = `${scenarioLabel} · ${state.match_id || ""}`;
    }
    if ($matchModalPhase) {
      const reason = String(state?.result?.reason || "");
      const suffix = reason ? ` · ${reason}` : "";
      $matchModalPhase.textContent = `Status ${state.status || "--"} · Phase ${state.phase || "--"} · Round ${state.round_number ?? "--"}${suffix}`;
    }
    const phaseKey = `${state.status || ""}:${state.phase || ""}:${state.round_number || ""}`;
    if (phaseKey !== lastSpectatePhase) {
      lastSpectatePhase = phaseKey;
      showMatchSceneBanner(`${scenarioLabel.toUpperCase()} · ${(state.phase || "phase").toUpperCase()}`);
    }
    renderMatchPlayers(state);
    renderMatchEventRows(events);
    renderMatchVoteTracker(state, events);
    renderMatchBoard(state, events);
  }

  async function pollMatchModal() {
    if (!activeSpectateMatchId) return;
    try {
      const payload = await apiFetch(`/scenarios/matches/${activeSpectateMatchId}/spectate`);
      if (!activeSpectateMatchId) return;
      pushMatchFrame(payload);
      if (!matchReplayMode) {
        const frame = currentMatchFrame();
        if (frame) renderSpectatorFrame(frame);
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
    const sectorLayer = map.createLayer("Sector Blocks", [collisions, blocks2, blocks3], 0, 0);
    if (sectorLayer) {
      sectorLayer.setVisible(false);
      sectorLayer.setDepth(-2);
    }
    this._sectorLayer = sectorLayer || null;
    const spatial = parseSpatialLookups(map);
    this._sectorLookup = (spatial && typeof spatial.sector_ids === "object" && spatial.sector_ids) || {};

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

    if (!this.textures.exists("night_firefly")) {
      const dot = this.add.graphics();
      dot.fillStyle(0xffe8a3, 1);
      dot.fillCircle(2, 2, 2);
      dot.generateTexture("night_firefly", 4, 4);
      dot.destroy();
    }
    const fireflyManager = this.add.particles("night_firefly").setDepth(997).setScrollFactor(0);
    const fireflyEmitter = fireflyManager.createEmitter({
      x: { min: 0, max: camera.width },
      y: { min: 0, max: camera.height },
      lifespan: { min: 2600, max: 5400 },
      speedY: { min: -14, max: -5 },
      speedX: { min: -8, max: 8 },
      scale: { start: 0.7, end: 0.05 },
      alpha: { start: 0.38, end: 0 },
      quantity: 1,
      frequency: 280,
      blendMode: "ADD",
    });
    fireflyEmitter.stop();
    this._nightParticles = {
      manager: fireflyManager,
      emitter: fireflyEmitter,
      active: false,
    };

    this._cursors = this.input.keyboard.createCursorKeys();
    this._keys = this.input.keyboard.addKeys("W,A,S,D,PLUS,MINUS");

    this.input.on("wheel", (pointer, gameObjects, deltaX, deltaY) => {
      const newZoom = Phaser.Math.Clamp(camera.zoom - deltaY * 0.001, 0.3, 3);
      camera.setZoom(newZoom);
    });
    this.input.on("pointermove", (pointer) => {
      showLocationTooltip(this, pointer);
    });
    this.input.on("pointerdown", (pointer) => {
      handleMapPointerDown(this, pointer);
    });
    this.input.on("gameout", () => {
      if ($locationTooltip) $locationTooltip.style.display = "none";
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
    startTownDataFlow();
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

    const particlePack = scene?._nightParticles;
    if (particlePack?.emitter) {
      const shouldGlow = hour >= 20 || hour < 7;
      if (shouldGlow && !particlePack.active) {
        particlePack.emitter.start();
        particlePack.active = true;
      } else if (!shouldGlow && particlePack.active) {
        particlePack.emitter.stop();
        particlePack.active = false;
      }
    }
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

    stopTownDataFlow();
    closeMatchModal();
    closeConversationModal();
    closeLocationPanel();
    closeReplayModal();
    closeRelationModal();
    if ($locationTooltip) $locationTooltip.style.display = "none";
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
    conversationHistory = {};
    activeConversationId = null;
    timelineSnapshots = [];
    latestLiveSnapshot = null;
    replayMode = false;
    replayIndex = 0;
    stopReplayPlayback();
    activeBubbles.forEach((bubble) => bubble?.destroy?.());
    activeBubbles = [];
    closeAgentDrawer();

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
    focusedAgentId = String(agentId);
    if (gameScene && agentSpriteMap[agentId]) {
      const sprite = agentSpriteMap[agentId].sprite;
      gameScene.cameras.main.pan(sprite.x, sprite.y, 500, "Sine.easeInOut");
    }
    document.querySelectorAll(".agent-card").forEach((el) => el.classList.remove("focused"));
    const card = document.querySelector(`.agent-card[data-agent-id="${agentId}"]`);
    if (card) card.classList.add("focused");
    renderAgentDrawer();
  };
  window._showConvoDetail = function (sessionId) {
    openConversationModal(sessionId);
  };

  $townSelect.addEventListener("change", (e) => selectTown(e.target.value));
  $btnRefreshTowns.addEventListener("click", loadTowns);
  $btnTick.addEventListener("click", runTick);
  $btnAutoTick.addEventListener("click", toggleAutoTick);
  $btnReplay?.addEventListener("click", openReplayModal);
  $btnRelations?.addEventListener("click", openRelationModal);
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
  $matchRolesToggle?.addEventListener("click", () => {
    matchShowRoles = !matchShowRoles;
    if ($matchRolesToggle) $matchRolesToggle.textContent = matchShowRoles ? "Roles: Visible" : "Roles: Hidden";
    const frame = currentMatchFrame();
    if (frame) renderSpectatorFrame(frame);
  });
  $matchReplayToggle?.addEventListener("click", () => {
    if (!activeMatchFrames.length) return;
    setMatchReplayMode(!matchReplayMode);
  });
  $matchReplayPlay?.addEventListener("click", toggleMatchReplayPlayback);
  $matchReplaySlider?.addEventListener("input", (event) => {
    if (!activeMatchFrames.length) return;
    if (!matchReplayMode) setMatchReplayMode(true);
    renderMatchReplayFrame(Number(event?.target?.value || 0));
  });
  $matchReplayLive?.addEventListener("click", () => setMatchReplayMode(false));
  $matchTranscriptBtn?.addEventListener("click", openMatchTranscriptModal);
  $matchTranscriptClose?.addEventListener("click", closeMatchTranscriptModal);
  $matchTranscriptModal?.addEventListener("click", (event) => {
    if (event.target === $matchTranscriptModal) closeMatchTranscriptModal();
  });
  $convoModalClose?.addEventListener("click", closeConversationModal);
  $convoModal?.addEventListener("click", (event) => {
    if (event.target === $convoModal) closeConversationModal();
  });
  $replayModalClose?.addEventListener("click", closeReplayModal);
  $replayModal?.addEventListener("click", (event) => {
    if (event.target === $replayModal) closeReplayModal();
  });
  $replaySlider?.addEventListener("input", (event) => {
    if (!timelineSnapshots.length) return;
    replayMode = true;
    stopReplayPlayback();
    renderReplayFrame(Number(event?.target?.value || 0));
  });
  $replayPlay?.addEventListener("click", toggleReplayPlayback);
  $replayLive?.addEventListener("click", replayToLiveState);
  $relationModalClose?.addEventListener("click", closeRelationModal);
  $relationModal?.addEventListener("click", (event) => {
    if (event.target === $relationModal) closeRelationModal();
  });
  $relationCanvas?.addEventListener("click", handleRelationCanvasClick);
  $convoModalJump?.addEventListener("click", jumpToConversationAgents);
  $agentDrawerClose?.addEventListener("click", closeAgentDrawer);
  $locationPanelClose?.addEventListener("click", closeLocationPanel);

  const urlTown = new URLSearchParams(window.location.search).get("town");

  loadTowns().then(() => {
    if (urlTown) {
      $townSelect.value = urlTown;
      selectTown(urlTown);
    }
  });
})();
