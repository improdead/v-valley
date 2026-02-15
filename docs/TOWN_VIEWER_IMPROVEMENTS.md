<div align="center">

# 🏘️ V-Valley Town Viewer

### *Make Watching the Simulation Feel Like Living In It*

---

**Status** `Implemented (P0 Core + Partial P1)` · **Last Updated** `2026-02-15`  
**Design North Star:** *When a spectator opens the town viewer, they should feel like they're peering through the window of a tiny living world — not polling a dashboard.*

</div>

---

## ✦ The Problem Today

Right now, opening the Town Viewer feels like opening a monitoring tool. Agents silently slide around the map. Conversations show `agent_abc123 & agent_def456` instead of names. There's no narrative — no sense of *what just happened* or *why it matters*. The map doesn't breathe. No day/night, no speech, no story.

Most of this can be fixed on the frontend, with small backend additions for scenario match visibility.

---

## ✦ Priority System

| Tag | What It Means |
|:---:|---|
| 🔴 **P0** | Must-have — the viewer feels broken without it |
| 🟡 **P1** | Should-have — significant experience upgrade |
| 🟢 **P2** | Nice-to-have — polish and delight |

---

## ✅ Current Implementation Snapshot (As Of 2026-02-15)

### Shipped now

- Live event feed with client-side diffing and ring buffer
- On-map speech/thought bubbles (conversation lines + pronunciatio updates)
- Phaser lifecycle cleanup on town switch
- Overlapping poll prevention + stronger `apiFetch` error handling
- Connection health indicator (`Connected` / `Polling` / `Disconnected`)
- Conversation cards now show agent names (not raw IDs)
- Day/night visual overlay tied to simulation clock
- Active games section in sidebar with `Watch Live` spectator modal
- Scenario markers on agent cards and map labels for in-match agents
- Zoom controls (`+` / `-`) and responsive layout improvements
- Landing-page town card deep-link fix (`town.html?api=...&town=...`)

### Still pending from this plan

- Conversation transcript detail modal
- Agent detail drawer with memory/relationship deep dive
- Full location interaction layer from Tiled location objects
- SSE streaming replacement for polling
- Relationship graph, replay/rewind, richer ambient polish

---

## 🔴 1 · Live Event Feed — *The Heartbeat*

> **Priority** P0 · **Effort** M · **Backend changes** None

### The Vision

A scrolling ticker above the agent list in the sidebar, narrating the world in real-time. Every movement, conversation, and arrival becomes a line in a living story.

```
 ╔══════════════════════════════════════╗
 ║  📋 EVENTS                          ║
 ╠══════════════════════════════════════╣
 ║  💬 Isabella & Klaus started talking ║
 ║     @ Hobbs Cafe · step 42          ║
 ║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ║
 ║  🚶 Carlos moved to Town Hall       ║
 ║     from Oak Hill College · step 41  ║
 ║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ║
 ║  🌅 Maria woke up                   ║
 ║     starting morning routine · s41   ║
 ║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ║
 ║  🎉 Eddy joined the town!           ║
 ║     spawned at park entrance · s40   ║
 ╚══════════════════════════════════════╝
```

### Why It Transforms the Experience

Without the feed, spectators see movement but have **zero narrative**. They don't know *what happened* between polls. The feed turns passive watching into active following — suddenly you're reading a story, not staring at a screensaver.

### Color Language

| Event Type | Color Accent | Emoji Class |
|---|---|---|
| Social (conversation, chat) | `--accent-purple` | 💬 💕 🤝 |
| Movement | `--accent-cyan` | 🚶 🏃 🚪 |
| Arrival / Departure | `--accent-green` | 🎉 👋 🌅 |
| Goal / Decision | `--accent-gold` | 🎯 💡 ⭐ |

### How It Works — Client-Side State Diffing

No backend changes needed. We diff the current state against the previous poll:

```javascript
// Client-side state diff engine
let prevNpcs = {};   // agent_id → { x, y, status, current_location, goal_reason }
let prevConvos = {}; // session_id → { turns, last_message }
let eventLog = [];   // ring buffer, max 200 entries

function diffState(newState) {
  const events = [];
  for (const npc of newState.npcs) {
    const prev = prevNpcs[npc.agent_id];
    if (!prev) {
      events.push({
        type: 'arrival', agent: npc.name, emoji: npc.pronunciatio,
        detail: `joined the town`, step: newState.step
      });
    } else {
      if (prev.current_location !== npc.current_location && npc.current_location) {
        events.push({
          type: 'movement', agent: npc.name, emoji: npc.pronunciatio,
          detail: `moved to ${npc.current_location}`, step: newState.step
        });
      }
      if (prev.goal_reason !== npc.goal?.reason && npc.goal?.reason) {
        events.push({
          type: 'goal', agent: npc.name, emoji: npc.pronunciatio,
          detail: npc.goal.reason, step: newState.step
        });
      }
    }
    prevNpcs[npc.agent_id] = { ...npc };
  }
  // Conversation diffs
  for (const convo of (newState.conversations || [])) {
    if (!prevConvos[convo.session_id]) {
      events.push({
        type: 'social',
        detail: `${convo.agent_a} & ${convo.agent_b} started talking`,
        step: newState.step
      });
    }
  }
  return events;
}
```

### UI Implementation

- Add `<div id="eventFeed">` section above `#agentList` in the sidebar
- Style with the existing `.convo-card` pattern but smaller (8px padding), with a color-coded **left border**
- Max 100 events in buffer; auto-scroll to bottom
- Manual scroll up → pauses auto-scroll (show a "↓ New events" pill at the bottom)
- Click any event → camera pans to that agent via `window._focusAgent()`

---

## 🔴 2 · On-Map Speech Bubbles — *The Voices*

> **Priority** P0 · **Effort** M · **Backend changes** None

### The Vision

When agents are in conversation, a pixel-art speech bubble floats above them showing their latest line. Idle agents show their pronunciatio emoji in a small thought bubble.

**This single change makes the map feel alive.** Without it, agents are just sprites sliding around silently. With it, you can *see* social dynamics playing out spatially.

### Implementation

```javascript
// Call in updateAgentSprites(), after position update
function showSpeechBubble(scene, sprite, text, duration = 8000) {
  const padding = 8;
  const maxWidth = 180;
  const bubbleText = scene.add.text(0, 0, text, {
    font: '13px VT323',
    color: '#1a2d48',
    wordWrap: { width: maxWidth },
    padding: { x: 4, y: 2 },
  }).setOrigin(0.5, 1).setDepth(20);

  const bounds = bubbleText.getBounds();
  const bubble = scene.add.graphics().setDepth(19);
  bubble.fillStyle(0xfaf6e8, 0.95);      // cream background
  bubble.lineStyle(2, 0x2a4a6b, 1);       // pixel border
  bubble.fillRoundedRect(
    bounds.x - padding, bounds.y - padding,
    bounds.width + padding * 2, bounds.height + padding * 2, 4
  );
  bubble.strokeRoundedRect(
    bounds.x - padding, bounds.y - padding,
    bounds.width + padding * 2, bounds.height + padding * 2, 4
  );

  // Position above sprite
  const container = scene.add.container(
    sprite.x, sprite.y - 64, [bubble, bubbleText]
  );
  container.setDepth(20);

  // Fade out gracefully
  scene.tweens.add({
    targets: container, alpha: 0,
    delay: duration - 1000, duration: 1000,
    onComplete: () => container.destroy()
  });
}
```

### Trigger Rules

- Fire when `conversation.last_message` changes between polls
- Show pronunciatio emoji as a small floating thought bubble (24×24) above idle agents
- Limit to **2–3 simultaneous bubbles** per screen to avoid visual clutter
- Prioritize active conversations over idle emotes

---

## 🔴 3 · Bug Fixes & Performance — *The Foundation*

> **Priority** P0 · **Effort** M · **Found by code review of `town.js`**

These are bugs and performance issues found in the current implementation that degrade the experience. Fix these before adding features.

### 3.1 Phaser Lifecycle Leak

**Problem:** When switching towns, `createPhaserGame()` calls `phaserGame.destroy(true)` before creating a new instance — but `selectTown()` doesn't clear `agentSpriteMap` references to the destroyed Phaser objects, and `pollTimer` / `autoTickTimer` may still fire during the transition.

**Fix:**

```javascript
function selectTown(townId) {
  if (!townId) return;
  stopPolling();
  if (autoTickTimer) { clearInterval(autoTickTimer); autoTickTimer = null; }

  // DESTROY old game instance fully
  if (phaserGame) {
    phaserGame.destroy(true);
    phaserGame = null;
    gameScene = null;
  }

  agentSpriteMap = {};
  agentCharMap = {};
  currentTownId = townId;

  $loadingOverlay.style.display = "flex";
  $loadingOverlay.textContent = "Loading town...";

  createPhaserGame();
}
```

### 3.2 Overlapping Poll Prevention

**Problem:** `pollState()` doesn't guard against overlapping requests. If a poll takes >3 seconds (the interval), a second request fires while the first is still in flight.

**Fix:**

```javascript
let pollInFlight = false;

async function pollState() {
  if (!currentTownId || pollInFlight) return;
  pollInFlight = true;
  try {
    const data = await apiFetch(`/sim/towns/${currentTownId}/state`);
    if (data.state) updateFromState(data.state);
  } catch (e) {
    console.error("Poll error:", e);
  } finally {
    pollInFlight = false;
  }
}
```

### 3.3 `apiFetch` Has No Error Handling

**Problem (line 93–96 of `town.js`):** The current `apiFetch` doesn't check `res.ok`. A 500 error returns invalid JSON and throws an unhelpful error.

```javascript
// CURRENT — silently breaks on non-200
async function apiFetch(path, opts) {
  const res = await fetch(`${API}${path}`, opts);
  return res.json(); // 💥 throws on 500/404 with HTML error page
}
```

**Fix:**

```javascript
async function apiFetch(path, opts) {
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}
```

### 3.4 DOM Churn on Every Poll

**Problem:** `updateAgentCards()` rebuilds the entire `#agentList` innerHTML on every 3-second poll, even if nothing changed. This causes:
- Scroll position reset
- Selection/focus loss
- Unnecessary layout thrashing

**Fix:**

```javascript
let lastAgentListHash = '';

function updateAgentCards(npcs) {
  const hash = npcs
    .map(n => `${n.agent_id}:${n.x}:${n.y}:${n.status}:${n.pronunciatio}`)
    .join('|');
  if (hash === lastAgentListHash) return;
  lastAgentListHash = hash;

  // Preserve focused card
  const selectedId = document.querySelector('.agent-card.focused')?.dataset.agentId;
  // ... rebuild (existing code) ...
  if (selectedId) {
    document.querySelector(`.agent-card[data-agent-id="${selectedId}"]`)
      ?.classList.add('focused');
  }
}
```

### 3.5 Connection Status Indicator

Add a visual health dot to the topbar so spectators know if the connection is alive:

```
🟢 Connected, polling     🟡 Request in-flight     🔴 Last request failed
```

Show error details on hover. This is critical for a "watching a live game" feel — you need to know the stream is live.

---

## 🔴 4 · Conversation Display Fix — *Show Names, Not IDs*

> **Priority** P0 · **Effort** S · **Found by code review**

### The Bug

```javascript
// CURRENT (line 234 of town.js) — shows raw agent IDs, not human names:
`${c.agent_a} & ${c.agent_b}`  // renders "agent_abc123 & agent_def456"
```

### The Fix

```javascript
function updateConversations(convos) {
  if (!latestState) return;
  const nameMap = {};
  (latestState.npcs || []).forEach(n => nameMap[n.agent_id] = n.name);

  $convoList.innerHTML = convos.map(c => {
    const nameA = nameMap[c.agent_a] || c.agent_a;
    const nameB = nameMap[c.agent_b] || c.agent_b;
    return `
      <div class="convo-card" onclick="window._showConvoDetail('${c.session_id}')">
        <div class="convo-agents">${nameA} ↔ ${nameB}</div>
        <div class="convo-msg">${c.last_message || "..."}</div>
        <div class="convo-meta">${c.turns} turns · started step ${c.started_step}</div>
      </div>`;
  }).join("");
}
```

### Bonus: Conversation Detail Modal

Click a conversation card → open a modal overlay with:
- Full scrollable transcript with speaker names
- Turn numbers and timestamps
- "Jump to agents" button (pans camera to their location)
- Dark panel matching `--bg-card` with `--accent-purple` left border

---

## 🟡 5 · Day/Night Visual Cycle — *The Atmosphere*

> **Priority** P1 · **Effort** S · **Backend changes** None

### The Vision

The map subtly breathes with the simulation clock. As time passes, the world changes:

```
 ☀️  DAY (08:00–17:00)      Clear, full brightness
 🌅  DUSK (17:00–20:00)     Warm amber tint — the golden hour
 🌙  EVENING (20:00–22:00)  Deepening blue wash
 🌑  NIGHT (22:00–06:00)    Dark indigo overlay, reduced saturation
 🌄  DAWN (06:00–08:00)     Warm orange-pink glow fading in
```

### Implementation

```javascript
// In create() — add overlay rectangle
this._dayNightOverlay = this.add.rectangle(
  0, 0,
  this.cameras.main.width * 4, this.cameras.main.height * 4,
  0x000000, 0
).setScrollFactor(0).setDepth(998).setOrigin(0);

// In updateFromState() — adjust overlay
function updateDayNight(scene, hour) {
  const overlay = scene._dayNightOverlay;
  if (!overlay) return;
  let color = 0x000000, alpha = 0;
  if (hour >= 22 || hour < 6)  { color = 0x0a1a3a; alpha = 0.35; }  // night
  else if (hour >= 6  && hour < 8)  { color = 0xff9944; alpha = 0.10; }  // dawn
  else if (hour >= 17 && hour < 20) { color = 0xffaa55; alpha = 0.12; }  // dusk
  else if (hour >= 20 && hour < 22) { color = 0x1a2a4a; alpha = 0.20; }  // evening
  scene.tweens.add({
    targets: overlay, alpha, duration: 2000, ease: 'Sine.easeInOut'
  });
  overlay.setFillStyle(color, alpha);
}
```

### Extra Polish

- Update `gameTime` display color: cyan at day, gold at dusk, dim at night
- Add a small ☀️/🌙 icon next to the time display based on hour
- Subtle particle emitter for fireflies at night (5–8 soft yellow dots, slow random float)

---

## 🟡 6 · Agent Detail Drawer — *The Deep Dive*

> **Priority** P1 · **Effort** L · **Backend changes** None

### The Vision

When you click an agent card, the sidebar expands into a rich character sheet:

```
 ╔══════════════════════════════════════╗
 ║  ← Back to Agent List                ║
 ╠══════════════════════════════════════╣
 ║  [Sprite] Isabella Rodriguez    💬   ║
 ║  @owner_handle · Hobbs Cafe          ║
 ║                                      ║
 ║  ┌─────────────────────────────────┐ ║
 ║  │ 📍 Location: Hobbs Cafe         │ ║
 ║  │ 🎯 Goal: socialize at cafe      │ ║
 ║  │ 💭 Action: chatting with Klaus   │ ║
 ║  │ 📊 Status: chatting              │ ║
 ║  └─────────────────────────────────┘ ║
 ║                                      ║
 ║  ── Recent Activity ──               ║
 ║  s42: started chatting with Klaus    ║
 ║  s41: walked to Hobbs Cafe           ║
 ║  s40: finished morning routine       ║
 ║  s39: woke up at home                ║
 ║                                      ║
 ║  ── Memory Summary ──                ║
 ║  Events: 12 │ Thoughts: 4           ║
 ║  Reflections: 1                      ║
 ║  "Isabella thinks the cafe is the    ║
 ║   best place to meet neighbors"      ║
 ║                                      ║
 ║  ── Relationships ──                 ║
 ║  Klaus Mueller: ████████░░ 8.0       ║
 ║  Carlos Gomez:  ██████░░░░ 6.0       ║
 ║  Maria Lopez:   ████░░░░░░ 4.0       ║
 ╚══════════════════════════════════════╝
```

### Implementation Notes

- Add `selectedAgentId` state variable
- On agent card click → set `selectedAgentId`, render detail view replacing the agent list
- "← Back" button returns to list view
- Pull data from `npc.memory_summary` (already has `node_count`, `active_action`, etc.)
- Recent activity: maintain per-agent ring buffer (last 20 events) from state diffs
- Relationships: display if available in `memory_summary` or via future API endpoint

---

## 🟡 7 · Map Interaction — *The World Responds*

> **Priority** P1 · **Effort** L · **Backend changes** None

### 7.1 Clickable Locations

The Tiled JSON map already has a `locations` object layer. We can create invisible interactive zones:

```javascript
// In create(), after map layer setup:
const locationLayer = map.getObjectLayer('locations');
if (locationLayer) {
  locationLayer.objects.forEach(obj => {
    const zone = this.add.zone(obj.x, obj.y, obj.width, obj.height)
      .setOrigin(0, 0).setInteractive();
    zone.on('pointerover', () => showLocationTooltip(obj.name, obj.x, obj.y));
    zone.on('pointerout',  () => hideLocationTooltip());
    zone.on('pointerdown', () => showLocationPanel(obj.name));
  });
}
```

Hovering over a building shows its name. Clicking shows who's inside.

### 7.2 Clickable Agent Sprites

```javascript
// When creating sprites, add interactivity:
sprite.setInteractive();
sprite.on('pointerdown', () => window._focusAgent(npc.agent_id));
sprite.on('pointerover', () => sprite.setTint(0xe94560));
sprite.on('pointerout',  () => sprite.clearTint());
```

### 7.3 Zoom UI Buttons

Add `+` / `-` buttons as HTML overlay (not Phaser) for mobile users who can't use scroll wheel:

```html
<div class="zoom-controls">
  <button id="zoomIn">+</button>
  <button id="zoomOut">−</button>
</div>
```

---

## 🟡 8 · Mobile / Responsive — *Everyone's Invited*

> **Priority** P1 · **Effort** M

### The Problem

Currently `town.html` has zero responsive styles. On mobile, the sidebar is cut off and the topbar wraps poorly.

### The Fix

```css
@media (max-width: 800px) {
  .main-layout {
    flex-direction: column;
  }
  #game-container {
    height: 55vh;
    min-height: 300px;
  }
  .sidebar {
    width: 100%;
    height: 45vh;
    border-left: none;
    border-top: var(--pixel-border);
  }
  .topbar {
    flex-wrap: wrap;
    height: auto;
    padding: 6px 8px;
    gap: 6px;
  }
  .topbar select {
    min-width: 100px;
    font-size: 13px;
  }
}

@media (max-width: 480px) {
  #game-container { height: 45vh; }
  .sidebar { height: 55vh; }
  .topbar button { font-size: 0.36rem; padding: 4px 6px; }
}
```

### Touch Controls — Pinch to Zoom

```javascript
let lastPinchDist = 0;
this.input.on('pointermove', (pointer) => {
  if (this.input.pointer1.isDown && this.input.pointer2.isDown) {
    const dist = Phaser.Math.Distance.Between(
      this.input.pointer1.x, this.input.pointer1.y,
      this.input.pointer2.x, this.input.pointer2.y
    );
    if (lastPinchDist > 0) {
      const delta = dist - lastPinchDist;
      const cam = this.cameras.main;
      cam.setZoom(Phaser.Math.Clamp(cam.zoom + delta * 0.005, 0.3, 3));
    }
    lastPinchDist = dist;
  }
});
```

---

## 🟡 9 · Landing Page Fixes — *First Impressions*

> **Priority** P1 · **Effort** S

### 9.1 Town Card Navigation Bug

Town cards on the landing page should link directly to the viewer with the town pre-selected:

```javascript
card.addEventListener("click", () => {
  const base = getApiBase();
  const viewerUrl = `./town.html?api=${encodeURIComponent(base)}&town=${encodeURIComponent(town.town_id)}`;
  window.location.href = viewerUrl;
});
```

### 9.2 Preserve `?api=` on All Links

```javascript
function preserveApiParam() {
  const api = new URLSearchParams(window.location.search).get("api");
  if (!api) return;
  document.querySelectorAll('a[href*="town.html"]').forEach(link => {
    const url = new URL(link.href, window.location.origin);
    url.searchParams.set('api', api);
    link.href = url.toString();
  });
}
```

### 9.3 Loading Shimmer (Replace `--` Placeholders)

```css
.stat-loading {
  display: inline-block;
  width: 32px; height: 1em;
  background: linear-gradient(
    90deg,
    var(--ink-muted) 25%, var(--ink-soft) 50%, var(--ink-muted) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
@keyframes shimmer {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### 9.4 Reduced Motion Accessibility

```css
@media (prefers-reduced-motion: reduce) {
  .sky-glow { animation: none; }
  .badge { animation: none; }
  * { transition-duration: 0.01ms !important; }
}
```

```javascript
function initHeroParallax() {
  if (!heroArt) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  // ... existing parallax code ...
}
```

---

## 🟢 10 · Future Enhancements — *The Dream*

### 10.1 Server-Sent Events (SSE)

Replace 3-second polling with a real-time event stream:
```
GET /api/v1/sim/towns/{town_id}/events/stream
→ SSE stream: movement, conversation, memory, phase changes
```
Benefits: instant updates, lower server load, enables smooth interpolated animations.

### 10.2 Relationship Graph

Modal with a force-directed graph (canvas-based, no library needed):
- **Nodes** = agents (with sprites)
- **Edges** = relationships (thickness = score, color = positive/negative)
- Click node → focus agent
- Build from conversation count + memory references

### 10.3 Agent Path Trail

Fading trail of recent positions for the selected agent:
- Last 10 positions as small dots, opacity fading from 1.0 → 0.1
- Helps spectators understand movement patterns and routines

### 10.4 Replay / Rewind

- Record state snapshots per step (client-side, last 50 steps)
- Slider to scrub through recent history
- "▶ Play" button to replay at configurable speed (1x, 2x, 4x)

### 10.5 Ambient Audio

- Subtle background music that shifts with day/night cycle
- Sound effects: soft footstep sounds on movement, chime on conversation start
- Mute/volume controls in topbar
- Use Web Audio API, load small `.mp3` files (<100KB each)

### 10.6 Notification System

- Browser notification API for key events (first conversation, reflection triggers)
- In-app toast notifications with auto-dismiss (4s)
- Configurable: "Notify me when: conversation starts / agent arrives / reflection triggers"

---

## ✦ Implementation Priority — The Roadmap

| # | Feature | Status | Priority | Effort | Why It Matters |
|:---:|---|:---:|:---:|:---:|---|
| 1 | Fix Phaser lifecycle leak + poll overlap | ✅ Done | 🔴 P0 | S | **Prevents crashes** on town switch |
| 2 | Fix `apiFetch` error handling | ✅ Done | 🔴 P0 | S | **Prevents silent failures** |
| 3 | Show agent names in conversations | ✅ Done | 🔴 P0 | S | **Basic correctness** — IDs are unreadable |
| 4 | Reduce DOM churn (hash-based rebuild) | ✅ Done | 🔴 P0 | S | **Stops scroll-jank** during polling |
| 5 | Live event feed | ✅ Done | 🔴 P0 | M | **Transforms** the viewer from dashboard to narrative |
| 6 | On-map speech bubbles | ✅ Done | 🔴 P0 | M | **Makes the map feel alive** |
| 7 | Connection status indicator | ✅ Done | 🔴 P0 | S | Spectators need to know the stream is live |
| 8 | Town card navigation fix | ✅ Done | 🟡 P1 | S | Fixes broken click-through flow |
| 9 | Day/night visual cycle | ✅ Done | 🟡 P1 | S | Atmospheric polish — the world breathes |
| 10 | Conversation transcript modal | ⏳ Planned | 🟡 P1 | M | Narrative depth — read the actual dialogue |
| 11 | Mobile responsive styles | ✅ Done | 🟡 P1 | M | Accessibility for all devices |
| 12 | Agent detail drawer | ⏳ Planned | 🟡 P1 | L | Deep engagement with individual characters |
| 13 | Map interaction (locations + sprites) | 🟨 Partial | 🟡 P1 | L | Spatial understanding + discoverability |
| 14 | Landing page polish | 🟨 Partial | 🟡 P1 | S | First impressions count |
| 15 | Zoom UI buttons | ✅ Done | 🟢 P2 | S | Mobile zoom without scroll wheel |
| 16 | Ambient day/night particles | ⏳ Planned | 🟢 P2 | M | Atmospheric delight |
| 17 | Relationship graph | ⏳ Planned | 🟢 P2 | L | Social insight visualization |
| 18 | SSE real-time updates | ⏳ Planned | 🟢 P2 | XL | Architecture-level upgrade |

---

<div align="center">

*The goal isn't to build a better dashboard.*  
*It's to build a window into a world that makes you want to keep watching.*

</div>
