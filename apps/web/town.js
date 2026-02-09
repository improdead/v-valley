/* V-Valley Town Viewer — Phaser 3 game + API integration */
(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------
  const API_BASE = (function () {
    // Detect API base: same origin or explicit override via query param
    const params = new URLSearchParams(window.location.search);
    if (params.get("api")) return params.get("api").replace(/\/+$/, "");
    // Default: same origin
    return window.location.origin;
  })();
  const API = `${API_BASE}/api/v1`;
  const TILE_SIZE = 32;
  const POLL_INTERVAL_MS = 3000;
  const ANIM_DURATION_MS = 800; // how long sprite slides to new tile

  // Available character sprite PNGs (shipped in assets/characters/)
  const CHARACTER_SPRITES = [
    "Abigail_Chen", "Adam_Smith", "Arthur_Burton", "Ayesha_Khan",
    "Carlos_Gomez", "Carmen_Ortiz", "Eddy_Lin", "Francisco_Lopez",
    "Giorgio_Rossi", "Hailey_Johnson", "Isabella_Rodriguez", "Jane_Moreno",
    "Jennifer_Moore", "John_Lin", "Klaus_Mueller", "Latoya_Williams",
    "Maria_Lopez", "Mei_Lin", "Rajiv_Patel", "Ryan_Park",
    "Sam_Moore", "Tamara_Taylor", "Tom_Moreno", "Wolfgang_Schulz",
    "Yuriko_Yamamoto",
  ];

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let currentTownId = null;
  let phaserGame = null;
  let gameScene = null;
  let autoTickTimer = null;
  let pollTimer = null;
  let agentSpriteMap = {}; // agent_id -> { sprite, label, spriteKey, tween, labelTween }
  let agentCharMap = {};   // agent_id -> character sprite name (stable)
  let latestState = null;

  // ---------------------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function hashStr(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
      h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    }
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
    return res.json();
  }

  // ---------------------------------------------------------------------------
  // Town list
  // ---------------------------------------------------------------------------
  async function loadTowns() {
    try {
      const data = await apiFetch("/towns");
      $townSelect.innerHTML = '<option value="">-- Select Town --</option>';
      (data.towns || []).forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.town_id;
        opt.textContent = `${t.name || t.town_id} (${t.population}/${t.max_agents})`;
        $townSelect.appendChild(opt);
      });
      if (currentTownId) $townSelect.value = currentTownId;
    } catch (e) {
      console.error("Failed to load towns:", e);
    }
  }

  // ---------------------------------------------------------------------------
  // Tick
  // ---------------------------------------------------------------------------
  async function runTick() {
    if (!currentTownId) return;
    $btnTick.textContent = "...";
    try {
      const data = await apiFetch(`/sim/towns/${currentTownId}/tick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steps: 1,
          planning_scope: $tickScope.value,
          control_mode: $tickMode.value,
        }),
      });
      if (data.tick && data.tick.state) {
        updateFromState(data.tick.state);
      }
    } catch (e) {
      console.error("Tick error:", e);
    }
    $btnTick.textContent = "Tick";
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

  // ---------------------------------------------------------------------------
  // Poll state
  // ---------------------------------------------------------------------------
  async function pollState() {
    if (!currentTownId) return;
    try {
      const data = await apiFetch(`/sim/towns/${currentTownId}/state`);
      if (data.state) {
        updateFromState(data.state);
      }
    } catch (e) {
      console.error("Poll error:", e);
    }
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollState, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  // ---------------------------------------------------------------------------
  // Update UI from state
  // ---------------------------------------------------------------------------
  function updateFromState(state) {
    latestState = state;

    // Clock
    if (state.clock) {
      $gameTime.textContent = formatClock(state.clock);
      $stepCount.textContent = `step ${state.step || 0}`;
    }

    // Agents
    const npcs = state.npcs || [];
    updateAgentCards(npcs);
    if (gameScene) updateAgentSprites(npcs);

    // Conversations
    updateConversations(state.conversations || []);
  }

  function updateAgentCards(npcs) {
    if (npcs.length === 0) {
      $agentList.innerHTML = '<div class="empty-state"><p>No agents in this town</p></div>';
      return;
    }
    $agentList.innerHTML = npcs
      .map((npc) => {
        const initials = getInitials(npc.name);
        const action = npc.memory_summary?.active_action || "idle";
        const goalReason = npc.goal?.reason || "";
        return `
        <div class="agent-card" data-agent-id="${npc.agent_id}" onclick="window._focusAgent('${npc.agent_id}')">
          <span class="agent-pronunciatio">${initials}</span>
          <div class="agent-name">${npc.name}</div>
          <div class="agent-owner">${npc.owner_handle ? "@" + npc.owner_handle : "unclaimed"}</div>
          <div class="agent-action">${action}${goalReason ? " — " + goalReason : ""}</div>
          <div class="agent-pos">(${npc.x}, ${npc.y})${npc.current_location ? " @ " + npc.current_location : ""}</div>
        </div>`;
      })
      .join("");
  }

  function updateConversations(convos) {
    if (convos.length === 0) {
      $convoList.innerHTML = '<div class="empty-state"><p>No active conversations</p></div>';
      return;
    }
    $convoList.innerHTML = convos
      .map(
        (c) => `
        <div class="convo-card">
          <div class="convo-agents">${c.agent_a} & ${c.agent_b}</div>
          <div class="convo-msg">${c.last_message || "..."} (${c.turns} turns)</div>
        </div>`
      )
      .join("");
  }

  // ---------------------------------------------------------------------------
  // Focus on agent (camera follows)
  // ---------------------------------------------------------------------------
  window._focusAgent = function (agentId) {
    if (!gameScene || !agentSpriteMap[agentId]) return;
    const sprite = agentSpriteMap[agentId].sprite;
    gameScene.cameras.main.pan(sprite.x, sprite.y, 500, "Sine.easeInOut");
    // Highlight card
    document.querySelectorAll(".agent-card").forEach((el) => el.classList.remove("focused"));
    const card = document.querySelector(`.agent-card[data-agent-id="${agentId}"]`);
    if (card) card.classList.add("focused");
  };

  // ---------------------------------------------------------------------------
  // Phaser scene
  // ---------------------------------------------------------------------------
  function createPhaserGame() {
    if (phaserGame) {
      phaserGame.destroy(true);
      phaserGame = null;
      gameScene = null;
      agentSpriteMap = {};
    }

    const container = document.getElementById("game-container");
    const config = {
      type: Phaser.AUTO,
      width: container.clientWidth,
      height: container.clientHeight,
      parent: "game-container",
      pixelArt: true,
      physics: {
        default: "arcade",
        arcade: { gravity: { y: 0 } },
      },
      scene: { preload, create, update },
      scale: {
        mode: Phaser.Scale.RESIZE,
        autoCenter: Phaser.Scale.CENTER_BOTH,
      },
    };
    phaserGame = new Phaser.Game(config);
  }

  // -- Preload --
  function preload() {
    $loadingOverlay.textContent = "Loading map assets...";

    // Tileset images
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

    // Tilemap JSON
    this.load.tilemapTiledJSON("map", "assets/map/the_ville.json");

    // Character atlases — load each character PNG with the shared atlas.json
    CHARACTER_SPRITES.forEach((name) => {
      this.load.atlas(
        `char_${name}`,
        `assets/characters/${name}.png`,
        "assets/characters/atlas.json"
      );
    });
  }

  // -- Create --
  function create() {
    gameScene = this;
    $loadingOverlay.textContent = "Building world...";

    // Build tilemap
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

    // Create visible layers
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

    // Camera setup — invisible "player" sprite for camera follow
    this._cameraTarget = this.physics.add
      .sprite(map.widthInPixels / 2, map.heightInPixels / 2, `char_${CHARACTER_SPRITES[0]}`, "down")
      .setSize(1, 1)
      .setAlpha(0);

    const camera = this.cameras.main;
    camera.startFollow(this._cameraTarget, true, 0.1, 0.1);
    camera.setBounds(0, 0, map.widthInPixels, map.heightInPixels);
    camera.setZoom(1);

    // Keyboard controls
    this._cursors = this.input.keyboard.createCursorKeys();
    this._keys = this.input.keyboard.addKeys("W,A,S,D,PLUS,MINUS");

    // Mouse wheel zoom
    this.input.on("wheel", (pointer, gameObjects, deltaX, deltaY) => {
      const newZoom = Phaser.Math.Clamp(camera.zoom - deltaY * 0.001, 0.3, 3);
      camera.setZoom(newZoom);
    });

    // Create walk animations for each character atlas
    CHARACTER_SPRITES.forEach((name) => {
      const key = `char_${name}`;
      const anims = this.anims;
      ["down", "left", "right", "up"].forEach((dir) => {
        const animKey = `${key}-${dir}-walk`;
        if (!anims.exists(animKey)) {
          anims.create({
            key: animKey,
            frames: anims.generateFrameNames(key, {
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

    // Fetch initial state
    pollState();
    startPolling();
  }

  // -- Update --
  function update(time, delta) {
    if (!this._cameraTarget || !this._cursors) return;

    const speed = 500;
    this._cameraTarget.body.setVelocity(0);

    if (this._cursors.left.isDown || this._keys.A.isDown) {
      this._cameraTarget.body.setVelocityX(-speed);
    }
    if (this._cursors.right.isDown || this._keys.D.isDown) {
      this._cameraTarget.body.setVelocityX(speed);
    }
    if (this._cursors.up.isDown || this._keys.W.isDown) {
      this._cameraTarget.body.setVelocityY(-speed);
    }
    if (this._cursors.down.isDown || this._keys.S.isDown) {
      this._cameraTarget.body.setVelocityY(speed);
    }
  }

  // ---------------------------------------------------------------------------
  // Agent sprite management
  // ---------------------------------------------------------------------------
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
        // Create new sprite
        const sprite = gameScene.physics.add
          .sprite(targetX, targetY, atlasKey, "down")
          .setSize(30, 40)
          .setOffset(0, 0)
          .setDepth(5);

        // Name label above sprite
        const label = gameScene.add
          .text(targetX, targetY - 48, getInitials(npc.name), {
            font: "bold 11px monospace",
            color: "#ffffff",
            backgroundColor: "#00000099",
            padding: { x: 4, y: 2 },
          })
          .setOrigin(0.5, 1)
          .setDepth(11);

        agentSpriteMap[npc.agent_id] = { sprite, label, atlasKey, lastX: npc.x, lastY: npc.y, tween: null, labelTween: null };
      } else {
        const entry = agentSpriteMap[npc.agent_id];
        const sprite = entry.sprite;
        const label = entry.label;

        // Determine movement direction for animation
        const dx = npc.x - entry.lastX;
        const dy = npc.y - entry.lastY;

        if (dx !== 0 || dy !== 0) {
          // Animate walk
          let dir = "down";
          if (Math.abs(dx) >= Math.abs(dy)) {
            dir = dx > 0 ? "right" : "left";
          } else {
            dir = dy > 0 ? "down" : "up";
          }

          sprite.anims.play(`${atlasKey}-${dir}-walk`, true);

          // Kill existing tweens before creating new ones
          if (entry.tween) entry.tween.stop();
          if (entry.labelTween) entry.labelTween.stop();

          // Tween sprite to new position
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

          // Tween label
          entry.labelTween = gameScene.tweens.add({
            targets: label,
            x: targetX,
            y: targetY - 48,
            duration: ANIM_DURATION_MS,
            ease: "Linear",
          });
        }

        // Update label text
        const initials = getInitials(npc.name);
        const action = npc.memory_summary?.active_action;
        label.setText(action ? `${initials}: ${action.substring(0, 30)}` : initials);

        entry.lastX = npc.x;
        entry.lastY = npc.y;
      }
    });

    // Remove sprites for agents that left
    Object.keys(agentSpriteMap).forEach((id) => {
      if (!seenIds.has(id)) {
        agentSpriteMap[id].sprite.destroy();
        agentSpriteMap[id].label.destroy();
        delete agentSpriteMap[id];
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Town selection
  // ---------------------------------------------------------------------------
  function selectTown(townId) {
    if (!townId) return;
    currentTownId = townId;
    stopPolling();
    if (autoTickTimer) {
      clearInterval(autoTickTimer);
      autoTickTimer = null;
      $btnAutoTick.classList.remove("active");
      $btnAutoTick.textContent = "Auto";
    }
    agentSpriteMap = {};
    agentCharMap = {};

    $loadingOverlay.style.display = "flex";
    $loadingOverlay.textContent = "Loading town...";

    createPhaserGame();
  }

  // ---------------------------------------------------------------------------
  // Event bindings
  // ---------------------------------------------------------------------------
  $townSelect.addEventListener("change", (e) => selectTown(e.target.value));
  $btnRefreshTowns.addEventListener("click", loadTowns);
  $btnTick.addEventListener("click", runTick);
  $btnAutoTick.addEventListener("click", toggleAutoTick);

  // Auto-select from URL param
  const urlTown = new URLSearchParams(window.location.search).get("town");

  // Init
  loadTowns().then(() => {
    if (urlTown) {
      $townSelect.value = urlTown;
      selectTown(urlTown);
    }
  });
})();
