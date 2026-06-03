// Vanilla JS dashboard. Polls /api/* endpoints, updates DOM.
// Falls back gracefully if Chart.js fails to load (just shows numbers).

const VARIANTS = ["baseline", "attack", "est"];
const PIECES = {
  // white
  "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
  // black
  "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
};

let lastCheckpoints = [];
let spectateSource = null;
let lossBuffers = { baseline: [], attack: [], est: [] };
let lossCharts = {};

document.addEventListener("DOMContentLoaded", () => {
  setupModeButtons();
  setupAutoModeToggle();
  setupPauseButton();
  setupMatchForm();
  setupSpectateModal();
  initLossCharts();
  pollStatus();
  pollCheckpoints();
  pollEvery(2000, () => { pollStatus(); pollCheckpoints(); });
});

function setupModeButtons() {
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const mode = btn.dataset.mode;
      btn.classList.add("active");
      try {
        const r = await postJson("/api/mode", { mode });
        if (r.ok) {
          flash(`Mode -> ${r.mode}`);
        } else {
          flash(`Mode change failed: ${r.mode}`);
        }
      } catch (e) {
        flash(`Mode change error: ${e.message}`);
      }
    });
  });
}

function setupAutoModeToggle() {
  const t = document.getElementById("auto-mode-toggle");
  t.addEventListener("change", async () => {
    try {
      await postJson("/api/auto_mode", { enabled: t.checked });
      flash(t.checked ? "Auto-mode enabled" : "Auto-mode disabled");
    } catch (e) {
      t.checked = !t.checked;
      flash(`Auto-mode error: ${e.message}`);
    }
  });
}

function setupPauseButton() {
  const btn = document.getElementById("pause-btn");
  btn.addEventListener("click", async () => {
    const isPaused = btn.classList.contains("paused");
    try {
      await postJson("/api/pause", { paused: !isPaused });
      btn.classList.toggle("paused", !isPaused);
      btn.textContent = isPaused ? "⏸ Pause" : "▶ Resume";
      flash(isPaused ? "Resumed" : "Paused");
    } catch (e) {
      flash(`Pause error: ${e.message}`);
    }
  });
}

function setupMatchForm() {
  document.getElementById("match-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const type = document.getElementById("match-type").value;
    const body = {
      type,
      white: document.getElementById("match-white").value,
      black: document.getElementById("match-black").value,
      visits: parseInt(document.getElementById("match-visits").value, 10),
    };
    if (type === "puzzle") {
      const pid = document.getElementById("match-puzzle-id").value;
      if (pid) body.puzzle_id = pid;
    }
    try {
      const r = await postJson("/api/matches", body);
      if (r.ok) {
        document.getElementById("match-status").textContent = `Match queued (#${r.match.id})`;
        openSpectate(r.match);
      } else {
        document.getElementById("match-status").textContent = `Failed: ${r.error || "unknown"}`;
      }
    } catch (e) {
      document.getElementById("match-status").textContent = `Error: ${e.message}`;
    }
  });
}

function setupSpectateModal() {
  document.getElementById("spectate-close").addEventListener("click", closeSpectate);
}

function initLossCharts() {
  if (typeof Chart === "undefined") {
    // Chart.js not loaded (offline or CDN blocked). Just show numbers.
    return;
  }
  Chart.defaults.color = "#8b949e";
  Chart.defaults.borderColor = "#2c3540";
  for (const v of VARIANTS) {
    const ctx = document.getElementById(`chart-${v}`).getContext("2d");
    lossCharts[v] = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          data: [],
          borderColor: { baseline: "#3fb950", attack: "#f85149", est: "#d29922" }[v],
          backgroundColor: "transparent",
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 0,
        }],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { display: false },
          y: { display: false, beginAtZero: false },
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    });
  }
}

async function pollStatus() {
  try {
    const s = await fetchJson("/api/status");
    updateHeader(s);
    updateResources(s.resources || {});
    updateLosses(s.losses || {});
    updateThroughput(s.throughput_gpm || {});
    updateBuffers(s.buffers || {});
    updateModeButtons(s);
    updatePauseButton(s);
    document.getElementById("last-update").textContent = `updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    // network blip; keep going
  }
}

async function pollCheckpoints() {
  try {
    const list = await fetchJson("/api/checkpoints");
    lastCheckpoints = list;
    updateCheckpoints(list);
  } catch (e) {}
}

function pollEvery(ms, fn) {
  setInterval(fn, ms);
}

function updateHeader(s) {
  document.getElementById("round-pill").textContent = `round ${s.round}`;
  document.getElementById("total-games").textContent = formatInt(s.total_games);
  document.getElementById("total-steps").textContent = formatInt(s.total_training_steps);
  document.getElementById("mode-value").textContent = s.performance_mode;
  document.getElementById("paused-value").textContent = s.training_paused ? "yes" : "no";
  const t = document.getElementById("auto-mode-toggle");
  if (t.checked !== s.auto_mode) t.checked = s.auto_mode;
}

function updateResources(r) {
  setBar("vram", r.vram_pct, `${r.vram_used_mb ?? "--"}/${r.vram_total_mb ?? "--"} MB`);
  setBar("cpu", r.cpu_pct, `${r.cpu_pct != null ? r.cpu_pct.toFixed(1) : "--"}%`);
  setBar("ram", r.ram_pct, `${r.ram_pct != null ? r.ram_pct.toFixed(1) : "--"}%`);
}

function setBar(name, pct, text) {
  pct = Math.max(0, Math.min(100, pct || 0));
  document.getElementById(`bar-${name}`).style.width = `${pct}%`;
  document.getElementById(`text-${name}`).textContent = text;
}

function updateLosses(losses) {
  for (const v of VARIANTS) {
    const v_loss = losses[v];
    const el = document.getElementById(`loss-${v}`);
    if (v_loss == null) {
      el.textContent = "--";
      continue;
    }
    el.textContent = v_loss.toFixed(3);
    // Append to chart buffer
    if (lossCharts[v]) {
      lossBuffers[v].push(v_loss);
      if (lossBuffers[v].length > 80) lossBuffers[v].shift();
      const chart = lossCharts[v];
      chart.data.labels = lossBuffers[v].map((_, i) => i);
      chart.data.datasets[0].data = lossBuffers[v];
      chart.update("none");
    }
  }
}

function updateThroughput(tput) {
  for (const v of VARIANTS) {
    const v_tput = tput[v];
    const el = document.getElementById(`tput-${v}`);
    el.textContent = v_tput == null ? "--" : v_tput.toFixed(1);
  }
}

function updateBuffers(buffers) {
  const grid = document.getElementById("buffer-grid");
  grid.innerHTML = "";
  for (const v of VARIANTS) {
    const b = buffers[v] || { size: 0, capacity: 0, fill_pct: 0 };
    const cell = document.createElement("div");
    cell.className = "buffer-cell";
    cell.innerHTML = `
      <div class="buffer-name">${v}</div>
      <div class="buffer-bar"><div class="buffer-bar-fill" style="width:${b.fill_pct}%"></div></div>
      <div class="buffer-text">${formatInt(b.size)} / ${formatInt(b.capacity)} (${b.fill_pct.toFixed(1)}%)</div>
    `;
    grid.appendChild(cell);
  }
}

function updateModeButtons(s) {
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === s.performance_mode);
  });
}

function updatePauseButton(s) {
  const btn = document.getElementById("pause-btn");
  btn.classList.toggle("paused", s.training_paused);
  btn.textContent = s.training_paused ? "▶ Resume" : "⏸ Pause";
}

function updateCheckpoints(list) {
  const tbody = document.getElementById("ckpt-tbody");
  if (!list.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="muted">No checkpoints yet</td></tr>`;
    return;
  }
  list.sort((a, b) => b.step - a.step);
  const now = Date.now() / 1000;
  tbody.innerHTML = "";
  for (const c of list.slice(0, 20)) {
    const age = formatAge(now - c.mtime);
    const row = document.createElement("tr");
    row.className = "ckpt-row";
    row.dataset.variant = c.variant;
    row.dataset.step = c.step;
    row.innerHTML = `
      <td>${c.variant}</td>
      <td>${c.step}</td>
      <td>${c.size_mb} MB</td>
      <td>${age}</td>
      <td><button class="play-btn">▶ Watch</button></td>
    `;
    row.addEventListener("click", () => {
      const otherVariant = c.variant === "baseline" ? "attack" : "baseline";
      document.getElementById("match-white").value = `${c.variant}_step_${c.step}`;
      document.getElementById("match-black").value = otherVariant;
      document.getElementById("match-type").value = "model";
      document.getElementById("match-form").requestSubmit();
    });
    tbody.appendChild(row);
  }
}

// Spectate ------------------------------------------------------------------

function openSpectate(match) {
  document.getElementById("spectate-modal").classList.remove("hidden");
  document.getElementById("spectate-title").textContent = `Match #${match.id} (${match.type})`;
  document.getElementById("spectate-moves").innerHTML = "";
  document.getElementById("spectate-status").textContent = "starting...";
  drawBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  connectSpectateStream();
}

function closeSpectate() {
  document.getElementById("spectate-modal").classList.add("hidden");
  if (spectateSource) {
    spectateSource.close();
    spectateSource = null;
  }
}

function connectSpectateStream() {
  if (spectateSource) spectateSource.close();
  // SSE — modern browsers (and EventSource polyfills) work.
  try {
    spectateSource = new EventSource("/api/matches/stream");
    spectateSource.onmessage = (e) => {
      try {
        const evt = JSON.parse(e.data);
        handleSpectateEvent(evt);
      } catch {}
    };
    spectateSource.onerror = () => {
      // Browser will auto-reconnect; nothing to do.
    };
  } catch (e) {
    document.getElementById("spectate-status").textContent = `SSE failed: ${e.message}`;
  }
}

function handleSpectateEvent(evt) {
  if (evt.type === "move" || evt.type === "drill_move") {
    if (evt.fen) drawBoard(evt.fen);
    if (evt.san) {
      const li = document.createElement("li");
      li.textContent = `${evt.ply ?? "?"}. ${evt.san}`;
      document.getElementById("spectate-moves").appendChild(li);
      document.getElementById("spectate-moves").scrollTop = 1e9;
    }
    if (evt.eval != null) updateEvalBar(evt.eval);
    if (evt.type === "drill_move") {
      const st = document.getElementById("spectate-status");
      st.textContent = evt.correct ? "✓ correct" : `✗ expected ${evt.expected_san || "(?)"}`;
    }
  } else if (evt.type === "done" || evt.type === "result") {
    document.getElementById("spectate-status").textContent = `Result: ${evt.result || "?"}`;
  } else if (evt.type === "error") {
    document.getElementById("spectate-status").textContent = `Error: ${evt.error || "?"}`;
  }
}

function drawBoard(fen) {
  const canvas = document.getElementById("spectate-board");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const size = 48;
  const board = parseFenBoard(fen);
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const light = (r + f) % 2 === 0;
      ctx.fillStyle = light ? "#f0d9b5" : "#b58863";
      ctx.fillRect(f * size, r * size, size, size);
      const piece = board[r][f];
      if (piece) {
        ctx.fillStyle = piece === piece.toUpperCase() ? "#fff" : "#000";
        ctx.strokeStyle = piece === piece.toUpperCase() ? "#000" : "#fff";
        ctx.lineWidth = 1;
        ctx.font = `${size - 8}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const sym = PIECES[piece] || piece;
        ctx.fillText(sym, f * size + size / 2, r * size + size / 2 + 2);
        ctx.strokeText(sym, f * size + size / 2, r * size + size / 2 + 2);
      }
    }
  }
}

function parseFenBoard(fen) {
  const rows = fen.split(" ")[0].split("/");
  const board = [];
  for (const row of rows) {
    const r = [];
    for (const ch of row) {
      if (/\d/.test(ch)) {
        for (let i = 0; i < parseInt(ch, 10); i++) r.push("");
      } else {
        r.push(ch);
      }
    }
    while (r.length < 8) r.push("");
    board.push(r);
  }
  return board;
}

function updateEvalBar(value) {
  // value in [-1, 1]; 0 = equal, 1 = white winning
  const clamped = Math.max(-1, Math.min(1, value));
  const whitePct = 50 + clamped * 50;
  document.getElementById("eval-bar-white").style.width = `${whitePct}%`;
  document.getElementById("eval-value").textContent = clamped.toFixed(2);
}

// Helpers -------------------------------------------------------------------

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function postJson(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

function formatInt(n) {
  if (n == null) return "--";
  return Math.round(n).toLocaleString();
}

function formatAge(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

let flashTimer = null;
function flash(msg) {
  const el = document.getElementById("last-update");
  el.textContent = msg;
  el.style.color = "var(--accent-2)";
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    el.style.color = "";
    pollStatus();
  }, 1500);
}
