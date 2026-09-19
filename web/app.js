const ALLOWED_EXT = [".md", ".txt", ".jsonl"];

function loadFlag(key, fallback) {
  const raw = localStorage.getItem(key);
  if (raw === null) return fallback;
  return raw === "1";
}

function saveFlag(key, value) {
  localStorage.setItem(key, value ? "1" : "0");
}

window.useFile = loadFlag("useFile", true);
window.useLora = loadFlag("useLora", true);

function setUseFile(value) {
  window.useFile = value;
  saveFlag("useFile", value);
}

function setUseLora(value) {
  window.useLora = value;
  saveFlag("useLora", value);
}

function allowedName(name) {
  const lower = (name || "").toLowerCase();
  return ALLOWED_EXT.some((ext) => lower.endsWith(ext));
}

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, opt);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function paint(st) {
  const pill = document.getElementById("pill");
  if (pill) {
    if (st.on) {
      pill.textContent = st.model + " ligado";
      pill.className = "pill on";
    } else {
      pill.textContent = (st.model || "modelo") + " desligado";
      pill.className = "pill off";
    }
  }

  const filePill = document.getElementById("filePill");
  if (filePill) {
    const n = (st.files || []).length;
    filePill.textContent = (window.useFile ? "consulta on" : "consulta off") + " · " + n + " arq";
    filePill.className = window.useFile ? "pill on" : "pill off";
  }

  const fileToggle = document.getElementById("fileToggle");
  if (fileToggle) {
    fileToggle.textContent = window.useFile ? "Desativar consulta" : "Ativar consulta";
  }

  const loraPill = document.getElementById("loraPill");
  if (loraPill) {
    const ready = !!st.lora;
    const on = ready && window.useLora;
    loraPill.textContent = on ? "lora ligado" : ready ? "lora desligado" : "lora ausente";
    loraPill.className = on ? "pill on" : "pill off";
  }

  const loraToggle = document.getElementById("loraToggle");
  if (loraToggle) {
    loraToggle.textContent = window.useLora ? "Desligar LoRA" : "Ligar LoRA";
    loraToggle.disabled = !st.lora;
  }

  const list = document.getElementById("fileList");
  if (list) {
    list.innerHTML = "";
    const files = st.files || [];
    if (!files.length) {
      const li = document.createElement("li");
      li.textContent = "Nenhum arquivo no caderno ainda.";
      list.appendChild(li);
    } else {
      files.forEach((row) => {
        const li = document.createElement("li");
        li.textContent = row.folder + "/" + row.name;
        list.appendChild(li);
      });
    }
  }
}

async function refresh() {
  paint(await api("/api/status"));
}

function appendLine(boxId, text, cls) {
  const box = document.getElementById(boxId);
  if (!box) return;
  const el = document.createElement("div");
  el.className = cls || "msg sys";
  el.textContent = text;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

const TZ = "America/Sao_Paulo";

function fmtTime(iso, withDate) {
  const d = new Date(iso);
  if (isNaN(d)) return iso || "";
  return withDate
    ? d.toLocaleString("pt-BR", { timeZone: TZ, dateStyle: "short", timeStyle: "medium" })
    : d.toLocaleTimeString("pt-BR", { timeZone: TZ });
}

function fmtSize(bytes) {
  return bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + " MB" : Math.round(bytes / 1024) + " KB";
}

function setSpin(on) {
  const spin = document.getElementById("spin");
  if (spin) spin.classList.toggle("on", on);
  const trainBtn = document.getElementById("trainBtn");
  if (trainBtn) trainBtn.disabled = on;
  ["trainFile"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = on;
  });
  const stopBtn = document.getElementById("stopBtn");
  if (stopBtn) stopBtn.disabled = !on;
  const onBtn = document.getElementById("onBtn");
  if (onBtn) onBtn.disabled = on;
  const banner = document.getElementById("trainBanner");
  if (banner) banner.classList.toggle("on", on);
}

let trainPoll = null;
let lastTrainLog = "";

async function watchTrain() {
  const job = await api("/api/train-job");
  setSpin(!!job.spinning);
  paintRounds(job);
  const line = job.log || "";
  if (line && line !== lastTrainLog) {
    lastTrainLog = line;
    appendLine("log", line);
  }
  if (window.onTrainJob) window.onTrainJob(job);
  if (job.state === "running") return;
  clearInterval(trainPoll);
  trainPoll = null;
  setSpin(false);
  refresh().catch(() => {});
}

function paintBar(job) {
  const bar = document.getElementById("bar");
  if (!bar) return;
  const rodando = job.state === "running";
  bar.classList.toggle("on", rodando || job.state === "done");
  bar.classList.toggle("good", job.state === "done");
  let pct = 0;
  if (job.state === "done") pct = 100;
  else if (job.need_streak && job.streak) pct = (job.streak / job.need_streak) * 100;
  else if (job.total && job.eval && job.eval.length) pct = (job.eval.length / job.total) * 100;
  else if (job.planned) pct = Math.min(95, (job.train_rounds / job.planned) * 100);
  else if (job.train_rounds) pct = Math.min(90, job.train_rounds * 12);
  bar.firstElementChild.style.width = pct + "%";
}

function paintRounds(job) {
  paintBar(job);
  const box = document.getElementById("rounds");
  if (box) {
    const seq = job.need_streak ? "Sequência limpa: " + (job.streak || 0) + "/" + job.need_streak + ". " : "";
    const head = job.state === "running" && job.train_rounds
      ? seq + "Rodadas feitas: " + job.train_rounds + (job.planned ? " (previstas por tentativa: " + job.planned + ")" : "") + ". "
      : (job.attempts && job.attempts.length ? seq : "");
    box.textContent = head + ((job.rounds || []).length
      ? "Tentativas: " + (job.attempts || []).map((a, i) => (i + 1) + "ª " + a.rounds + " rod. " + a.passed + "/" + a.total + (a.first ? " ✓de primeira" : " (calibrou)")).join(" · ")
      : "");
  }
}

function startTrainWatch() {
  if (trainPoll) clearInterval(trainPoll);
  trainPoll = setInterval(() => watchTrain().catch((err) => {
    appendLine("trainLog", err.message);
    appendLine("log", err.message);
  }), 3000);
}

async function resumeTrainIfRunning() {
  const job = await api("/api/train-job");
  paintRounds(job);
  if (window.onTrainJob) window.onTrainJob(job);
  if (job.state === "running") {
    setSpin(true);
    startTrainWatch();
    return true;
  }
  return false;
}

function markNav() {
  const here = location.pathname.replace(/\/index\.html$/, "/") || "/";
  document.querySelectorAll("nav a").forEach((a) => {
    const href = a.getAttribute("href");
    a.classList.toggle("here", href === here || (here === "/" && href === "/"));
  });
}

async function paintHealth() {
  const el = document.getElementById("conn");
  if (!el) return;
  try {
    const h = await api("/api/health");
    el.textContent = h.conectado ? "conectado" : "não conectado";
    el.className = "pill " + (h.conectado ? "on" : "off");
    el.title = h.resumo + " · ollama em " + h.ollama_host;
    if (!h.conectado) el.textContent = "não conectado: " + h.resumo.replace("falta: ", "");
  } catch (err) {
    el.textContent = "não conectado";
    el.className = "pill off";
    el.title = err.message;
  }
}

function addConnPill() {
  const nav = document.querySelector("nav");
  if (!nav || document.getElementById("conn")) return;
  const el = document.createElement("span");
  el.id = "conn";
  el.className = "pill off";
  el.textContent = "verificando…";
  el.style.marginLeft = "auto";
  el.style.alignSelf = "center";
  nav.appendChild(el);
  paintHealth();
  setInterval(paintHealth, 20000);
}

document.addEventListener("DOMContentLoaded", () => { markNav(); addConnPill(); });
