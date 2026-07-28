const els = {
  modelStatus: document.getElementById("modelStatus"),
  loadBtn: document.getElementById("loadBtn"),
  adapterSelect: document.getElementById("adapterSelect"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearBtn: document.getElementById("clearBtn"),
  refreshDevicesBtn: document.getElementById("refreshDevicesBtn"),
  inputDevice: document.getElementById("inputDevice"),
  outputDevice: document.getElementById("outputDevice"),
  vadThreshold: document.getElementById("vadThreshold"),
  vadValue: document.getElementById("vadValue"),
  tickSeconds: document.getElementById("tickSeconds"),
  contextSeconds: document.getElementById("contextSeconds"),
  bargeIn: document.getElementById("bargeIn"),
  sessionId: document.getElementById("sessionId"),
  sampleRate: document.getElementById("sampleRate"),
  clock: document.getElementById("clock"),
  rms: document.getElementById("rms"),
  vadState: document.getElementById("vadState"),
  silence: document.getElementById("silence"),
  lastLabel: document.getElementById("lastLabel"),
  skips: document.getElementById("skips"),
  clientState: document.getElementById("clientState"),
  probCanvas: document.getElementById("probCanvas"),
  bigLabel: document.getElementById("bigLabel"),
  mWait: document.getElementById("mWait"),
  mBack: document.getElementById("mBack"),
  mSupport: document.getElementById("mSupport"),
  pWait: document.getElementById("pWait"),
  pBack: document.getElementById("pBack"),
  pSupport: document.getElementById("pSupport"),
  actionProbabilitiesTitle: document.getElementById("actionProbabilitiesTitle"),
  actionProbabilities: document.getElementById("actionProbabilities"),
  latPrep: document.getElementById("latPrep"),
  latForward: document.getElementById("latForward"),
  latLabel: document.getElementById("latLabel"),
  latText: document.getElementById("latText"),
  latTalker: document.getElementById("latTalker"),
  assistantText: document.getElementById("assistantText"),
  logBody: document.getElementById("logBody"),
  logPath: document.getElementById("logPath"),
};

let ws = null;
let audioContext = null;
let mediaStream = null;
let processor = null;
let sourceNode = null;
let muteGain = null;
let running = false;
let skipped = 0;
let probHistory = [];
let currentAssistantAudio = null;
let loadedStatus = null;
let expectedSocketClose = false;
let wsPingTimer = null;
let adapterSwitching = false;

const maxHistory = 240;
const maxLogRows = 80;

function fmtMs(v) {
  if (!v && v !== 0) return "-";
  return `${v.toFixed(0)}ms`;
}

function setStatus(text, ok = false) {
  els.modelStatus.textContent = text;
  els.modelStatus.style.color = ok ? "#65d68a" : "#9aa7b5";
}

function setClientState(text, ok = false) {
  els.clientState.textContent = text;
  els.clientState.style.color = ok ? "#65d68a" : "#ffd36f";
}

function clearWsPingTimer() {
  if (wsPingTimer) {
    clearInterval(wsPingTimer);
    wsPingTimer = null;
  }
}

function startWsPingTimer() {
  clearWsPingTimer();
  wsPingTimer = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping", client_time: Date.now() / 1000 }));
    }
  }, 15000);
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  loadedStatus = data;
  const stage = data.runtime_stage && data.runtime_stage !== "idle" ? ` / ${data.runtime_stage}` : "";
  const adapter = data.active_adapter_name ? ` / ${data.active_adapter_name}` : "";
  setStatus(`model: ${data.status}${adapter}${data.detail ? " / " + data.detail : ""}${stage}`, data.status === "loaded");
  syncAdapterOptions(data);
  if (!running && els.actionProbabilities.children.length === 0) renderActionProbabilities({});
  els.startBtn.disabled = data.status !== "loaded" || running;
  els.adapterSelect.disabled = data.status !== "loaded" || running || adapterSwitching;
  return data;
}

function syncAdapterOptions(data) {
  if (!Array.isArray(data.adapters) || data.adapters.length === 0) return;
  const signature = data.adapters.map((item) => `${item.key}:${item.display_name}:${item.loaded}`).join("|");
  if (els.adapterSelect.dataset.signature !== signature) {
    els.adapterSelect.innerHTML = "";
    data.adapters.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.key;
      option.textContent = `${item.display_name}${item.loaded ? "" : " (not loaded)"}`;
      option.disabled = !item.loaded;
      els.adapterSelect.appendChild(option);
    });
    els.adapterSelect.dataset.signature = signature;
  }
  if (data.active_adapter) els.adapterSelect.value = data.active_adapter;
}

async function switchAdapter() {
  if (running) throw new Error("Stop the microphone before switching LoRA adapters.");
  const adapterKey = els.adapterSelect.value;
  if (!adapterKey || adapterKey === loadedStatus?.active_adapter) return;
  adapterSwitching = true;
  els.adapterSelect.disabled = true;
  els.startBtn.disabled = true;
  setStatus(`model: switching to ${adapterKey}...`);
  try {
    const res = await fetch(`/api/adapter/${encodeURIComponent(adapterKey)}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `adapter switch failed (${res.status})`);
    loadedStatus = data;
    probHistory = [];
    drawProbGraph();
    renderActionProbabilities({});
    els.bigLabel.textContent = "/W /B /S";
    els.lastLabel.textContent = "-";
    els.assistantText.textContent = "-";
    addLog({ type: "adapter", response: `active: ${data.active_adapter_name}` });
  } finally {
    adapterSwitching = false;
    await refreshStatus();
  }
}

async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    addLog({ type: "device", message: "enumerateDevices is not supported by this browser" });
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const currentIn = els.inputDevice.value;
    const currentOut = els.outputDevice.value;
    const inputs = devices.filter((d) => d.kind === "audioinput");
    const outputs = devices.filter((d) => d.kind === "audiooutput");
    els.inputDevice.innerHTML = `<option value="">Default microphone</option>`;
    els.outputDevice.innerHTML = `<option value="">Default speaker</option>`;
    inputs.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${i + 1}`;
      els.inputDevice.appendChild(opt);
    });
    outputs.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Speaker ${i + 1}`;
      els.outputDevice.appendChild(opt);
    });
    if ([...els.inputDevice.options].some((o) => o.value === currentIn)) els.inputDevice.value = currentIn;
    if ([...els.outputDevice.options].some((o) => o.value === currentOut)) els.outputDevice.value = currentOut;
    addLog({ type: "device", response: `inputs=${inputs.length}, outputs=${outputs.length}` });
  } catch (err) {
    addLog({ type: "device", message: String(err) });
  }
}

async function loadModel() {
  els.loadBtn.disabled = true;
  setStatus("model: loading...");
  await fetch("/api/load", { method: "POST" });
  const timer = setInterval(async () => {
    const data = await refreshStatus();
    if (data.status === "loaded" || data.status === "error") {
      clearInterval(timer);
      els.loadBtn.disabled = data.status === "loading";
    }
  }, 1500);
}

function floatToInt16Buffer(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out.buffer;
}

function localRms(buf) {
  let sum = 0;
  for (let i = 0; i < buf.length; i += 1) sum += buf[i] * buf[i];
  return Math.sqrt(sum / Math.max(1, buf.length));
}

async function startMic() {
  if (running) return;
  els.startBtn.disabled = true;
  setClientState("checking model");
  try {
    const status = await refreshStatus();
    if (status.status !== "loaded") throw new Error(`model is not loaded: ${status.status}`);
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("getUserMedia is not supported by this browser");

    setClientState("requesting mic permission");
    addLog({ type: "client", response: "requesting microphone permission" });
    const selectedInput = els.inputDevice.value;
    const audioConstraint = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
      channelCount: 1,
    };
    if (selectedInput) audioConstraint.deviceId = { exact: selectedInput };
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraint });
    await refreshDevices();

    setClientState("opening audio context");
    audioContext = new AudioContext();
    if (audioContext.state === "suspended") await audioContext.resume();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    muteGain = audioContext.createGain();
    muteGain.gain.value = 0;
    sourceNode.connect(processor);
    processor.connect(muteGain);
    muteGain.connect(audioContext.destination);

    setClientState("connecting websocket");
    const proto = location.protocol === "https:" ? "wss" : "ws";
    expectedSocketClose = false;
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = onMessage;
    ws.onclose = (event) => {
      clearWsPingTimer();
      running = false;
      els.startBtn.disabled = loadedStatus?.status !== "loaded";
      els.stopBtn.disabled = true;
      els.adapterSelect.disabled = loadedStatus?.status !== "loaded";
      const normal = expectedSocketClose || event.code === 1000 || event.code === 1001;
      setClientState(normal ? "stopped" : "socket closed unexpectedly");
      addLog({
        type: "socket",
        label: normal ? "closed normally" : "closed unexpectedly",
        response: `WebSocket closed code=${event.code}${event.reason ? " reason=" + event.reason : ""}`,
      });
    };
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = () => reject(new Error("WebSocket connection failed"));
    });
    startWsPingTimer();
    ws.send(
      JSON.stringify({
        type: "start",
        sampleRate: audioContext.sampleRate,
        vadThreshold: parseFloat(els.vadThreshold.value),
        tickSeconds: parseFloat(els.tickSeconds.value),
        maxContextSeconds: parseFloat(els.contextSeconds.value),
      }),
    );
    processor.onaudioprocess = (event) => {
      if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      if (els.bargeIn.checked && currentAssistantAudio && !currentAssistantAudio.paused) {
        if (localRms(copy) >= parseFloat(els.vadThreshold.value)) {
          currentAssistantAudio.pause();
          currentAssistantAudio.currentTime = 0;
        }
      }
      ws.send(floatToInt16Buffer(copy));
    };
    running = true;
    els.startBtn.disabled = true;
    els.stopBtn.disabled = false;
    els.adapterSelect.disabled = true;
    els.sampleRate.textContent = `${audioContext.sampleRate} Hz`;
    setClientState("recording", true);
    addLog({ type: "client", response: "microphone streaming started" });
  } catch (err) {
    addLog({ type: "error", message: String(err) });
    setClientState("start failed");
    els.startBtn.disabled = false;
    els.stopBtn.disabled = true;
    els.adapterSelect.disabled = loadedStatus?.status !== "loaded";
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    if (audioContext) await audioContext.close().catch(() => {});
    ws = null;
    mediaStream = null;
    audioContext = null;
    processor = null;
    sourceNode = null;
    muteGain = null;
  }
}

async function stopMic() {
  running = false;
  if (processor) processor.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (muteGain) muteGain.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (audioContext) await audioContext.close();
  processor = null;
  sourceNode = null;
  muteGain = null;
  mediaStream = null;
  audioContext = null;
  if (ws && ws.readyState === WebSocket.OPEN) {
    expectedSocketClose = true;
    ws.send(JSON.stringify({ type: "stop" }));
    setTimeout(() => ws.close(), 250);
  }
  clearWsPingTimer();
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.adapterSelect.disabled = loadedStatus?.status !== "loaded";
  setClientState("stopped");
}

function updateProbGraph(tick) {
  probHistory.push({
    wait: tick.p_WAIT ?? 0,
    back: tick.p_BACKCHANNEL ?? 0,
    support: tick.p_SUPPORT ?? 0,
  });
  if (probHistory.length > maxHistory) probHistory.shift();
  drawProbGraph();
}

function drawLine(ctx, values, color, w, h) {
  if (values.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (maxHistory - 1)) * w;
    const y = h - v * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawProbGraph() {
  const canvas = els.probCanvas;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0c1014";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#24303b";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  drawLine(ctx, probHistory.map((p) => p.wait), "#6fa8ff", w, h);
  drawLine(ctx, probHistory.map((p) => p.back), "#ffd36f", w, h);
  drawLine(ctx, probHistory.map((p) => p.support), "#65d68a", w, h);
}

function addLog(row) {
  const tr = document.createElement("tr");
  const probabilityEntries = row.label_probabilities
    ? Object.entries(row.label_probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3)
    : [];
  const probs = probabilityEntries.length
    ? probabilityEntries.map(([label, value]) => `${label} ${value.toFixed(2)}`).join(" / ")
    : row.p_WAIT !== undefined
      ? `W ${row.p_WAIT.toFixed(2)} / B ${row.p_BACKCHANNEL.toFixed(2)} / S ${row.p_SUPPORT.toFixed(2)}`
      : "";
  const lat = row.latency_ms
    ? `label ${fmtMs(row.latency_ms.label_total)} / text ${fmtMs(row.latency_ms.text_generate)} / talker ${fmtMs(row.latency_ms.talker_generate)}`
    : "";
  [
    row.clock_s !== undefined ? row.clock_s.toFixed(2) : "",
    row.type || "",
    row.label || "",
    row.silence_elapsed !== undefined ? row.silence_elapsed.toFixed(2) + "s" : "",
    probs,
    lat,
    row.generated_response || row.response || row.message || "",
  ].forEach((value) => {
    const cell = document.createElement("td");
    cell.textContent = String(value);
    tr.appendChild(cell);
  });
  els.logBody.prepend(tr);
  while (els.logBody.children.length > maxLogRows) els.logBody.lastChild.remove();
}

function renderActionProbabilities(probabilities) {
  const values = probabilities || {};
  const labels = Object.keys(values).length
    ? Object.keys(values)
    : loadedStatus?.active_labels || [];
  const entries = labels.map((label) => [label, Number(values[label] ?? 0)]);
  const showExtended = entries.length > 3;
  els.actionProbabilitiesTitle.hidden = !showExtended;
  els.actionProbabilities.hidden = !showExtended;
  els.actionProbabilities.innerHTML = "";
  if (!showExtended) return;
  entries.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "action-probability";
    const code = loadedStatus?.active_codebook?.[label] || "";
    const name = document.createElement("span");
    name.textContent = `${code} ${label}`.trim();
    const meter = document.createElement("meter");
    meter.min = 0;
    meter.max = 1;
    meter.value = value;
    const score = document.createElement("b");
    score.textContent = Number(value).toFixed(3);
    row.append(name, meter, score);
    els.actionProbabilities.appendChild(row);
  });
}

function updateTick(data) {
  els.lastLabel.textContent = data.label;
  const code = data.control_code || loadedStatus?.active_codebook?.[data.label] || "";
  els.bigLabel.textContent = `${code} ${data.label}`.trim();
  els.mWait.value = data.p_WAIT;
  els.mBack.value = data.p_BACKCHANNEL;
  els.mSupport.value = data.p_SUPPORT;
  els.pWait.textContent = data.p_WAIT.toFixed(3);
  els.pBack.textContent = data.p_BACKCHANNEL.toFixed(3);
  els.pSupport.textContent = data.p_SUPPORT.toFixed(3);
  els.latPrep.textContent = fmtMs(data.latency_ms.prep);
  els.latForward.textContent = fmtMs(data.latency_ms.forward);
  els.latLabel.textContent = fmtMs(data.latency_ms.label_total);
  els.latText.textContent = fmtMs(data.latency_ms.text_generate);
  els.latTalker.textContent = fmtMs(data.latency_ms.talker_generate);
  if (data.generated_response) els.assistantText.textContent = data.generated_response;
  renderActionProbabilities(data.label_probabilities || {});
  updateProbGraph(data);
  addLog(data);
}

async function playAssistantAudio(audio) {
  if (els.outputDevice.value && typeof audio.setSinkId === "function") {
    await audio.setSinkId(els.outputDevice.value).catch((err) => addLog({ type: "audio", message: `setSinkId failed: ${err}` }));
  }
  currentAssistantAudio = audio;
  await audio.play().catch((err) => addLog({ type: "audio", message: String(err) }));
}

function updateGenerationStatus(data) {
  if (data.latency_ms?.text_generate !== undefined) els.latText.textContent = fmtMs(data.latency_ms.text_generate);
  if (data.latency_ms?.talker_generate !== undefined) els.latTalker.textContent = fmtMs(data.latency_ms.talker_generate);
  if (data.generated_response || data.response_text) els.assistantText.textContent = data.generated_response || data.response_text;
  addLog(data);
}

function onMessage(event) {
  const data = JSON.parse(event.data);
  if (data.type === "session_started") {
    els.sessionId.textContent = data.session_id;
    els.logPath.textContent = `logs: artifacts/omni3b_realtime_webui_v1/${data.session_id}`;
    addLog({ type: "session", response: data.session_id });
  } else if (data.type === "vad") {
    els.clock.textContent = `${data.clock_s.toFixed(2)}s`;
    els.rms.textContent = data.rms.toFixed(4);
    els.vadState.textContent = data.speaking ? "speaking" : "below threshold";
    els.vadState.style.color = data.speaking ? "#65d68a" : "#ffd36f";
    els.silence.textContent = `${data.silence_elapsed.toFixed(2)}s`;
  } else if (data.type === "tick") {
    updateTick(data);
  } else if (data.type === "event_detected") {
    addLog({ ...data, response: `event: ${data.previous_label} -> ${data.label}` });
  } else if (data.type === "generation_status") {
    updateGenerationStatus(data);
  } else if (data.type === "skip") {
    skipped += 1;
    els.skips.textContent = String(skipped);
    addLog({ ...data, label: "skip", response: data.reason });
  } else if (data.type === "assistant_audio_ready") {
    updateGenerationStatus(data);
    const audio = new Audio(`${data.audio_url}?t=${Date.now()}`);
    playAssistantAudio(audio);
  } else if (data.type === "assistant_audio") {
    const audio = new Audio(`data:${data.mime};base64,${data.audio_b64}`);
    playAssistantAudio(audio);
  } else if (data.type === "session_stopped") {
    expectedSocketClose = true;
    addLog({ type: "session", response: `stopped: ${data.path}` });
  } else if (data.type === "error") {
    addLog({ type: "error", message: data.message });
  }
}

function clearUi() {
  probHistory = [];
  skipped = 0;
  els.logBody.innerHTML = "";
  els.assistantText.textContent = "-";
  els.skips.textContent = "0";
  renderActionProbabilities({});
  drawProbGraph();
}

els.loadBtn.addEventListener("click", loadModel);
els.adapterSelect.addEventListener("change", () => switchAdapter().catch((err) => {
  addLog({ type: "adapter", message: String(err) });
  if (loadedStatus?.active_adapter) els.adapterSelect.value = loadedStatus.active_adapter;
}));
els.startBtn.addEventListener("click", () => startMic().catch((err) => addLog({ type: "error", message: String(err) })));
els.stopBtn.addEventListener("click", () => stopMic().catch((err) => addLog({ type: "error", message: String(err) })));
els.clearBtn.addEventListener("click", clearUi);
els.refreshDevicesBtn.addEventListener("click", refreshDevices);
els.vadThreshold.addEventListener("input", () => {
  els.vadValue.textContent = Number(els.vadThreshold.value).toFixed(3);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "set", vadThreshold: parseFloat(els.vadThreshold.value) }));
});
els.tickSeconds.addEventListener("change", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "set", tickSeconds: parseFloat(els.tickSeconds.value) }));
});
els.contextSeconds.addEventListener("change", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "set", maxContextSeconds: parseFloat(els.contextSeconds.value) }));
});

window.addEventListener("resize", drawProbGraph);
refreshStatus().catch(() => setStatus("model: server unavailable"));
refreshDevices();
drawProbGraph();
