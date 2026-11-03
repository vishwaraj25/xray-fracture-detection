"use strict";

const STEPS = [
  { key: "decode", label: "Load image" },
  { key: "preprocess", label: "Preprocess" },
  { key: "classify", label: "Model inference" },
  { key: "explain", label: "Attention map" },
  { key: "decide", label: "Decision" },
];
const STEP_PACE_MS = 380;
const CHECK_SVG = '<svg viewBox="0 0 12 12"><path d="M2.5 6.2 5 8.5 9.5 3.8"/></svg>';

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const els = {
  status: $("status"), statusText: $("status-text"),
  upload: $("upload-view"), analysis: $("analysis-view"),
  dropzone: $("dropzone"), fileInput: $("file-input"),
  samples: $("samples"), sampleGrid: $("sample-grid"),
  stage: $("stage"), viewerEmpty: $("viewer-empty"),
  original: $("img-original"), attention: $("img-attention"), input: $("img-input"),
  peak: $("peak"), fileMeta: $("file-meta"),
  steps: $("steps"), result: $("result"), finding: $("finding"), note: $("note"),
  meterFill: $("meter-fill"), meterThreshold: $("meter-threshold"),
  score: $("score"), threshold: $("threshold"),
  referenceRow: $("reference-row"), reference: $("reference"),
  focusRow: $("focus-row"), focus: $("focus"),
  boxes: $("boxes"), toggleBoxes: $("toggle-boxes"),
  performance: $("performance-view"),
  navAnalyse: $("nav-analyse"), navPerformance: $("nav-performance"),
  about: $("about"), modelFigures: $("model-figures"),
  newAnalysis: $("new-analysis"), toast: $("toast"), dragOverlay: $("drag-overlay"),
  tabs: [...document.querySelectorAll(".segmented button")],
};

let modelInfo = null;
let runId = 0;
let objectUrl = null;
// Which part of the "Analyse" section to show when coming back from Performance
let analyseView = "upload";

/* ---------- Startup ---------- */

async function init() {
  try {
    const res = await fetch("/api/info");
    if (!res.ok) throw new Error();
    modelInfo = await res.json();
    els.status.classList.add("ready");
    els.statusText.textContent = `${modelInfo.model} · ready`;
  } catch {
    els.status.classList.add("error");
    els.statusText.textContent = "Model unavailable";
  }
  loadSamples();
}

async function loadSamples() {
  try {
    const samples = await (await fetch("/api/samples")).json();
    if (!samples.length) return;
    for (const s of samples) {
      const btn = document.createElement("button");
      btn.className = "sample";
      btn.title = `${s.region} X-ray from the test set`;
      btn.innerHTML = `<img src="/api/samples/${s.id}/thumb" alt="" loading="lazy"><span>${s.region}</span>`;
      btn.addEventListener("click", () => analyze(null, s));
      els.sampleGrid.appendChild(btn);
    }
    els.samples.hidden = false;
  } catch { /* samples are optional */ }
}

/* ---------- Input ---------- */

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files[0];
  if (file) analyze(file);
  els.fileInput.value = "";
});

els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
});

let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (![...e.dataTransfer.types].includes("Files")) return;
  e.preventDefault();
  dragDepth++;
  els.dragOverlay.hidden = false;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) els.dragOverlay.hidden = true;
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  els.dragOverlay.hidden = true;
  const file = e.dataTransfer.files[0];
  if (file) analyze(file);
});
window.addEventListener("paste", (e) => {
  const file = [...e.clipboardData.files][0];
  if (file) analyze(file);
});

/* ---------- Analysis flow ---------- */

// `sample` is a held-out test image with its reference label and the radiologist's boxes
async function analyze(file, sample = null) {
  if (sample) {
    try {
      const blob = await (await fetch(`/api/samples/${sample.id}/file`)).blob();
      file = new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" });
    } catch {
      showToast("Could not load the sample image");
      return;
    }
  }
  const id = ++runId;
  resetAnalysis();
  analyseView = "analysis";
  showSection("analyse");
  els.fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;

  // Show the image straight away when the browser can render it (not DICOM)
  if (file.type.startsWith("image/")) {
    objectUrl = URL.createObjectURL(file);
    await setImage(els.original, objectUrl);
    if (id !== runId) return;
    els.viewerEmpty.hidden = true;
    els.original.classList.add("shown");
  }
  els.stage.classList.add("scanning");
  setStep(0, "active");

  const started = performance.now();
  let data;
  try {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/analyze", { method: "POST", body });
    data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed");
  } catch (err) {
    if (id !== runId) return;
    showToast(err.message || "Analysis failed");
    goHome();
    return;
  }
  if (id !== runId) return;

  // Keep the scan visible briefly so a fast result still reads as a process
  await sleep(Math.max(0, 600 - (performance.now() - started)));

  for (let i = 0; i < STEPS.length; i++) {
    if (id !== runId) return;
    const info = data.steps.find((s) => s.key === STEPS[i].key);
    setStep(i, "active");
    await sleep(i === 0 ? 120 : STEP_PACE_MS * 0.6);
    setStep(i, "done", info);
    await onStepDone(STEPS[i].key, data, sample);
    await sleep(STEP_PACE_MS * 0.4);
  }
  if (id !== runId) return;
  showResult(data, sample);
}

async function onStepDone(key, data, sample) {
  if (key === "decode") {
    await setImage(els.original, data.images.original);
    els.viewerEmpty.hidden = true;
    els.original.classList.add("shown");
  } else if (key === "preprocess") {
    els.input.src = data.images.model_input;
    enableTab("input");
  } else if (key === "explain") {
    els.attention.src = data.images.attention;
    els.peak.style.left = `${data.peak.x * 100}%`;
    els.peak.style.top = `${data.peak.y * 100}%`;
    // Only point at a location when the model actually flags a fracture
    els.peak.hidden = !data.positive;
    drawBoxes(sample);
    enableTab("attention");
    els.stage.classList.remove("scanning");
    if (data.positive) setView("attention");
  }
}

function showResult(data, sample) {
  els.result.hidden = false;
  els.result.classList.toggle("positive", data.positive);
  els.finding.textContent = data.finding;
  els.note.textContent = data.confidence_note;
  els.score.textContent = data.score.toFixed(3);
  els.threshold.textContent = data.threshold.toFixed(3);
  els.meterThreshold.style.left = `${data.threshold * 100}%`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    els.meterFill.style.width = `${Math.max(data.score * 100, 0.8)}%`;
  }));

  els.referenceRow.hidden = !sample;
  if (sample) els.reference.textContent = sample.label;
  if (sample?.boxes.length) {
    els.focus.textContent = !data.positive ? "Fracture missed"
      : focusOnBox(data, sample.boxes) ? "On the fracture" : "Away from the fracture";
    els.focusRow.hidden = false;
  }

  if (modelInfo) {
    const t = modelInfo.test;
    els.modelFigures.innerHTML = [
      ["Model", modelInfo.model],
      ["Trained on", `${modelInfo.dataset}`],
      ["Test sensitivity", pct(t.sensitivity)],
      ["Test specificity", pct(t.specificity)],
      ["Test ROC-AUC", t.roc_auc.toFixed(3)],
    ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
    els.about.hidden = false;
  }
}

/* ---------- Steps ---------- */

function renderSteps() {
  els.steps.innerHTML = STEPS.map((s) => `
    <li class="step">
      <span class="step-icon">${CHECK_SVG}</span>
      <span><span class="step-label">${s.label}</span><span class="step-detail"></span></span>
      <span class="step-ms"></span>
    </li>`).join("");
}

function setStep(index, state, info) {
  const li = els.steps.children[index];
  li.classList.remove("active", "done");
  li.classList.add(state);
  if (info) {
    li.querySelector(".step-detail").textContent = info.detail;
    li.querySelector(".step-ms").textContent = `${info.ms} ms`;
  }
}

/* ---------- Radiologist boxes ---------- */

function drawBoxes(sample) {
  if (!sample?.boxes.length) return;
  els.boxes.innerHTML = sample.boxes.map((b) => `<div class="box" style="left:${b.x * 100}%;top:${b.y * 100}%;` +
    `width:${b.w * 100}%;height:${b.h * 100}%"></div>`).join("");
  els.toggleBoxes.hidden = false;
}

// The same check as reports/gradcam.md: is the attention peak within one Grad-CAM cell of a box?
function focusOnBox(data, boxes) {
  const { width, height } = data.size;
  const cell = (modelInfo?.cam_cell ?? 1 / 12) * Math.max(width, height);
  const mx = cell / width, my = cell / height;
  const { x, y } = data.peak;
  return boxes.some((b) => x >= b.x - mx && x <= b.x + b.w + mx && y >= b.y - my && y <= b.y + b.h + my);
}

function setBoxesVisible(on) {
  els.toggleBoxes.setAttribute("aria-pressed", String(on));
  els.stage.classList.toggle("no-boxes", !on);
}

els.toggleBoxes.addEventListener("click", () =>
  setBoxesVisible(els.toggleBoxes.getAttribute("aria-pressed") !== "true"));

/* ---------- View switching ---------- */

els.tabs.forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));

window.addEventListener("keydown", (e) => {
  if (els.analysis.hidden || e.metaKey || e.ctrlKey || e.altKey) return;
  const view = { 1: "image", 2: "attention", 3: "input" }[e.key];
  if (view && !els.tabs.find((t) => t.dataset.view === view).disabled) setView(view);
  if (e.key === "Escape") goHome();
  if (e.key === "b" && !els.toggleBoxes.hidden) els.toggleBoxes.click();
});

function setView(view) {
  els.stage.dataset.view = view;
  els.stage.classList.toggle("dimmed", view === "attention");
  els.tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === view)));
}

function enableTab(view) {
  els.tabs.find((t) => t.dataset.view === view).disabled = false;
}

/* ---------- Reset ---------- */

els.newAnalysis.addEventListener("click", goHome);

function goHome() {
  runId++;
  analyseView = "upload";
  showSection("analyse");
  resetAnalysis();
}

function resetAnalysis() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  renderSteps();
  els.original.classList.remove("shown");
  els.original.removeAttribute("src");
  els.attention.removeAttribute("src");
  els.input.removeAttribute("src");
  els.peak.hidden = true;
  els.boxes.innerHTML = "";
  els.toggleBoxes.hidden = true;
  els.focusRow.hidden = true;
  setBoxesVisible(true);
  els.viewerEmpty.hidden = false;
  els.stage.classList.remove("scanning", "dimmed");
  els.result.hidden = true;
  els.about.hidden = true;
  els.meterFill.style.width = "0";
  els.tabs.forEach((t) => { t.disabled = t.dataset.view !== "image"; });
  setView("image");
}

/* ---------- Sections ---------- */

els.navAnalyse.addEventListener("click", () => showSection("analyse"));
els.navPerformance.addEventListener("click", () => showSection("performance"));

function showSection(name) {
  const perf = name === "performance";
  els.performance.hidden = !perf;
  els.upload.hidden = perf || analyseView !== "upload";
  els.analysis.hidden = perf || analyseView !== "analysis";
  els.navAnalyse.toggleAttribute("aria-current", !perf);
  els.navPerformance.toggleAttribute("aria-current", perf);
  if (perf) loadPerformance();
}

/* ---------- Performance: test-set scores at any threshold ---------- */

const SVG_NS = "http://www.w3.org/2000/svg";
// Strip plot geometry (viewBox units). Scores use a square-root axis: most normal
// X-rays score below 0.05 and would otherwise collapse into one blob at zero.
const STRIP = { w: 640, h: 196, left: 16, right: 16, axisY: 170 };
const ROWS = [
  { key: "fractured", label: "Fractured", top: 26 },
  { key: "normal", label: "Normal", top: 100 },
];
const ROW_H = 54;
const TICKS = [0, 0.01, 0.05, 0.15, 0.3, 0.5, 0.75, 1];

let perf = null;
let perfThreshold = 0;
let stripParts = null;

const stripX = (score) => STRIP.left + Math.sqrt(score) * (STRIP.w - STRIP.left - STRIP.right);
const stripScore = (x) => Math.min(1, Math.max(0, (x - STRIP.left) / (STRIP.w - STRIP.left - STRIP.right))) ** 2;

function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

async function loadPerformance() {
  if (perf) return;
  try {
    const res = await fetch("/api/performance");
    if (!res.ok) throw new Error();
    perf = await res.json();
  } catch {
    showToast("Test-set results are not available");
    return;
  }
  $("perf-total").textContent = perf.fractured.length + perf.normal.length;
  $("perf-auc").textContent = perf.roc_auc.toFixed(3);
  $("perf-chosen").textContent = perf.threshold.toFixed(3);
  buildStrip();
  buildRoc();
  setThreshold(perf.threshold);
}

function buildStrip() {
  const svg = svgEl("svg", { viewBox: `0 0 ${STRIP.w} ${STRIP.h}`, class: "strip-svg" });
  const flagZone = svgEl("rect", { class: "flag-zone", y: 8, height: STRIP.axisY - 8 }, svg);
  const dots = [];
  for (const row of ROWS) {
    const scores = perf[row.key];
    svgEl("text", { x: STRIP.left, y: row.top - 8, class: "row-label" }, svg).textContent =
      `${row.label} · ${scores.length}`;
    scores.forEach((score, i) => {
      // Deterministic vertical jitter so the picture is the same on every load
      const jitter = ((i * 0.6180339887) % 1) * (ROW_H - 8) + 4;
      dots.push({ score, node: svgEl("circle", { cx: stripX(score), cy: row.top + jitter, r: 2.6, class: "dot" }, svg) });
    });
  }
  svgEl("line", { x1: STRIP.left, x2: STRIP.w - STRIP.right, y1: STRIP.axisY, y2: STRIP.axisY, class: "axis" }, svg);
  for (const t of TICKS) {
    const x = stripX(t);
    svgEl("line", { x1: x, x2: x, y1: STRIP.axisY, y2: STRIP.axisY + 4, class: "axis" }, svg);
    svgEl("text", { x, y: STRIP.axisY + 17, class: "tick", "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle" },
      svg).textContent = t;
  }
  const handle = svgEl("g", { class: "handle" }, svg);
  svgEl("line", { y1: 8, y2: STRIP.axisY, class: "handle-line" }, handle);
  svgEl("rect", { y: 0, width: 54, height: 18, rx: 4, class: "handle-tag" }, handle);
  const tagText = svgEl("text", { y: 13, class: "handle-text", "text-anchor": "middle" }, handle);

  els.strip = $("strip");
  els.strip.replaceChildren(svg);
  stripParts = { svg, dots, handle, flagZone, tagText };

  const fromPointer = (e) => {
    const rect = svg.getBoundingClientRect();
    setThreshold(stripScore(((e.clientX - rect.left) / rect.width) * STRIP.w));
  };
  svg.addEventListener("pointerdown", (e) => {
    svg.setPointerCapture(e.pointerId);
    svg.classList.add("dragging");
    fromPointer(e);
  });
  svg.addEventListener("pointermove", (e) => { if (svg.hasPointerCapture(e.pointerId)) fromPointer(e); });
  svg.addEventListener("pointerup", () => svg.classList.remove("dragging"));
  els.strip.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    const u = Math.sqrt(perfThreshold);
    const next = { ArrowLeft: u - step, ArrowDown: u - step, ArrowRight: u + step, ArrowUp: u + step,
      Home: 0, End: 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    setThreshold(Math.min(1, Math.max(0, next)) ** 2);
  });
}

function rocPoints() {
  const all = [...perf.fractured.map((s) => [s, 1]), ...perf.normal.map((s) => [s, 0])].sort((a, b) => b[0] - a[0]);
  const P = perf.fractured.length, N = perf.normal.length;
  const points = [[0, 0]];
  let tp = 0, fp = 0;
  all.forEach(([score, y], i) => {
    y ? tp++ : fp++;
    if (i === all.length - 1 || all[i + 1][0] !== score) points.push([fp / N, tp / P]);
  });
  return points;
}

const ROC = { size: 260, pad: 34 };
const rocX = (fpr) => ROC.pad + fpr * (ROC.size - ROC.pad - 10);
const rocY = (tpr) => ROC.size - ROC.pad - tpr * (ROC.size - ROC.pad - 10);

function buildRoc() {
  const svg = svgEl("svg", { viewBox: `0 0 ${ROC.size} ${ROC.size}`, class: "roc-svg", role: "img",
    "aria-label": `ROC curve, area ${perf.roc_auc.toFixed(3)}` });
  for (const v of [0, 0.5, 1]) {
    svgEl("line", { x1: rocX(v), x2: rocX(v), y1: rocY(0), y2: rocY(1), class: "grid" }, svg);
    svgEl("line", { x1: rocX(0), x2: rocX(1), y1: rocY(v), y2: rocY(v), class: "grid" }, svg);
    svgEl("text", { x: rocX(v), y: rocY(0) + 15, class: "tick", "text-anchor": "middle" }, svg).textContent = v;
    svgEl("text", { x: rocX(0) - 7, y: rocY(v) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = v;
  }
  svgEl("line", { x1: rocX(0), y1: rocY(0), x2: rocX(1), y2: rocY(1), class: "diagonal" }, svg);
  const d = rocPoints().map(([x, y], i) => `${i ? "L" : "M"}${rocX(x).toFixed(1)},${rocY(y).toFixed(1)}`).join("");
  svgEl("path", { d, class: "roc-line" }, svg);
  svgEl("text", { x: rocX(0.5), y: ROC.size - 4, class: "axis-label", "text-anchor": "middle" }, svg)
    .textContent = "False alarm rate";
  svgEl("text", { x: 0, y: 0, class: "axis-label", "text-anchor": "middle",
    transform: `translate(10 ${rocY(0.5)}) rotate(-90)` }, svg).textContent = "Fractures caught";
  stripParts.rocDot = svgEl("circle", { r: 5, class: "roc-dot" }, svg);
  $("roc").replaceChildren(svg);
}

function setThreshold(t) {
  perfThreshold = t;
  const P = perf.fractured.length, N = perf.normal.length;
  const caught = perf.fractured.filter((s) => s >= t).length;
  const falseAlarms = perf.normal.filter((s) => s >= t).length;
  const flagged = caught + falseAlarms;

  const { dots, handle, flagZone, tagText, rocDot } = stripParts;
  const x = stripX(t);
  for (const dot of dots) dot.node.classList.toggle("flagged", dot.score >= t);
  flagZone.setAttribute("x", x);
  flagZone.setAttribute("width", Math.max(0, STRIP.w - STRIP.right - x));
  handle.setAttribute("transform", `translate(${x} 0)`);
  // Keep the value tag inside the plot at both ends
  const tagX = Math.min(Math.max(x, STRIP.left + 27), STRIP.w - STRIP.right - 27) - x;
  handle.querySelector(".handle-tag").setAttribute("x", tagX - 27);
  tagText.setAttribute("x", tagX);
  tagText.textContent = t.toFixed(3);
  rocDot.setAttribute("cx", rocX(falseAlarms / N));
  rocDot.setAttribute("cy", rocY(caught / P));

  $("perf-caught").textContent = caught;
  $("perf-caught-of").textContent = ` / ${P}`;
  $("perf-sens").textContent = `${pct(caught / P)} sensitivity · ${P - caught} missed`;
  $("perf-fp").textContent = falseAlarms;
  $("perf-fp-of").textContent = ` / ${N}`;
  $("perf-spec").textContent = `${pct(1 - falseAlarms / N)} of normal X-rays cleared`;
  $("perf-ppv").textContent = flagged ? pct(caught / flagged) : "–";
  $("perf-flagged").textContent = `${caught} of ${flagged} flagged X-rays`;

  els.strip.setAttribute("aria-valuenow", t.toFixed(3));
  els.strip.setAttribute("aria-valuetext",
    `threshold ${t.toFixed(3)}: ${caught} of ${P} fractures caught, ${falseAlarms} false alarms`);
  $("perf-reset").hidden = Math.abs(t - perf.threshold) < 1e-9;
}

$("perf-reset").addEventListener("click", () => setThreshold(perf.threshold));

/* ---------- Helpers ---------- */

function setImage(img, src) {
  return new Promise((resolve) => {
    img.onload = img.onerror = () => resolve();
    img.src = src;
  });
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { els.toast.hidden = true; }, 3500);
}

const pct = (v) => `${Math.round(v * 100)}%`;
const formatBytes = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} KB`);

renderSteps();
init();
