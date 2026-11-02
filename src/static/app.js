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
  about: $("about"), modelFigures: $("model-figures"),
  newAnalysis: $("new-analysis"), toast: $("toast"), dragOverlay: $("drag-overlay"),
  tabs: [...document.querySelectorAll(".segmented button")],
};

let modelInfo = null;
let runId = 0;
let objectUrl = null;

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
      btn.addEventListener("click", () => analyzeSample(s));
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

async function analyzeSample(sample) {
  try {
    const blob = await (await fetch(`/api/samples/${sample.id}/file`)).blob();
    const file = new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" });
    analyze(file, sample.label);
  } catch {
    showToast("Could not load the sample image");
  }
}

/* ---------- Analysis flow ---------- */

async function analyze(file, referenceLabel = null) {
  const id = ++runId;
  resetAnalysis();
  els.upload.hidden = true;
  els.analysis.hidden = false;
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
    await onStepDone(STEPS[i].key, data);
    await sleep(STEP_PACE_MS * 0.4);
  }
  if (id !== runId) return;
  showResult(data, referenceLabel);
}

async function onStepDone(key, data) {
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
    enableTab("attention");
    els.stage.classList.remove("scanning");
    if (data.positive) setView("attention");
  }
}

function showResult(data, referenceLabel) {
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

  els.referenceRow.hidden = !referenceLabel;
  if (referenceLabel) els.reference.textContent = referenceLabel;

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

/* ---------- View switching ---------- */

els.tabs.forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));

window.addEventListener("keydown", (e) => {
  if (els.analysis.hidden || e.metaKey || e.ctrlKey || e.altKey) return;
  const view = { 1: "image", 2: "attention", 3: "input" }[e.key];
  if (view && !els.tabs.find((t) => t.dataset.view === view).disabled) setView(view);
  if (e.key === "Escape") goHome();
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
  els.analysis.hidden = true;
  els.upload.hidden = false;
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
  els.viewerEmpty.hidden = false;
  els.stage.classList.remove("scanning", "dimmed");
  els.result.hidden = true;
  els.about.hidden = true;
  els.meterFill.style.width = "0";
  els.tabs.forEach((t) => { t.disabled = t.dataset.view !== "image"; });
  setView("image");
}

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
