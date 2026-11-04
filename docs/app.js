"use strict";

const STEPS = [
  { key: "decode", label: "Open image" },
  { key: "preprocess", label: "Prepare image" },
  { key: "classify", label: "Score the image" },
  { key: "explain", label: "Where it looked" },
  { key: "decide", label: "Estimate" },
];
const STEP_PACE_MS = 380;
const CHECK_SVG = '<svg viewBox="0 0 12 12"><path d="M2.5 6.2 5 8.5 9.5 3.8"/></svg>';

// Measured once in the project's reports (graz.md, gradcam.md, results.md); not in data.json
const GRAZ_AUC = 0.976;
const CAM_HIT_RATE = 0.61;

// What a radiologist is asked to do next, per level. Never a diagnosis.
const NEXT_STEP = {
  likely: "Prioritise this image for radiologist review. Start with the highlighted region, but read the whole " +
    "image: the model's attention can fall away from a fracture.",
  possible: "Radiologist review recommended. The model is uncertain, and a fracture may lie outside the " +
    "highlighted region.",
  unlikely: "No region flagged. Read as usual: a low score does not exclude a fracture.",
};

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const els = {
  status: $("status"), statusText: $("status-text"),
  upload: $("upload-view"), analysis: $("analysis-view"), reportView: $("report-view"), report: $("report"),
  dropzone: $("dropzone"), fileInput: $("file-input"),
  samples: $("samples"), sampleGrid: $("sample-grid"),
  stage: $("stage"), viewerEmpty: $("viewer-empty"),
  original: $("img-original"), attention: $("img-attention"), input: $("img-input"),
  peak: $("peak"), fileMeta: $("file-meta"),
  steps: $("steps"), result: $("result"), finding: $("finding"), note: $("note"),
  scale: $("scale"), evidence: $("evidence"), limits: $("limits"),
  score: $("score"), threshold: $("threshold"),
  referenceRow: $("reference-row"), reference: $("reference"),
  focusRow: $("focus-row"), focus: $("focus"),
  boxes: $("boxes"), toggleBoxes: $("toggle-boxes"),
  panelActions: $("panel-actions"),
  toast: $("toast"), dragOverlay: $("drag-overlay"),
  tabs: [...document.querySelectorAll(".segmented button")],
};

let modelInfo = null;
let runId = 0;
let objectUrl = null;
// The last finished analysis, for the report
let last = null;

/* ---------- Startup ---------- */

let siteData = null;
let modelReady = null;

async function init() {
  try {
    siteData = await (await fetch("data.json")).json();
    modelInfo = siteData.info;
  } catch {
    els.status.classList.add("error");
    els.statusText.textContent = "Could not load the demo data";
    return;
  }
  showFacts();
  loadSamples();
  // The model is ~22 MB: download it in the background while the visitor looks around
  modelReady = WebModel.load((p) => { els.statusText.textContent = `Loading model ${Math.round(p * 100)}%`; })
    .then(() => {
      els.status.classList.add("ready");
      els.statusText.textContent = "Model ready";
    })
    .catch(() => {
      els.status.classList.add("error");
      els.statusText.textContent = "Model failed to load";
      throw new Error("The model could not be loaded. Please reload the page.");
    });
  modelReady.catch(() => {});
}

// Counts on the held-out test split, at the chosen cut-off
function testCounts() {
  const { fractured, normal, threshold } = siteData.performance;
  const caught = fractured.filter((s) => s >= threshold).length;
  const falseAlarms = normal.filter((s) => s >= threshold).length;
  return { P: fractured.length, N: normal.length, caught, falseAlarms };
}

function showFacts() {
  const t = modelInfo.test;
  const fact = (value, text) => `<div><dt>${value}</dt><dd>${text}</dd></div>`;
  $("facts").innerHTML =
    fact(t.roc_auc.toFixed(2), `ROC-AUC on ${t.images} held-out X-rays (1.0 is perfect, 0.5 is chance)`) +
    fact(pct(t.sensitivity), `of fractures flagged in that test, with ${pct(t.specificity)} of healthy X-rays cleared`) +
    fact("2", "public datasets: FracAtlas (Bangladesh) and GRAZPEDWRI-DX (Austria)");
}

function loadSamples() {
  const samples = siteData.samples;
  if (!samples.length) return;
  for (const s of samples) {
    const btn = document.createElement("button");
    btn.className = "sample";
    btn.title = `${s.region} X-ray from the test set: ${s.label.toLowerCase()}, model ${s.outcome}`;
    btn.innerHTML = `<img src="samples/${s.id}.jpg" alt="" loading="lazy">` +
      `<span>${s.region}<em class="outcome ${["caught", "cleared"].includes(s.outcome) ? "good" : "bad"}">${s.outcome}</em></span>`;
    btn.addEventListener("click", () => analyze(null, s));
    els.sampleGrid.appendChild(btn);
  }
  els.samples.hidden = false;
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
      const blob = await (await fetch(`samples/${sample.id}.jpg`)).blob();
      file = new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" });
    } catch {
      showToast("Could not load the sample image");
      return;
    }
  }
  const id = ++runId;
  resetAnalysis();
  showView("analysis");
  els.fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;

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
    await modelReady;
    data = await WebModel.analyze(file, modelInfo, siteData.level_text);
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
  last = { data, sample, name: file.name, bytes: file.size, at: new Date() };
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
  els.result.dataset.level = data.level;
  els.finding.textContent = data.finding;
  els.note.textContent = data.confidence_note;
  els.score.textContent = data.score.toFixed(3);
  els.threshold.textContent = data.threshold.toFixed(3);
  for (const li of els.scale.children) li.classList.toggle("on", li.dataset.level === data.level);
  els.evidence.textContent = levelEvidence(data.level);
  const t = modelInfo.test;
  els.limits.textContent = `In testing it flagged ${pct(t.sensitivity)} of fractures and cleared ` +
    `${pct(t.specificity)} of healthy X-rays. Not validated for clinical use.`;

  els.referenceRow.hidden = !sample;
  if (sample) els.reference.textContent = sample.label;
  if (sample?.boxes.length) {
    els.focus.textContent = focusText(data, sample);
    els.focusRow.hidden = false;
  }
  els.panelActions.hidden = false;
}

function levelEvidence(level) {
  const per100 = modelInfo.levels?.[level];
  return per100 == null ? "" : `In testing, of 100 X-rays rated "${level}", about ${per100} had a fracture.`;
}

function focusText(data, sample) {
  if (!data.positive) return "Not flagged: a miss";
  return focusOnBox(data, sample.boxes) ? "Yes, peak on the annotated region" : "No, peak outside the annotated region";
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
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (!els.reportView.hidden) {
    if (e.key === "Escape") showView("analysis");
    return;
  }
  if (els.analysis.hidden) return;
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

function showView(name) {
  els.upload.hidden = name !== "upload";
  els.analysis.hidden = name !== "analysis";
  els.reportView.hidden = name !== "report";
}

/* ---------- Reset ---------- */

$("new-analysis").addEventListener("click", goHome);

function goHome() {
  runId++;
  showView("upload");
  resetAnalysis();
}

function resetAnalysis() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  last = null;
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
  els.panelActions.hidden = true;
  els.tabs.forEach((t) => { t.disabled = t.dataset.view !== "image"; });
  setView("image");
}

/* ---------- Report ---------- */

$("open-report").addEventListener("click", openReport);
$("report-back").addEventListener("click", () => showView("analysis"));
$("report-print").addEventListener("click", () => window.print());

// Where the attention peak sits, in thirds of the image as displayed (not the patient's left or right)
function regionText(peak) {
  const col = peak.x < 1 / 3 ? "left" : peak.x > 2 / 3 ? "right" : "centre";
  const row = peak.y < 1 / 3 ? "upper" : peak.y > 2 / 3 ? "lower" : "middle";
  return row === "middle" && col === "centre" ? "Centre of the image" : `${capitalise(row)} ${col} of the image`;
}

// The original, dimmed, with the heat map, the peak ring and any radiologist boxes drawn on top
async function annotatedImage(data, sample) {
  const [base, heat] = await Promise.all([loadImage(data.images.original), loadImage(data.images.attention)]);
  const canvas = Object.assign(document.createElement("canvas"), { width: base.width, height: base.height });
  const ctx = canvas.getContext("2d");
  ctx.filter = "brightness(0.55)";
  ctx.drawImage(base, 0, 0);
  ctx.filter = "none";
  ctx.drawImage(heat, 0, 0, base.width, base.height);
  const unit = Math.max(base.width, base.height);
  if (data.positive) {
    ctx.lineWidth = Math.max(2, unit / 300);
    ctx.strokeStyle = "#fff";
    ctx.beginPath();
    ctx.arc(data.peak.x * base.width, data.peak.y * base.height, unit * 0.045, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (const b of sample?.boxes ?? []) {
    ctx.setLineDash([unit / 90, unit / 140]);
    ctx.lineWidth = Math.max(1.5, unit / 400);
    ctx.strokeStyle = "#fff";
    ctx.strokeRect(b.x * base.width, b.y * base.height, b.w * base.width, b.h * base.height);
  }
  return canvas.toDataURL("image/jpeg", 0.92);
}

async function openReport() {
  if (!last) return;
  const { data, sample, name, bytes, at } = last;
  const t = modelInfo.test;
  const c = testCounts();
  const missedPer100 = Math.round((1 - c.caught / c.P) * 100);
  const ppv = c.caught / (c.caught + c.falseAlarms);
  const when = at.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  const two = (n) => String(n).padStart(2, "0");
  const reportId = `FS-${at.getFullYear()}${two(at.getMonth() + 1)}${two(at.getDate())}-${two(at.getHours())}${two(at.getMinutes())}`;
  const kv = (rows) => `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  const annotated = await annotatedImage(data, sample);

  els.report.innerHTML = `
    <header class="report-head">
      <div>
        <div class="report-kicker">AI-assisted fracture screening · research prototype</div>
        <h1>Screening report</h1>
      </div>
      <dl class="report-meta"><dt>Report</dt><dd>${reportId}</dd><dt>Generated</dt><dd>${escapeHtml(when)}</dd></dl>
    </header>
    <p class="report-warning"><strong>Not a diagnosis.</strong> This estimate comes from a research model that has not
      been clinically validated. It is meant to support, never replace, review by a qualified radiologist.</p>

    <section>
      <h2>1. Study</h2>
      ${kv([
        ["Image", `<span class="mono">${escapeHtml(name)}</span>`],
        ["Size", `${data.size.width} × ${data.size.height} px · ${formatBytes(bytes)}`],
        ["Body region", sample ? `${capitalise(sample.region)} (from the dataset label)` : "Not provided"],
        ["Source", sample ? "Held-out FracAtlas test image, not used in training" : "Uploaded on this device, not stored"],
      ])}
    </section>

    <section>
      <h2>2. Model estimate</h2>
      <p class="report-finding"><span class="level-mark" data-level="${data.level}"></span>${escapeHtml(data.finding)}</p>
      <p>${escapeHtml(data.confidence_note)}</p>
      <ol class="scale">${["unlikely", "possible", "likely"].map((l) =>
        `<li class="${l === data.level ? "on" : ""}">${capitalise(l)}</li>`).join("")}</ol>
      ${kv([
        ["Model score", `<span class="mono">${data.score.toFixed(3)}</span> on a 0 to 1 scale; flagged at
          <span class="mono">${data.threshold.toFixed(3)}</span> or above`],
        ["What this level meant in testing", levelEvidence(data.level).replace("In testing, of", "Of")],
        ["Highlighted region", data.positive ? `${regionText(data.peak)}, as displayed` : "None: no fracture suggested"],
      ])}
    </section>

    <section>
      <h2>3. Images</h2>
      <div class="report-figures">
        <figure><img src="${data.images.original}" alt="Original X-ray"><figcaption>Original image.</figcaption></figure>
        <figure><img src="${annotated}" alt="X-ray with the model's attention map"><figcaption>Where the model looked
          (Grad-CAM).${data.positive ? " Ring: strongest response." : ""}${sample?.boxes.length ? " Dashed box: radiologist's annotation." : ""}
          This is not an outline of a fracture.</figcaption></figure>
      </div>
    </section>

    <section>
      <h2>4. Suggested review</h2>
      <p>${NEXT_STEP[data.level]}</p>
    </section>

    ${sample ? `<section>
      <h2>5. Reference for this test image</h2>
      ${kv([
        ["Radiologist label", sample.label],
        ...(sample.boxes.length ? [["Attention vs annotation", focusText(data, sample)]] : []),
      ])}
    </section>` : ""}

    <section>
      <h2>${sample ? 6 : 5}. What the model reviewed, and how it was evaluated</h2>
      <div class="report-cols">
        <div>
          <h3>Reviewed for this report</h3>
          <ul>
            <li>This single image, converted to grayscale and resized to 384 × 384 pixels.</li>
            <li>The image and its mirror image; the two scores are averaged.</li>
            <li>No patient history, symptoms, other views or previous studies.</li>
          </ul>
        </div>
        <div>
          <h3>Evaluated before release</h3>
          <ul>
            <li>${t.images} held-out FracAtlas X-rays (${c.P} fractured): ROC-AUC ${t.roc_auc.toFixed(2)},
              ${c.caught} of ${c.P} fractures flagged, ${pct(1 - c.falseAlarms / c.N)} of healthy X-rays cleared.</li>
            <li>Held-out patients from a second dataset (children's wrists, Graz): ROC-AUC ${GRAZ_AUC.toFixed(2)}.
              Other patients from that hospital were used in training.</li>
            <li>Cut-off chosen on separate validation images: 95% confidence of flagging at least 85% of fractures.</li>
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>${sample ? 7 : 6}. Limitations</h2>
      <ul>
        <li>Not validated in clinical use, or at any hospital outside its training data.</li>
        <li>In testing it missed about ${missedPer100} in 100 fractures. A low score does not exclude a fracture.</li>
        <li>Many flags are false alarms: about ${pct(ppv)} of flagged test X-rays had a fracture.</li>
        <li>Casts and metal implants can raise the score whether or not a fracture is present.</li>
        <li>The heat map is coarse (32-pixel cells). Its peak was near the annotated fracture in about
          ${pct(CAM_HIT_RATE)} of the fractures it flagged.</li>
        <li>Few hip and shoulder fractures were tested, so results there are less certain.</li>
      </ul>
    </section>

    <footer class="report-foot">
      <span>${escapeHtml(modelInfo.model)} · trained on ${escapeHtml(modelInfo.dataset)} · computed on this device</span>
      <span>vishwaraj25.github.io/xray-fracture-detection</span>
    </footer>`;
  showView("report");
  els.reportView.scrollTop = 0;
}

/* ---------- Helpers ---------- */

function setImage(img, src) {
  return new Promise((resolve) => {
    img.onload = img.onerror = () => resolve();
    img.src = src;
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
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
const capitalise = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const formatBytes = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} KB`);

renderSteps();
init();
