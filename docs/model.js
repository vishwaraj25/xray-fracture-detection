"use strict";

/* Threshold explorer on the model page: the test-set scores at any threshold. */

const $ = (id) => document.getElementById(id);
const pct = (v) => `${Math.round(v * 100)}%`;

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
let strip = null;

const stripX = (score) => STRIP.left + Math.sqrt(score) * (STRIP.w - STRIP.left - STRIP.right);
const stripScore = (x) => Math.min(1, Math.max(0, (x - STRIP.left) / (STRIP.w - STRIP.left - STRIP.right))) ** 2;

function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

async function init() {
  try {
    perf = (await (await fetch("data.json")).json()).performance;
  } catch {
    $("strip").textContent = "Test-set results could not be loaded.";
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

  strip = $("strip");
  strip.replaceChildren(svg);
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
  strip.addEventListener("keydown", (e) => {
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
    .textContent = "Healthy X-rays flagged";
  svgEl("text", { x: 0, y: 0, class: "axis-label", "text-anchor": "middle",
    transform: `translate(10 ${rocY(0.5)}) rotate(-90)` }, svg).textContent = "Fractures flagged";
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
  $("perf-spec").textContent = `${pct(1 - falseAlarms / N)} of healthy X-rays cleared`;
  $("perf-ppv").textContent = flagged ? pct(caught / flagged) : "–";
  $("perf-flagged").textContent = `${caught} of ${flagged} flagged X-rays`;

  strip.setAttribute("aria-valuenow", t.toFixed(3));
  strip.setAttribute("aria-valuetext",
    `threshold ${t.toFixed(3)}: ${caught} of ${P} fractures flagged, ${falseAlarms} false alarms`);
  $("perf-reset").hidden = Math.abs(t - perf.threshold) < 1e-9;
}

$("perf-reset").addEventListener("click", () => setThreshold(perf.threshold));

init();
