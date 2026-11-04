"use strict";

/*
 * The CNN running in the visitor's browser (onnxruntime-web). Mirrors FractureAnalyzer.analyze() in
 * src/predict.py: decode, pad to a square, score the image and its mirror, draw a Grad-CAM heat map.
 * analyze() returns the same fields as that Python method.
 */
const WebModel = (() => {
  const INPUT = 384;
  const LOAD = Math.round((INPUT * 8) / 7); // images are padded slightly larger, as in training
  const DISPLAY_MAX = 1024;
  const CAM_GRID = 12;
  const MODEL_BYTES = 22.4e6; // used for the progress bar when the server does not report a size
  let session = null;

  async function load(onProgress) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = true; // run inference in a worker so the page stays responsive
    const res = await fetch("model.onnx");
    if (!res.ok) throw new Error("Could not download the model");
    const total = Number(res.headers.get("Content-Length")) || MODEL_BYTES;
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.(Math.min(received / total, 1));
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  }

  const makeCanvas = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });

  // Longer side to `size`, centred on a black square (src/preprocessing.py: letterbox)
  function letterbox(bitmap, size) {
    const scale = size / Math.max(bitmap.width, bitmap.height);
    const w = Math.round(bitmap.width * scale), h = Math.round(bitmap.height * scale);
    const top = Math.floor((size - h) / 2), left = Math.floor((size - w) / 2);
    const canvas = makeCanvas(size, size);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, left, top, w, h);
    return { canvas, top, left, h, w };
  }

  function toGray(canvas) {
    const { data } = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height);
    const gray = new Uint8ClampedArray(data.length / 4);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = (data[4 * i] * 299 + data[4 * i + 1] * 587 + data[4 * i + 2] * 114) / 1000;
    }
    return gray;
  }

  function grayToCanvas(gray, size) {
    const canvas = makeCanvas(size, size);
    const image = new ImageData(size, size);
    for (let i = 0; i < gray.length; i++) {
      image.data[4 * i] = image.data[4 * i + 1] = image.data[4 * i + 2] = gray[i];
      image.data[4 * i + 3] = 255;
    }
    canvas.getContext("2d").putImageData(image, 0, 0);
    return canvas;
  }

  // Bilinear resize of a w x h float map (align_corners=False, like torch.nn.functional.interpolate)
  function resize(map, w, h, outW, outH) {
    const out = new Float32Array(outW * outH);
    for (let y = 0; y < outH; y++) {
      const fy = Math.min(Math.max(((y + 0.5) * h) / outH - 0.5, 0), h - 1);
      const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, h - 1), wy = fy - y0;
      for (let x = 0; x < outW; x++) {
        const fx = Math.min(Math.max(((x + 0.5) * w) / outW - 0.5, 0), w - 1);
        const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, w - 1), wx = fx - x0;
        out[y * outW + x] = (map[y0 * w + x0] * (1 - wx) + map[y0 * w + x1] * wx) * (1 - wy) +
                            (map[y1 * w + x0] * (1 - wx) + map[y1 * w + x1] * wx) * wy;
      }
    }
    return out;
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function levelOf(score, info) {
    if (score < info.threshold) return "unlikely";
    return score < info.likely_score ? "possible" : "likely";
  }

  async function analyze(file, info, levelText) {
    const steps = [];
    const step = (key, start, detail) => steps.push({ key, ms: Math.round(performance.now() - start), detail });

    let t = performance.now();
    let bitmap;
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
      throw new Error("Could not read this file. Please use a JPEG or PNG X-ray.");
    }
    const { width: w, height: h } = bitmap;
    step("decode", t, `${(file.type.split("/")[1] || "image").toUpperCase()} · ${w}×${h} px`);

    t = performance.now();
    const square = letterbox(bitmap, LOAD);
    const resized = makeCanvas(INPUT, INPUT);
    const rctx = resized.getContext("2d", { willReadFrequently: true });
    rctx.imageSmoothingQuality = "high";
    rctx.drawImage(grayToCanvas(toGray(square.canvas), LOAD), 0, 0, INPUT, INPUT);
    const gray = toGray(resized);
    const batch = new Float32Array(2 * INPUT * INPUT); // the image, then its mirror image
    for (let y = 0; y < INPUT; y++) {
      for (let x = 0; x < INPUT; x++) {
        batch[y * INPUT + x] = gray[y * INPUT + x];
        batch[INPUT * INPUT + y * INPUT + x] = gray[y * INPUT + (INPUT - 1 - x)];
      }
    }
    const modelInput = resized.toDataURL("image/png");
    step("preprocess", t, `made square and sharper · ${INPUT}×${INPUT} px`);

    t = performance.now();
    const out = await session.run({ gray: new ort.Tensor("float32", batch, [2, 1, INPUT, INPUT]) });
    const logits = out.logit.data;
    const score = (sigmoid(logits[0]) + sigmoid(logits[1])) / 2;
    step("classify", t, "AI model scores the X-ray");

    t = performance.now();
    const grid = Float32Array.from(out.cam.data.slice(0, CAM_GRID * CAM_GRID), (v) => Math.max(v, 0));
    let heat = resize(grid, CAM_GRID, CAM_GRID, INPUT, INPUT);
    let heatMax = 1e-8;
    for (const v of heat) if (v > heatMax) heatMax = v;
    heat = heat.map((v) => v / heatMax);

    // Crop the padding off the heat map and stretch it onto the displayed image
    const displayScale = Math.min(1, DISPLAY_MAX / Math.max(w, h));
    const dw = Math.round(w * displayScale), dh = Math.round(h * displayScale);
    const k = INPUT / LOAD;
    const y0 = Math.round(square.top * k), y1 = Math.round((square.top + square.h) * k);
    const x0 = Math.round(square.left * k), x1 = Math.round((square.left + square.w) * k);
    const crop = new Float32Array((x1 - x0) * (y1 - y0));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) crop[(y - y0) * (x1 - x0) + (x - x0)] = heat[y * INPUT + x];
    const cam = resize(crop, x1 - x0, y1 - y0, dw, dh);
    let camMax = 1e-8, peak = 0;
    cam.forEach((v, i) => { if (v > camMax) { camMax = v; peak = i; } });

    const positive = score >= info.threshold;
    // The map is normalised per image, so fade it on negative images to avoid implying a finding
    const strength = positive ? 1 : 0.25 + (0.5 * score) / info.threshold;
    const overlay = new ImageData(dw, dh);
    for (let i = 0; i < cam.length; i++) {
      const a = Math.min(Math.max((cam[i] / camMax - 0.35) / 0.65, 0), 1);
      overlay.data.set([255, 255, 255, Math.round(a ** 1.5 * 200 * strength)], 4 * i);
    }
    const overlayCanvas = makeCanvas(dw, dh);
    overlayCanvas.getContext("2d").putImageData(overlay, 0, 0);
    const displayCanvas = makeCanvas(dw, dh);
    const dctx = displayCanvas.getContext("2d");
    dctx.imageSmoothingQuality = "high";
    dctx.drawImage(bitmap, 0, 0, dw, dh);
    step("explain", t, "heat map of where it looked");

    t = performance.now();
    step("decide", t, `cut-off ${info.threshold.toFixed(2)}`);

    const level = levelOf(score, info);
    return {
      score, threshold: info.threshold, positive, level,
      finding: levelText[level].finding,
      confidence_note: levelText[level].note,
      peak: { x: (peak % dw) / dw, y: Math.floor(peak / dw) / dh },
      images: {
        original: displayCanvas.toDataURL("image/jpeg", 0.9),
        model_input: modelInput,
        attention: overlayCanvas.toDataURL("image/png"),
      },
      size: { width: w, height: h },
      steps,
    };
  }

  return { load, analyze };
})();
