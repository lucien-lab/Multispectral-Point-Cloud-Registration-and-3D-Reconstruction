import {
  RegistrationModel,
  WHEEL_IDLE_MS,
  clamp,
  cloneConfig,
  composeCamera,
  configsEqual,
  createDebouncedTask,
  dragConfig,
  median,
  medianPositiveDepth,
  parseCommittedNumber,
  projectCamera,
  projectPoints,
  resetGeometry,
} from "./registration_core.mjs";


const colors = {
  graphite: "#11161C",
  cyan: "#22D3EE",
  amber: "#FACC15",
  offwhite: "#E8EEF2",
};

const canvas = document.getElementById("registration-canvas");
const context = canvas.getContext("2d");
const canvasMessage = document.getElementById("canvas-message");
const saveStatus = document.getElementById("save-status");
const baseStatus = document.getElementById("base-status");
const configWarning = document.getElementById("config-warning");
const metricsSource = document.getElementById("metrics-source");
const saveButton = document.getElementById("save-button");
const undoButton = document.getElementById("undo-button");
const redoButton = document.getElementById("redo-button");
const resetButton = document.getElementById("reset-button");
const exportButton = document.getElementById("export-button");
const interactiveElements = [...document.querySelectorAll("input, button")];

const metricFields = {
  inFrame: document.getElementById("metric-in-frame"),
  visible: document.getElementById("metric-visible"),
  coverage: document.getElementById("metric-coverage"),
  median: document.getElementById("metric-control-median"),
  max: document.getElementById("metric-control-max"),
  focal: document.getElementById("metric-focal"),
};

const parameterSpecs = {
  tx: numberAndSlider("translation-x", "translation-x-slider", -2000, 2000, 0),
  ty: numberAndSlider("translation-y", "translation-y-slider", -2000, 2000, 0),
  tz: numberAndSlider("translation-z", "translation-z-slider", -2000, 2000, 0),
  rx: numberAndSlider("rotation-x", "rotation-x-slider", -30, 30, 0),
  ry: numberAndSlider("rotation-y", "rotation-y-slider", -30, 30, 0),
  rz: numberAndSlider("rotation-z", "rotation-z-slider", -30, 30, 0),
  zoom: sliderOnly("projection-zoom", 0.5, 2, 1),
  opacity: sliderOnly("point-opacity", 0, 1, 0.75),
  size: sliderOnly("point-size", 0.5, 8, 3),
  brightness: sliderOnly("photo-brightness", 0.25, 2, 1),
};

const state = {
  ready: false,
  session: null,
  photo: null,
  model: null,
  originalLeft: null,
  originalRight: null,
  drag: null,
  framePending: false,
  metricSequence: 0,
  saving: false,
  exporting: false,
  sliderStarts: new WeakMap(),
  wheelUiTimer: null,
};


function numberAndSlider(numberId, sliderId, minimum, maximum, defaultValue) {
  return {
    number: document.getElementById(numberId),
    slider: document.getElementById(sliderId),
    minimum,
    maximum,
    defaultValue,
  };
}


function sliderOnly(sliderId, minimum, maximum, defaultValue) {
  return {
    number: null,
    slider: document.getElementById(sliderId),
    minimum,
    maximum,
    defaultValue,
  };
}


function isReady() {
  return state.ready && state.model !== null && state.session !== null;
}


function getParameter(config, key) {
  const translation = config.translation_delta_mm;
  const rotation = config.rotation_delta_deg;
  return {
    tx: translation[0],
    ty: translation[1],
    tz: translation[2],
    rx: rotation[0],
    ry: rotation[1],
    rz: rotation[2],
    zoom: config.projection_zoom,
    opacity: config.point_opacity,
    size: config.point_size,
    brightness: config.photo_brightness,
  }[key];
}


function setParameter(config, key, value) {
  const next = cloneConfig(config);
  const mapping = {
    tx: () => { next.translation_delta_mm[0] = value; },
    ty: () => { next.translation_delta_mm[1] = value; },
    tz: () => { next.translation_delta_mm[2] = value; },
    rx: () => { next.rotation_delta_deg[0] = value; },
    ry: () => { next.rotation_delta_deg[1] = value; },
    rz: () => { next.rotation_delta_deg[2] = value; },
    zoom: () => { next.projection_zoom = value; },
    opacity: () => { next.point_opacity = value; },
    size: () => { next.point_size = value; },
    brightness: () => { next.photo_brightness = value; },
  };
  mapping[key]();
  return next;
}


function setReady(ready) {
  state.ready = ready;
  for (const element of interactiveElements) {
    element.disabled = !ready;
  }
  canvas.setAttribute("aria-disabled", String(!ready));
  if (ready) {
    updateHistoryButtons();
    updateSaveState();
  } else {
    document.body.dataset.saveState = "loading";
  }
}


function signed(value) {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1).padStart(6, "0")}`;
}


function syncFields() {
  if (!isReady()) {
    return;
  }
  for (const [key, spec] of Object.entries(parameterSpecs)) {
    const value = getParameter(state.model.current, key);
    spec.slider.value = value;
    if (spec.number && !(document.activeElement === spec.number && spec.number.dataset.editing === "true")) {
      spec.number.value = value;
      spec.number.removeAttribute("aria-invalid");
    }
  }
  const config = state.model.current;
  document.getElementById("show-left").checked = config.show_left;
  document.getElementById("show-right").checked = config.show_right;
  document.getElementById("projection-zoom-value").value = `${state.model.current.projection_zoom.toFixed(2)}×`;
  document.getElementById("point-opacity-value").value = `${Math.round(config.point_opacity * 100)}%`;
  document.getElementById("point-size-value").value = `${config.point_size.toFixed(1)} px`;
  document.getElementById("photo-brightness-value").value = `${config.photo_brightness.toFixed(2)}×`;
  document.getElementById("frame-x").value = signed(config.translation_delta_mm[0]);
  document.getElementById("frame-y").value = signed(config.translation_delta_mm[1]);
}


function updateHistoryButtons() {
  if (!isReady()) {
    undoButton.disabled = true;
    redoButton.disabled = true;
    return;
  }
  undoButton.disabled = state.model.history.length === 0;
  redoButton.disabled = state.model.redoStack.length === 0;
}


function updateSaveState(message = null) {
  if (!isReady()) {
    saveButton.disabled = true;
    return;
  }
  const dirty = state.model.dirty;
  document.body.dataset.saveState = dirty ? "dirty" : "clean";
  saveButton.disabled = state.saving || !dirty;
  saveStatus.textContent = message || (dirty ? "有未保存修改" : "配置已保存");
}


function afterPreview() {
  syncFields();
  updateSaveState();
  requestRender();
  scheduleMetrics();
}


function afterCommit() {
  syncFields();
  updateHistoryButtons();
  updateSaveState();
  requestRender();
  scheduleMetrics();
}


function requestRender() {
  if (state.framePending || !isReady()) {
    return;
  }
  state.framePending = true;
  requestAnimationFrame(() => {
    state.framePending = false;
    render();
  });
}


function resizeCanvas(width, height) {
  const ratio = clamp(window.devicePixelRatio || 1, 1, 2);
  const pixelWidth = Math.max(1, Math.round(width * ratio));
  const pixelHeight = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
}


function drawPointSet(points, color, size, opacity) {
  context.save();
  context.fillStyle = color;
  context.globalAlpha = opacity;
  context.beginPath();
  for (const point of points) {
    context.moveTo(point.x + size, point.y);
    context.arc(point.x, point.y, size, 0, Math.PI * 2);
  }
  context.fill();
  context.restore();
}


function drawInstrumentFrame(width, height) {
  const translation = state.model.current.translation_delta_mm;
  const rotation = state.model.current.rotation_delta_deg;
  const centerX = width * (
    0.5 + clamp(translation[0] / 2000, -1, 1) * 0.16 + (rotation[1] / 30) * 0.04
  );
  const centerY = height * (
    0.5 + clamp(translation[1] / 2000, -1, 1) * 0.16 - (rotation[0] / 30) * 0.04
  );
  const arm = Math.min(width, height) * (
    0.07 + clamp(translation[2] / 2000, -1, 1) * 0.018
  );

  context.save();
  context.strokeStyle = "rgba(232, 238, 242, 0.42)";
  context.lineWidth = 0.7;
  context.beginPath();
  for (let index = 0; index <= 40; index += 1) {
    const x = (index / 40) * width;
    const length = index % 5 === 0 ? 8 : 4;
    context.moveTo(x, 0);
    context.lineTo(x, length);
    context.moveTo(x, height);
    context.lineTo(x, height - length);
  }
  for (let index = 0; index <= 30; index += 1) {
    const y = (index / 30) * height;
    const length = index % 5 === 0 ? 8 : 4;
    context.moveTo(0, y);
    context.lineTo(length, y);
    context.moveTo(width, y);
    context.lineTo(width - length, y);
  }
  context.stroke();
  context.translate(centerX, centerY);
  context.rotate((rotation[2] * Math.PI) / 180);
  context.strokeStyle = "rgba(232, 238, 242, 0.78)";
  context.beginPath();
  context.moveTo(-arm, 0);
  context.lineTo(-7, 0);
  context.moveTo(7, 0);
  context.lineTo(arm, 0);
  context.moveTo(0, -arm);
  context.lineTo(0, -7);
  context.moveTo(0, 7);
  context.lineTo(0, arm);
  context.stroke();
  context.strokeStyle = colors.cyan;
  context.globalAlpha = 0.8;
  context.strokeRect(-3, -3, 6, 6);
  context.restore();
}


function localMetrics(cameraModel, leftProjected, rightProjected) {
  const total = state.originalLeft.length + state.originalRight.length;
  const inFrame = leftProjected.length + rightProjected.length;
  const controls = projectCamera(state.session.controls.points_xyz, cameraModel).pixels;
  const errors = controls.map((point, index) => Math.hypot(
    point[0] - state.session.controls.pixels_xy[index][0],
    point[1] - state.session.controls.pixels_xy[index][1]
  )).filter(Number.isFinite);
  setMetrics({
    in_frame_count: inFrame,
    visible_count: inFrame,
    coverage_fraction: total ? inFrame / total : 0,
    control_median_px: median(errors),
    control_max_px: errors.length ? Math.max(...errors) : 0,
    focal_px: cameraModel.focal,
  }, "本地");
}


function setMetrics(metrics, source) {
  metricFields.inFrame.textContent = String(metrics.in_frame_count);
  metricFields.visible.textContent = String(metrics.visible_count);
  metricFields.coverage.textContent = `${(metrics.coverage_fraction * 100).toFixed(1)}%`;
  metricFields.median.textContent = `${metrics.control_median_px.toFixed(2)} px`;
  metricFields.max.textContent = `${metrics.control_max_px.toFixed(2)} px`;
  metricFields.focal.textContent = `${metrics.focal_px.toFixed(1)} px`;
  metricsSource.textContent = source;
}


function render() {
  if (!isReady()) {
    return;
  }
  const { width, height } = state.session.photo;
  resizeCanvas(width, height);
  context.clearRect(0, 0, width, height);
  context.fillStyle = colors.graphite;
  context.fillRect(0, 0, width, height);
  context.save();
  context.filter = `brightness(${state.model.current.photo_brightness})`;
  context.drawImage(state.photo, 0, 0, width, height);
  context.restore();

  const cameraModel = composeCamera(state.session.base_camera, state.model.current);
  const leftProjected = projectPoints(state.originalLeft, cameraModel, width, height);
  const rightProjected = projectPoints(state.originalRight, cameraModel, width, height);
  const config = state.model.current;
  if (config.show_left) {
    drawPointSet(leftProjected, colors.cyan, config.point_size, config.point_opacity);
  }
  if (config.show_right) {
    drawPointSet(rightProjected, colors.amber, config.point_size, config.point_opacity);
  }
  drawInstrumentFrame(width, height);
  localMetrics(cameraModel, leftProjected, rightProjected);
}


async function sendMetrics(submitted) {
  if (!isReady()) {
    return;
  }
  const sequence = ++state.metricSequence;
  try {
    const response = await fetch("/api/metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(submitted),
    });
    if (!response.ok) {
      throw new Error(`指标请求失败（${response.status}）`);
    }
    const metrics = await response.json();
    if (
      sequence === state.metricSequence
      && isReady()
      && configsEqual(submitted, state.model.current)
    ) {
      setMetrics(metrics, "服务器");
    }
  } catch (_error) {
    if (sequence === state.metricSequence) {
      metricsSource.textContent = "本地";
    }
  }
}

const metricsDebounce = createDebouncedTask(sendMetrics, 200, window);


function scheduleMetrics() {
  if (isReady()) {
    metricsDebounce.schedule(cloneConfig(state.model.current));
  }
}


function bindNumberInputs() {
  for (const [key, spec] of Object.entries(parameterSpecs)) {
    if (!spec.number) {
      continue;
    }
    const input = spec.number;
    input.addEventListener("input", () => {
      if (!isReady()) {
        return;
      }
      input.dataset.editing = "true";
      input.removeAttribute("aria-invalid");
    });
    input.addEventListener("change", () => commitNumberInput(key));
    input.addEventListener("blur", () => commitNumberInput(key));
    input.addEventListener("keydown", (event) => {
      if (!isReady() || event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      commitNumberInput(key);
      input.blur();
    });
  }
}


function commitNumberInput(key) {
  if (!isReady()) {
    return;
  }
  const spec = parameterSpecs[key];
  const input = spec.number;
  if (input.dataset.editing !== "true") {
    return;
  }
  input.dataset.editing = "false";
  const parsed = parseCommittedNumber(input.value, spec.minimum, spec.maximum);
  if (!parsed.ok) {
    input.setAttribute("aria-invalid", "true");
    syncFields();
    updateSaveState("输入无效，已恢复上一个有效值");
    return;
  }
  input.removeAttribute("aria-invalid");
  state.model.apply(setParameter(state.model.current, key, parsed.value));
  afterCommit();
}


function bindSliders() {
  for (const [key, spec] of Object.entries(parameterSpecs)) {
    const slider = spec.slider;
    slider.addEventListener("input", () => {
      if (!isReady()) {
        return;
      }
      if (!state.sliderStarts.has(slider)) {
        state.sliderStarts.set(slider, cloneConfig(state.model.current));
      }
      state.model.preview(setParameter(
        state.model.current,
        key,
        clamp(Number(slider.value), spec.minimum, spec.maximum)
      ));
      afterPreview();
    });
    slider.addEventListener("change", () => {
      if (!isReady()) {
        return;
      }
      const previous = state.sliderStarts.get(slider);
      state.sliderStarts.delete(slider);
      if (previous) {
        state.model.commitFrom(previous);
      }
      afterCommit();
    });
  }
}


function bindDisplayToggles() {
  for (const [id, key] of [["show-left", "show_left"], ["show-right", "show_right"]]) {
    document.getElementById(id).addEventListener("change", (event) => {
      if (!isReady()) {
        return;
      }
      const next = cloneConfig(state.model.current);
      next[key] = event.target.checked;
      state.model.apply(next);
      afterCommit();
    });
  }
}


function bindSingleResetLabels() {
  for (const label of document.querySelectorAll("[data-reset-parameter]")) {
    label.addEventListener("dblclick", () => {
      if (!isReady()) {
        return;
      }
      const key = label.dataset.resetParameter;
      const spec = parameterSpecs[key];
      state.model.apply(setParameter(state.model.current, key, spec.defaultValue));
      afterCommit();
    });
  }
}


function pointerPosition(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - bounds.left) / bounds.width) * state.session.photo.width,
    y: ((event.clientY - bounds.top) / bounds.height) * state.session.photo.height,
  };
}


function endDrag(event) {
  if (!isReady() || !state.drag || event.pointerId !== state.drag.pointerId) {
    return;
  }
  const previous = state.drag.startConfig;
  state.drag = null;
  canvas.classList.remove("is-dragging");
  if (event.type !== "lostpointercapture" && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  state.model.commitFrom(previous);
  afterCommit();
}


function bindCanvas() {
  canvas.addEventListener("pointerdown", (event) => {
    if (!isReady() || event.button !== 0) {
      return;
    }
    state.model.finishWheel();
    const cameraModel = composeCamera(state.session.base_camera, state.model.current);
    const depth = medianPositiveDepth(
      state.originalLeft.concat(state.originalRight),
      cameraModel
    );
    if (!(depth > 0)) {
      showError("没有正相机深度点，当前无法拖动。", true);
      return;
    }
    clearError();
    state.drag = {
      pointerId: event.pointerId,
      start: pointerPosition(event),
      startConfig: cloneConfig(state.model.current),
      depth,
      focal: cameraModel.focal,
    };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("is-dragging");
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!isReady() || !state.drag || event.pointerId !== state.drag.pointerId) {
      return;
    }
    const current = pointerPosition(event);
    state.model.preview(dragConfig(
      state.drag.startConfig,
      current.x - state.drag.start.x,
      current.y - state.drag.start.y,
      state.drag.depth,
      state.drag.focal
    ));
    afterPreview();
  });
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", endDrag);

  canvas.addEventListener("wheel", (event) => {
    if (!isReady()) {
      return;
    }
    event.preventDefault();
    state.model.wheel(event.deltaY);
    afterPreview();
    window.clearTimeout(state.wheelUiTimer);
    state.wheelUiTimer = window.setTimeout(afterCommit, WHEEL_IDLE_MS + 1);
  }, { passive: false });
}


async function saveConfiguration() {
  if (!isReady() || state.saving) {
    return;
  }
  const submitted = state.model.beginSave();
  state.saving = true;
  updateSaveState("正在保存配置…");
  try {
    const response = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(submitted),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `保存失败（${response.status}）`);
    }
    state.model.completeSave(submitted, result.config);
    state.saving = false;
    configWarning.hidden = true;
    clearError();
    updateSaveState(state.model.dirty ? "已保存提交版本；仍有较新的修改" : "配置已保存");
  } catch (error) {
    state.saving = false;
    document.body.dataset.saveState = "error";
    saveButton.disabled = false;
    saveStatus.textContent = error.message;
  }
}


async function exportConfiguration() {
  if (!isReady() || state.exporting) {
    return;
  }
  const submitted = state.model.beginSave();
  state.exporting = true;
  exportButton.disabled = true;
  exportButton.textContent = "正在导出…";
  updateSaveState("正在生成彩色点云和投影图…");
  try {
    const response = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(submitted),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `导出失败（${response.status}）`);
    }
    state.model.completeSave(submitted, submitted);
    state.exporting = false;
    exportButton.disabled = false;
    exportButton.textContent = "导出点云";
    clearError();
    updateSaveState(state.model.dirty ? "已导出提交版本；仍有较新的修改" : "点云与投影图已导出");
  } catch (error) {
    state.exporting = false;
    exportButton.disabled = false;
    exportButton.textContent = "导出点云";
    document.body.dataset.saveState = "error";
    saveStatus.textContent = error.message;
  }
}


function bindActions() {
  undoButton.addEventListener("click", () => {
    if (!isReady() || !state.model.undo()) {
      return;
    }
    afterCommit();
  });
  redoButton.addEventListener("click", () => {
    if (!isReady() || !state.model.redo()) {
      return;
    }
    afterCommit();
  });
  resetButton.addEventListener("click", () => {
    if (!isReady()) {
      return;
    }
    state.model.apply(resetGeometry(state.model.current));
    afterCommit();
  });
  saveButton.addEventListener("click", saveConfiguration);
  exportButton.addEventListener("click", exportConfiguration);
}


function showError(message, canvasOnly = false) {
  canvasMessage.textContent = message;
  canvasMessage.hidden = false;
  if (!canvasOnly) {
    document.body.dataset.saveState = "error";
  }
}


function clearError() {
  canvasMessage.textContent = "";
  canvasMessage.hidden = true;
  if (isReady()) {
    document.body.dataset.saveState = state.model.dirty ? "dirty" : "clean";
  }
}


function showSessionStatus(session) {
  if (session.base_status.accepted) {
    baseStatus.hidden = true;
  } else {
    baseStatus.textContent = `基础自动配准未通过：${session.base_status.reason}`;
    baseStatus.hidden = false;
  }
  if (session.config_warning) {
    configWarning.textContent = `已保存配置无效，当前使用默认值且尚未保存：${session.config_warning}`;
    configWarning.hidden = false;
  } else {
    configWarning.hidden = true;
  }
}


async function loadSession() {
  setReady(false);
  try {
    const [sessionResponse, photo] = await Promise.all([
      fetch("/api/session"),
      new Promise((resolve, reject) => {
        const image = new Image();
        image.addEventListener("load", () => resolve(image), { once: true });
        image.addEventListener("error", () => reject(new Error("照片加载失败。")), { once: true });
        image.src = "/api/photo";
      }),
    ]);
    if (!sessionResponse.ok) {
      throw new Error(`会话加载失败（${sessionResponse.status}）。`);
    }
    const session = await sessionResponse.json();
    state.session = session;
    state.photo = photo;
    state.model = new RegistrationModel(session.config, {
      saved: !session.config_warning,
    });
    if (session.config_warning) {
      state.model.markUnsaved();
    }
    state.originalLeft = Object.freeze(
      session.left_points.map((point) => Object.freeze(point.slice()))
    );
    state.originalRight = Object.freeze(
      session.right_points.map((point) => Object.freeze(point.slice()))
    );
    showSessionStatus(session);
    setReady(true);
    clearError();
    syncFields();
    updateHistoryButtons();
    updateSaveState(session.config_warning ? "默认配置尚未保存" : "配置已保存");
    requestRender();
    scheduleMetrics();
  } catch (error) {
    setReady(false);
    saveStatus.textContent = error.message;
    showError(`${error.message} 请确认本机配准服务正在运行。`);
  }
}


bindNumberInputs();
bindSliders();
bindDisplayToggles();
bindSingleResetLabels();
bindCanvas();
bindActions();
window.addEventListener("resize", () => {
  if (isReady()) {
    requestRender();
  }
});
loadSession();
