import test from "node:test";
import assert from "node:assert/strict";

import {
  RegistrationModel,
  composeCamera,
  createDebouncedTask,
  deriveManualTransform,
  dragConfig,
  parseCommittedNumber,
  projectCamera,
  resetGeometry,
} from "../manual_registration_web/registration_core.mjs";


function config() {
  return {
    schema_version: 1,
    translation_delta_mm: [0, 0, 0],
    rotation_delta_deg: [0, 0, 0],
    manual_transform_4x4: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    projection_zoom: 1,
    show_left: true,
    show_right: false,
    point_opacity: 0.61,
    point_size: 4.2,
    photo_brightness: 1.15,
    base_camera_model: "camera_model.json",
    base_right_transform: "right_to_left_transform.json",
  };
}


function withTranslation(source, x) {
  const next = structuredClone(source);
  next.translation_delta_mm[0] = x;
  next.manual_transform_4x4 = deriveManualTransform(next);
  return next;
}


class FakeTimers {
  constructor() {
    this.now = 0;
    this.nextId = 1;
    this.jobs = new Map();
  }

  setTimeout = (callback, delay) => {
    const id = this.nextId++;
    this.jobs.set(id, { callback, due: this.now + delay });
    return id;
  };

  clearTimeout = (id) => {
    this.jobs.delete(id);
  };

  advance(milliseconds) {
    const target = this.now + milliseconds;
    while (true) {
      const due = [...this.jobs.entries()]
        .filter(([, job]) => job.due <= target)
        .sort((left, right) => left[1].due - right[1].due)[0];
      if (!due) {
        break;
      }
      const [id, job] = due;
      this.jobs.delete(id);
      this.now = job.due;
      job.callback();
    }
    this.now = target;
  }
}


test("browser projection matches the fixed Python camera sample", () => {
  const base = {
    rotation_vector: [0.12, -0.08, 0.03],
    translation_mm: [25, -40, 1100],
    focal_px: 875.5,
    principal_point_px: [640, 480],
    radial_k1: -0.015,
  };
  const manual = config();
  manual.translation_delta_mm = [12, -7, 18];
  manual.rotation_delta_deg = [4.5, -2.25, 1.75];
  manual.manual_transform_4x4 = deriveManualTransform(manual);
  manual.projection_zoom = 1.13;
  const points = [[0, 0, 0], [120, -35, 80], [-240, 60, 300]];
  const expectedPixels = [
    [638.1430412132792, 361.0607070047659],
    [731.3319754562457, 332.8622750079291],
    [442.15814405534246, 375.0291256748267],
  ];
  const expectedDepth = [1111.6091723872457, 1197.3427655832484, 1385.9076750092863];

  const result = projectCamera(points, composeCamera(base, manual));

  result.pixels.forEach((pixel, row) => pixel.forEach((value, column) => {
    assert.ok(Math.abs(value - expectedPixels[row][column]) <= 1e-6);
  }));
  result.depth.forEach((value, row) => {
    assert.ok(Math.abs(value - expectedDepth[row]) <= 1e-9);
  });
});


test("projection uses Python safe-depth sign for a tiny negative depth", () => {
  const camera = {
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    translation: [0, 0, -1e-13],
    focal: 1,
    principal: [0, 0],
    radial: 0,
  };

  const result = projectCamera([[1e-15, 0, 0]], camera);

  assert.equal(result.depth[0], -1e-13);
  assert.ok(Math.abs(result.pixels[0][0] - 0.001) <= 1e-15);
});


test("drag converts canvas pixels to camera-frame X and Y translation", () => {
  const moved = dragConfig(config(), 20, -10, 1000, 500);

  assert.deepEqual(moved.translation_delta_mm, [40, -20, 0]);
  assert.deepEqual(moved.manual_transform_4x4.map((row) => row[3]), [40, -20, 0, 1]);
});


test("wheel burst commits once after idle and undo restores the whole gesture", () => {
  const timers = new FakeTimers();
  const model = new RegistrationModel(config(), { timers, wheelIdleMs: 150 });

  model.wheel(-1);
  timers.advance(100);
  model.wheel(-1);
  assert.equal(model.history.length, 0);
  assert.equal(model.current.projection_zoom, 1.1);
  timers.advance(149);
  assert.equal(model.history.length, 0);
  timers.advance(1);
  assert.equal(model.history.length, 1);

  model.undo();
  assert.equal(model.current.projection_zoom, 1);
  model.redo();
  assert.equal(model.current.projection_zoom, 1.1);
});


test("history keeps the latest 100 states and a new action clears redo", () => {
  const model = new RegistrationModel(config());
  for (let index = 1; index <= 105; index += 1) {
    model.apply(withTranslation(model.current, index));
  }
  assert.equal(model.history.length, 100);
  for (let index = 0; index < 100; index += 1) {
    model.undo();
  }
  assert.equal(model.current.translation_delta_mm[0], 5);

  model.redo();
  assert.equal(model.redoStack.length, 99);
  model.apply(withTranslation(model.current, 77));
  assert.equal(model.redoStack.length, 0);
});


test("geometry reset preserves every display preference", () => {
  const changed = config();
  changed.translation_delta_mm = [20, -30, 40];
  changed.rotation_delta_deg = [2, -3, 4];
  changed.manual_transform_4x4 = deriveManualTransform(changed);
  changed.projection_zoom = 1.7;

  const reset = resetGeometry(changed);

  assert.deepEqual(reset.translation_delta_mm, [0, 0, 0]);
  assert.deepEqual(reset.rotation_delta_deg, [0, 0, 0]);
  assert.deepEqual(reset.manual_transform_4x4, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
  assert.equal(reset.projection_zoom, 1);
  for (const key of ["point_size", "point_opacity", "photo_brightness", "show_left", "show_right"]) {
    assert.equal(reset[key], changed[key]);
  }
});


test("metrics debounce emits only the latest payload after exactly 200 ms", () => {
  const timers = new FakeTimers();
  const seen = [];
  const debounce = createDebouncedTask((payload) => seen.push(payload), 200, timers);

  debounce.schedule("first");
  timers.advance(100);
  debounce.schedule("second");
  timers.advance(199);
  assert.deepEqual(seen, []);
  timers.advance(1);
  assert.deepEqual(seen, ["second"]);
});


test("number commit rejects intermediate text and accepts a bounded decimal", () => {
  for (const text of ["", "-", ".", "-.", "1e", "NaN"] ) {
    assert.equal(parseCommittedNumber(text, -30, 30).ok, false);
  }
  assert.deepEqual(parseCommittedNumber("-1.25", -30, 30), { ok: true, value: -1.25 });
  assert.equal(parseCommittedNumber("31", -30, 30).ok, false);
});


test("save completion cleans only its submitted snapshot", () => {
  const model = new RegistrationModel(config());
  model.apply(withTranslation(model.current, 10));
  const submitted = model.beginSave();
  model.apply(withTranslation(model.current, 20));

  model.completeSave(submitted, submitted);

  assert.equal(model.current.translation_delta_mm[0], 20);
  assert.equal(model.saved.translation_delta_mm[0], 10);
  assert.equal(model.dirty, true);
  const latest = model.beginSave();
  model.completeSave(latest, latest);
  assert.equal(model.dirty, false);

  model.markUnsaved();
  assert.equal(model.dirty, true);
});
