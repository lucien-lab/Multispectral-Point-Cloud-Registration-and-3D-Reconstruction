export const HISTORY_LIMIT = 100;
export const WHEEL_IDLE_MS = 150;


export function cloneConfig(value) {
  return JSON.parse(JSON.stringify(value));
}


export function configsEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}


export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}


export function matrixMultiply(left, right) {
  const result = new Array(9).fill(0);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      for (let index = 0; index < 3; index += 1) {
        result[row * 3 + column] += left[row * 3 + index] * right[index * 3 + column];
      }
    }
  }
  return result;
}


export function matrixVector(matrix, vector) {
  return [
    matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
    matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
    matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
  ];
}


export function rodriguesMatrix(rotationVector) {
  const [x, y, z] = rotationVector;
  const angle = Math.hypot(x, y, z);
  if (angle < 1e-12) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }
  const kx = x / angle;
  const ky = y / angle;
  const kz = z / angle;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const complement = 1 - cosine;
  return [
    cosine + kx * kx * complement,
    kx * ky * complement - kz * sine,
    kx * kz * complement + ky * sine,
    ky * kx * complement + kz * sine,
    cosine + ky * ky * complement,
    ky * kz * complement - kx * sine,
    kz * kx * complement - ky * sine,
    kz * ky * complement + kx * sine,
    cosine + kz * kz * complement,
  ];
}


export function eulerXYZMatrix(degrees) {
  const [x, y, z] = degrees.map((value) => (value * Math.PI) / 180);
  const rx = [1, 0, 0, 0, Math.cos(x), -Math.sin(x), 0, Math.sin(x), Math.cos(x)];
  const ry = [Math.cos(y), 0, Math.sin(y), 0, 1, 0, -Math.sin(y), 0, Math.cos(y)];
  const rz = [Math.cos(z), -Math.sin(z), 0, Math.sin(z), Math.cos(z), 0, 0, 0, 1];
  return matrixMultiply(rz, matrixMultiply(ry, rx));
}


export function deriveManualTransform(config) {
  const rotation = eulerXYZMatrix(config.rotation_delta_deg);
  const translation = config.translation_delta_mm;
  return [
    [rotation[0], rotation[1], rotation[2], translation[0]],
    [rotation[3], rotation[4], rotation[5], translation[1]],
    [rotation[6], rotation[7], rotation[8], translation[2]],
    [0, 0, 0, 1],
  ];
}


export function synchronizeManualTransform(config) {
  const synchronized = cloneConfig(config);
  synchronized.manual_transform_4x4 = deriveManualTransform(synchronized);
  return synchronized;
}


export function composeCamera(baseCamera, config) {
  const deltaRotation = eulerXYZMatrix(config.rotation_delta_deg);
  const baseRotation = rodriguesMatrix(baseCamera.rotation_vector);
  const rotatedTranslation = matrixVector(deltaRotation, baseCamera.translation_mm);
  return {
    rotation: matrixMultiply(deltaRotation, baseRotation),
    translation: rotatedTranslation.map(
      (value, index) => value + config.translation_delta_mm[index]
    ),
    focal: baseCamera.focal_px * config.projection_zoom,
    principal: baseCamera.principal_point_px.slice(),
    radial: baseCamera.radial_k1,
  };
}


export function cameraPoint(point, cameraModel) {
  const rotated = matrixVector(cameraModel.rotation, point);
  return rotated.map((value, index) => value + cameraModel.translation[index]);
}


export function pythonSafeDepth(depth) {
  if (Math.abs(depth) >= 1e-12) {
    return depth;
  }
  return depth + 1e-12 < 0 ? -1e-12 : 1e-12;
}


export function projectCamera(points, cameraModel) {
  const pixels = [];
  const depth = [];
  for (const point of points) {
    const cameraXYZ = cameraPoint(point, cameraModel);
    const pointDepth = cameraXYZ[2];
    const safeDepth = pythonSafeDepth(pointDepth);
    let normalizedX = cameraXYZ[0] / safeDepth;
    let normalizedY = cameraXYZ[1] / safeDepth;
    const radiusSquared = normalizedX * normalizedX + normalizedY * normalizedY;
    const radialScale = 1 + cameraModel.radial * radiusSquared;
    normalizedX *= radialScale;
    normalizedY *= radialScale;
    pixels.push([
      normalizedX * cameraModel.focal + cameraModel.principal[0],
      normalizedY * cameraModel.focal + cameraModel.principal[1],
    ]);
    depth.push(pointDepth);
  }
  return { pixels, depth };
}


export function projectPoints(points, cameraModel, width, height) {
  const result = projectCamera(points, cameraModel);
  const projected = [];
  result.pixels.forEach((pixel, index) => {
    const [x, y] = pixel;
    const depth = result.depth[index];
    if (
      depth > 0
      && Number.isFinite(x)
      && Number.isFinite(y)
      && x >= 0
      && x < width
      && y >= 0
      && y < height
    ) {
      projected.push({ x, y, depth });
    }
  });
  return projected;
}


export function median(values) {
  if (!values.length) {
    return 0;
  }
  const ordered = values.slice().sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}


export function medianPositiveDepth(points, cameraModel) {
  return median(
    points
      .map((point) => cameraPoint(point, cameraModel)[2])
      .filter((depth) => depth > 0 && Number.isFinite(depth))
  );
}


export function dragConfig(startConfig, deltaX, deltaY, depth, focal) {
  const next = cloneConfig(startConfig);
  next.translation_delta_mm[0] = clamp(
    startConfig.translation_delta_mm[0] + (deltaX * depth) / focal,
    -2000,
    2000
  );
  next.translation_delta_mm[1] = clamp(
    startConfig.translation_delta_mm[1] + (deltaY * depth) / focal,
    -2000,
    2000
  );
  return synchronizeManualTransform(next);
}


export function resetGeometry(config) {
  const reset = cloneConfig(config);
  reset.translation_delta_mm = [0, 0, 0];
  reset.rotation_delta_deg = [0, 0, 0];
  reset.projection_zoom = 1;
  return synchronizeManualTransform(reset);
}


export function parseCommittedNumber(text, minimum, maximum) {
  const normalized = String(text).trim();
  if (!normalized) {
    return { ok: false };
  }
  const value = Number(normalized);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    return { ok: false };
  }
  return { ok: true, value };
}


export function createDebouncedTask(callback, delay, timers = globalThis) {
  let timer = null;
  let latest;
  return {
    schedule(payload) {
      latest = payload;
      if (timer !== null) {
        timers.clearTimeout(timer);
      }
      timer = timers.setTimeout(() => {
        timer = null;
        callback(latest);
      }, delay);
    },
    cancel() {
      if (timer !== null) {
        timers.clearTimeout(timer);
        timer = null;
      }
    },
  };
}


export class RegistrationModel {
  constructor(initialConfig, options = {}) {
    this.current = synchronizeManualTransform(initialConfig);
    this.saved = options.saved === false ? null : cloneConfig(this.current);
    this.history = [];
    this.redoStack = [];
    this.historyLimit = options.historyLimit ?? HISTORY_LIMIT;
    this.timers = options.timers ?? globalThis;
    this.wheelIdleMs = options.wheelIdleMs ?? WHEEL_IDLE_MS;
    this.wheelStart = null;
    this.wheelTimer = null;
  }

  get dirty() {
    return this.saved === null || !configsEqual(this.current, this.saved);
  }

  preview(next) {
    this.current = synchronizeManualTransform(next);
    return this.current;
  }

  commitFrom(previous) {
    if (configsEqual(previous, this.current)) {
      return false;
    }
    this.history.push(cloneConfig(previous));
    if (this.history.length > this.historyLimit) {
      this.history.shift();
    }
    this.redoStack = [];
    return true;
  }

  apply(next) {
    this.finishWheel();
    const previous = cloneConfig(this.current);
    this.preview(next);
    this.commitFrom(previous);
    return this.current;
  }

  wheel(deltaY) {
    if (deltaY === 0) {
      return this.current;
    }
    if (this.wheelStart === null) {
      this.wheelStart = cloneConfig(this.current);
    }
    const direction = deltaY > 0 ? -1 : 1;
    const next = cloneConfig(this.current);
    next.projection_zoom = clamp(
      Number((next.projection_zoom + direction * 0.05).toFixed(2)),
      0.5,
      2
    );
    this.preview(next);
    if (this.wheelTimer !== null) {
      this.timers.clearTimeout(this.wheelTimer);
    }
    this.wheelTimer = this.timers.setTimeout(
      () => this.finishWheel(),
      this.wheelIdleMs
    );
    return this.current;
  }

  finishWheel() {
    if (this.wheelTimer !== null) {
      this.timers.clearTimeout(this.wheelTimer);
      this.wheelTimer = null;
    }
    if (this.wheelStart === null) {
      return false;
    }
    const previous = this.wheelStart;
    this.wheelStart = null;
    return this.commitFrom(previous);
  }

  undo() {
    this.finishWheel();
    if (!this.history.length) {
      return false;
    }
    this.redoStack.push(cloneConfig(this.current));
    this.current = this.history.pop();
    return true;
  }

  redo() {
    this.finishWheel();
    if (!this.redoStack.length) {
      return false;
    }
    this.history.push(cloneConfig(this.current));
    if (this.history.length > this.historyLimit) {
      this.history.shift();
    }
    this.current = this.redoStack.pop();
    return true;
  }

  beginSave() {
    return cloneConfig(this.current);
  }

  completeSave(submitted, savedConfig) {
    this.saved = synchronizeManualTransform(savedConfig ?? submitted);
  }

  markUnsaved() {
    this.saved = null;
  }
}
