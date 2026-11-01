/*
 * Classical computer vision, for the one detector that is not a model.
 *
 * The seatbelt detector in models/detectors/seatbelt_detector.py is
 * built on OpenCV: CLAHE, a Gaussian blur, Canny, and HoughLinesP. None
 * of that exists in a browser, so it is here, written out, because the
 * alternative was to drop the only detector in the project that works
 * without a neural network.
 *
 * Two deliberate differences from the Python, both documented where
 * they occur:
 *
 *   - CLAHE is replaced by a GLOBAL histogram equalisation. The
 *     adaptive part -- equalising each tile separately, so a region
 *     holding both a bright window and a dark footwell gets contrast
 *     in both -- is a lot of work for a chest, which is one roughly
 *     evenly lit surface. The equalisation itself is not optional
 *     though, and the first attempt here got that wrong: a percentile
 *     contrast stretch left the region's median at 255, the Canny
 *     thresholds are taken from that median, the high one clamped to
 *     255, and hysteresis then threw away all but 39 of the edge
 *     pixels in a picture with a strap painted straight across it.
 *     Equalisation puts the median at the middle of the range by
 *     construction, which is the property those thresholds assume.
 *
 *   - HoughLinesP is replaced by a full Hough transform with a segment
 *     walk. The probabilistic variant samples edge pixels at random,
 *     which makes it fast and makes its output differ run to run; the
 *     full transform over a band of angles this narrow costs about the
 *     same and gives the same answer every time, which a test can
 *     assert on.
 *
 * Everything else -- the angle band, the strap width, the parallel
 * tolerance, the overlap and the midline test -- is the same arithmetic
 * on the same constants.
 */

const BLUR_KERNEL = [1, 4, 6, 4, 1];   // 5-tap binomial, separable
const BLUR_WEIGHT = 16;

/** Luma from RGBA, ITU-R BT.601, the same weights cvtColor uses. */
export function grayscale(rgba, width, height) {
  const out = new Float32Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return out;
}

/**
 * Stand-in for CLAHE: global histogram equalisation.
 *
 * Each level is mapped to where it sits in the cumulative histogram,
 * which spreads whatever contrast the region has across the full
 * range. CLAHE is this done per tile and with the histogram clipped;
 * the adaptive part is what buys contrast in a region holding both a
 * bright window and a dark footwell, and a chest is not that.
 *
 * It does NOT promise a mid-range median, and an earlier comment here
 * claimed it did: on a two-level picture the majority level is also
 * the top level present, so it maps to 255 and takes the median with
 * it. Hence the clamp where the Canny thresholds are derived.
 *
 * A flat region (every pixel the same) has nothing to spread, so it is
 * returned untouched rather than divided by zero.
 */
export function equalize(gray) {
  const histogram = new Int32Array(256);
  for (let i = 0; i < gray.length; i++) {
    histogram[Math.max(0, Math.min(255, gray[i] | 0))]++;
  }

  // The first non-empty level maps to 0, so the darkest thing present
  // stays the darkest thing present.
  let first = 0;
  while (first < 255 && histogram[first] === 0) first++;

  const floor = histogram[first];
  const span = gray.length - floor;

  if (span <= 0) return gray;

  const map = new Float32Array(256);
  let seen = 0;

  for (let v = 0; v < 256; v++) {
    seen += histogram[v];
    map[v] = Math.max(0, Math.min(255, ((seen - floor) / span) * 255));
  }

  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = map[Math.max(0, Math.min(255, gray[i] | 0))];
  }
  return out;
}

/** Separable 5x5 Gaussian, edges clamped. */
export function blur(gray, width, height) {
  const pass = new Float32Array(gray.length);
  const out = new Float32Array(gray.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const sx = Math.max(0, Math.min(width - 1, x + k));
        sum += gray[y * width + sx] * BLUR_KERNEL[k + 2];
      }
      pass[y * width + x] = sum / BLUR_WEIGHT;
    }
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const sy = Math.max(0, Math.min(height - 1, y + k));
        sum += pass[sy * width + x] * BLUR_KERNEL[k + 2];
      }
      out[y * width + x] = sum / BLUR_WEIGHT;
    }
  }

  return out;
}

/**
 * The median level, clamped to a band where the 0.66x / 1.33x Canny
 * thresholds derived from it still mean something.
 *
 * Unclamped, a median of 255 gives a high threshold of 255 after the
 * clamp to the byte range -- and hysteresis then needs a gradient
 * magnitude of 255 to seed an edge at all, which is how the first
 * version of this threw a painted strap away. A median of 0 gives the
 * opposite failure, where everything is an edge.
 */
export function medianForThresholds(gray) {
  return Math.min(192, Math.max(32, median(gray)));
}

export function median(gray) {
  const histogram = new Int32Array(256);
  for (let i = 0; i < gray.length; i++) {
    histogram[Math.max(0, Math.min(255, gray[i] | 0))]++;
  }
  const half = gray.length / 2;
  let seen = 0;
  for (let v = 0; v < 256; v++) {
    seen += histogram[v];
    if (seen >= half) return v;
  }
  return 128;
}

/**
 * Canny: Sobel gradients, non-maximum suppression along the gradient,
 * then a double threshold with hysteresis.
 *
 * Returns a Uint8Array of 0 and 1, one per pixel.
 */
export function canny(gray, width, height, lowThreshold, highThreshold) {
  const magnitude = new Float32Array(gray.length);
  const direction = new Uint8Array(gray.length);  // 0:- 1:/ 2:| 3:\

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;

      const gx =
        -gray[i - width - 1] - 2 * gray[i - 1] - gray[i + width - 1] +
         gray[i - width + 1] + 2 * gray[i + 1] + gray[i + width + 1];

      const gy =
        -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1] +
         gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];

      magnitude[i] = Math.hypot(gx, gy);

      // Quantise the gradient to the four directions a pixel grid can
      // actually express, which is what the suppression step needs.
      let angle = Math.atan2(gy, gx) * 180 / Math.PI;
      if (angle < 0) angle += 180;

      direction[i] = angle < 22.5 ? 0
                   : angle < 67.5 ? 1
                   : angle < 112.5 ? 2
                   : angle < 157.5 ? 3 : 0;
    }
  }

  // Thin the ridges: keep a pixel only if it is the local maximum along
  // its own gradient. Without this, one strap edge becomes a band five
  // pixels wide and the Hough peak smears across five rho bins.
  const thin = new Float32Array(gray.length);

  // Indexed by the quantised gradient direction above. A gradient at
  // 45 degrees points along (+1, +1) in array coordinates, where y
  // grows downward, and one at 135 along (+1, -1).
  //
  // These two were the wrong way round at first, which suppressed the
  // ridge of every DIAGONAL edge -- so the seatbelt detector, whose
  // whole signature is two diagonal lines, found 48 edge pixels in a
  // picture with a strap painted straight across it. Horizontal and
  // vertical edges were unaffected, which is why nothing else noticed.
  const NEIGHBOURS = [[1, 0], [1, 1], [0, 1], [1, -1]];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const [dx, dy] = NEIGHBOURS[direction[i]];
      const a = magnitude[i + dy * width + dx];
      const b = magnitude[i - dy * width - dx];
      if (magnitude[i] >= a && magnitude[i] >= b) thin[i] = magnitude[i];
    }
  }

  // Hysteresis: strong pixels are edges, weak ones only if they touch
  // an edge, walked transitively so a faint stretch in the middle of a
  // strong line survives.
  const out = new Uint8Array(gray.length);
  const stack = [];

  for (let i = 0; i < thin.length; i++) {
    if (thin[i] >= highThreshold) { out[i] = 1; stack.push(i); }
  }

  while (stack.length) {
    const i = stack.pop();
    const x = i % width, y = (i - x) / width;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const n = ny * width + nx;
        if (!out[n] && thin[n] >= lowThreshold) { out[n] = 1; stack.push(n); }
      }
    }
  }

  return out;
}

/**
 * Line segments in an edge map, restricted to a band of angles.
 *
 * A full Hough transform over (rho, theta), peaks picked out of the
 * accumulator, then each peak walked along its own line to find the
 * longest run of edge pixels. Only the angles the caller asks for are
 * accumulated at all, which is what keeps a full transform affordable:
 * a strap is 30 to 70 degrees off horizontal either way, so 82 of the
 * 180 columns are ever touched.
 */
export function segments(edges, width, height, options) {
  const {
    minAngle, maxAngle, minLength, maxGap,
    maxEdges = 40, minVotes = 30
  } = options;

  const diagonal = Math.ceil(Math.hypot(width, height));
  const rhoCount = 2 * diagonal + 1;

  // Which theta columns are worth accumulating. A line whose normal is
  // at theta runs at theta - 90 from horizontal.
  const thetas = [];
  for (let degrees = 0; degrees < 180; degrees++) {
    let lineAngle = degrees - 90;
    if (lineAngle <= -90) lineAngle += 180;
    const magnitude = Math.abs(lineAngle);
    if (magnitude >= minAngle && magnitude <= maxAngle) thetas.push(degrees);
  }

  const cos = new Float32Array(thetas.length);
  const sin = new Float32Array(thetas.length);
  thetas.forEach((degrees, k) => {
    cos[k] = Math.cos(degrees * Math.PI / 180);
    sin[k] = Math.sin(degrees * Math.PI / 180);
  });

  const accumulator = new Int32Array(thetas.length * rhoCount);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!edges[y * width + x]) continue;
      for (let k = 0; k < thetas.length; k++) {
        const rho = Math.round(x * cos[k] + y * sin[k]) + diagonal;
        accumulator[k * rhoCount + rho]++;
      }
    }
  }

  // Peaks, non-maximum suppressed over a small window so one strong
  // line does not produce six copies of itself one bin apart.
  const peaks = [];
  const RHO_SPREAD = 4, THETA_SPREAD = 2;

  for (let k = 0; k < thetas.length; k++) {
    for (let rho = 0; rho < rhoCount; rho++) {
      const votes = accumulator[k * rhoCount + rho];
      if (votes < minVotes) continue;

      let best = true;
      for (let dk = -THETA_SPREAD; dk <= THETA_SPREAD && best; dk++) {
        for (let dr = -RHO_SPREAD; dr <= RHO_SPREAD; dr++) {
          const nk = k + dk, nr = rho + dr;
          if (nk < 0 || nr < 0 || nk >= thetas.length || nr >= rhoCount) continue;
          if (dk === 0 && dr === 0) continue;
          // > on one side and >= on the other, so a plateau of equal
          // bins keeps exactly one of its members.
          const other = accumulator[nk * rhoCount + nr];
          if (other > votes || (other === votes && (dk < 0 || (dk === 0 && dr < 0)))) {
            best = false;
            break;
          }
        }
      }

      if (best) peaks.push({ k, rho: rho - diagonal, votes });
    }
  }

  peaks.sort((a, b) => b.votes - a.votes);

  const found = [];

  for (const peak of peaks) {
    if (found.length >= maxEdges) break;

    const c = cos[peak.k], s = sin[peak.k];

    // A point on the line, and the direction along it.
    const originX = peak.rho * c, originY = peak.rho * s;
    const ux = -s, uy = c;

    let runStart = null, gap = 0, lastHit = null;
    let bestStart = null, bestEnd = null, bestLength = 0;

    const close = () => {
      if (runStart === null || lastHit === null) return;
      const length = lastHit - runStart;
      if (length > bestLength) {
        bestLength = length;
        bestStart = runStart;
        bestEnd = lastHit;
      }
      runStart = null;
      lastHit = null;
    };

    for (let t = -diagonal; t <= diagonal; t++) {
      const x = Math.round(originX + t * ux);
      const y = Math.round(originY + t * uy);

      if (x < 0 || y < 0 || x >= width || y >= height) {
        if (runStart !== null) close();
        continue;
      }

      // One pixel either side, because a rounded walk down a line at
      // 37 degrees steps past pixels the rasteriser put next door.
      let hit = false;
      for (let dx = -1; dx <= 1 && !hit; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          if (edges[ny * width + nx]) { hit = true; break; }
        }
      }

      if (hit) {
        if (runStart === null) runStart = t;
        lastHit = t;
        gap = 0;
      } else if (runStart !== null) {
        gap++;
        if (gap > maxGap) close();
      }
    }

    close();

    if (bestLength < minLength) continue;

    const x1 = originX + bestStart * ux, y1 = originY + bestStart * uy;
    const x2 = originX + bestEnd * ux, y2 = originY + bestEnd * uy;

    let angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
    if (angle <= -90) angle += 180;
    else if (angle > 90) angle -= 180;

    found.push({
      x1, y1, x2, y2, angle,
      length: Math.hypot(x2 - x1, y2 - y1),
      votes: peak.votes
    });
  }

  found.sort((a, b) => b.length - a.length);

  return found.slice(0, maxEdges);
}

// ----------------------------------------------------------------------
// pairing edges into a strap
// ----------------------------------------------------------------------

function direction(edge) {
  return [(edge.x2 - edge.x1) / edge.length, (edge.y2 - edge.y1) / edge.length];
}

/** Distance from `edge`'s midpoint to the infinite line through `line`. */
function distanceToLine(edge, line) {
  const [ux, uy] = direction(line);
  const mx = (edge.x1 + edge.x2) / 2 - line.x1;
  const my = (edge.y1 + edge.y2) / 2 - line.y1;
  return Math.abs(mx * uy - my * ux);
}

/** Length over which b runs alongside a, measured along a. */
function overlap(a, b) {
  const [ux, uy] = direction(a);
  const start = (b.x1 - a.x1) * ux + (b.y1 - a.y1) * uy;
  const end   = (b.x2 - a.x1) * ux + (b.y2 - a.y1) * uy;
  const low  = Math.max(0, Math.min(start, end));
  const high = Math.min(a.length, Math.max(start, end));
  return high - low;
}

/**
 * Whether the strap with edges a and b crosses the body's centre line.
 *
 * This is the test that separates a strap from suit lapels and shirt
 * collars, which are just as diagonal but converge toward the centre
 * and stop there.
 */
function crossesMidline(a, b, midline, share) {
  const xs = [a.x1, a.x2, b.x1, b.x2];
  const left = Math.min(...xs), right = Math.max(...xs);
  const span = right - left;
  if (span <= 0) return false;
  return Math.min(midline - left, right - midline) >= share * span;
}

/** The longest pair of edges that could be a strap's two sides. */
export function findStrap(edges, faceWidth, midline, limits) {
  const {
    minWidth, maxWidth, parallelTolerance, minOverlap, midlineShare
  } = limits;

  let best = null, bestLength = 0;

  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      const a = edges[i], b = edges[j];

      // Both diagonals are accepted, so left- and right-hand-drive
      // cars both work -- but the two edges of one strap lean the same
      // way as each other.
      if ((a.angle > 0) !== (b.angle > 0)) continue;
      if (Math.abs(a.angle - b.angle) > parallelTolerance) continue;

      const width = distanceToLine(b, a) / faceWidth;
      if (width < minWidth || width > maxWidth) continue;

      if (overlap(a, b) < minOverlap * Math.min(a.length, b.length)) continue;
      if (!crossesMidline(a, b, midline, midlineShare)) continue;

      const length = Math.min(a.length, b.length);
      if (length > bestLength) { best = [a, b]; bestLength = length; }
    }
  }

  return best;
}
