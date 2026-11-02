"""Multi-object tracking from scratch: assignment, Kalman filtering, SORT and ByteTrack.

Boxes are (x, y, width, height). A tracker takes the detections of one frame at a time,
each as (x, y, w, h, score), and returns tracks as (track id, x, y, w, h).
"""
import numpy as np


def hungarian(cost):
    """Minimum-cost assignment on an n×m matrix, by shortest augmenting paths with potentials.
    Returns (rows, cols), in the same form as scipy.optimize.linear_sum_assignment."""
    c = np.asarray(cost, float)
    flip = c.shape[0] > c.shape[1]
    if flip:
        c = c.T
    n, m = c.shape
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, int)            # p[j]: the row matched to column j
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = c[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            k = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(k)) + 1
            delta = minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows, cols = np.array(rows, int), np.array(cols, int)
    if flip:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    w = np.minimum((a[:, 0] + a[:, 2])[:, None], (b[:, 0] + b[:, 2])[None]) - np.maximum(a[:, 0, None], b[None, :, 0])
    h = np.minimum((a[:, 1] + a[:, 3])[:, None], (b[:, 1] + b[:, 3])[None]) - np.maximum(a[:, 1, None], b[None, :, 1])
    inter = np.clip(w, 0, None) * np.clip(h, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


class KalmanBox:
    """Constant-velocity Kalman filter over centre, area and aspect ratio, as in SORT."""
    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1
    H = np.eye(4, 7)
    R = np.diag([1, 1, 10, 10.0])
    Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.0001])

    def __init__(self, box):
        self.x = np.zeros(7)
        self.x[:4] = self.measure(box)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4])

    @staticmethod
    def measure(box):
        x, y, w, h = box[:4]
        return np.array([x + w / 2, y + h / 2, w * h, w / max(h, 1e-6)])

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        y = self.measure(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def box(self):
        cx, cy, s, r = self.x[:4]
        s, r = max(s, 1e-6), max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / w
        return np.array([cx - w / 2, cy - h / 2, w, h])


class Track:
    next_id = 1

    def __init__(self, box):
        self.kf = KalmanBox(box)
        self.id = None
        self.hits = 1
        self.lost = 0

    def assign_id(self, tracker):
        if self.id is None:
            tracker.counter += 1
            self.id = tracker.counter


class SORT:
    """Simple online and realtime tracking (Bewley et al. 2016)."""

    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3, min_score=0.6):
        self.max_age, self.min_hits, self.iou, self.min_score = max_age, min_hits, iou_threshold, min_score
        self.tracks = []
        self.frame = 0
        self.counter = 0

    def _match(self, tracks, boxes, threshold):
        ious = iou_matrix([t.kf.box() for t in tracks], boxes)
        pairs, un_t, un_d = [], set(range(len(tracks))), set(range(len(boxes)))
        if ious.size:
            for r, c in zip(*hungarian(-ious)):
                if ious[r, c] >= threshold:
                    pairs.append((int(r), int(c)))
                    un_t.discard(int(r))
                    un_d.discard(int(c))
        return pairs, sorted(un_t), sorted(un_d)

    def update(self, detections):
        self.frame += 1
        det = np.asarray(detections, float).reshape(-1, 5)
        det = det[det[:, 4] >= self.min_score]         # the usual practice, which ByteTrack sets out to improve on
        for t in self.tracks:
            t.kf.predict()
        pairs, un_t, un_d = self._match(self.tracks, det[:, :4], self.iou)
        for ti, di in pairs:
            t = self.tracks[ti]
            t.kf.update(det[di])
            t.hits += 1
            t.lost = 0
        for ti in un_t:
            self.tracks[ti].lost += 1
        for di in un_d:
            self.tracks.append(Track(det[di]))
        self.tracks = [t for t in self.tracks if t.lost <= self.max_age]
        return self._report()

    def _report(self):
        out = []
        for t in self.tracks:
            if t.lost == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits):
                t.assign_id(self)
                out.append((t.id, *t.kf.box()))
        return out


class ByteTrack(SORT):
    """ByteTrack (Zhang et al. 2022): low-score detections are not thrown away. A second round
    uses them to keep tracks alive through occlusion, but they never start a new track."""

    def __init__(self, high=0.6, low=0.1, max_age=30, min_hits=1, iou_threshold=0.3, second_iou=0.5, new_track=0.7):
        super().__init__(max_age, min_hits, iou_threshold)
        self.high, self.low, self.second, self.new_track = high, low, second_iou, new_track

    def update(self, detections):
        self.frame += 1
        det = np.asarray(detections, float).reshape(-1, 5)
        strong = det[det[:, 4] >= self.high]
        weak = det[(det[:, 4] >= self.low) & (det[:, 4] < self.high)]
        for t in self.tracks:
            t.kf.predict()
        pairs, un_t, un_d = self._match(self.tracks, strong[:, :4], self.iou)
        for ti, di in pairs:
            self._hit(self.tracks[ti], strong[di])
        left = [self.tracks[i] for i in un_t]
        pairs, still, _ = self._match(left, weak[:, :4], self.second)
        for ti, di in pairs:
            self._hit(left[ti], weak[di])
        for i in still:
            left[i].lost += 1
        for di in un_d:
            if strong[di, 4] >= self.new_track:
                self.tracks.append(Track(strong[di]))
        self.tracks = [t for t in self.tracks if t.lost <= self.max_age]
        return self._report()

    @staticmethod
    def _hit(t, d):
        t.kf.update(d)
        t.hits += 1
        t.lost = 0


def simulate(frames=60, objects=5, seed=0, size=(640, 480), jitter=2.0, miss=0.1, false_rate=0.5, occlusion=None):
    """A synthetic scene. Returns (ground truth, detections) as lists, one per frame.
    Ground truth rows are (id, x, y, w, h). Detections are (x, y, w, h, score).
    occlusion: (start, stop) frames in which every object gets weak scores, as in a crowd."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform((20, 20), (size[0] - 80, size[1] - 80), (objects, 2))
    vel = rng.uniform(-5, 5, (objects, 2))
    dim = rng.uniform(30, 70, (objects, 2))
    truth, dets = [], []
    for f in range(frames):
        pos = pos + vel
        bounce = (pos < 0) | (pos + dim > size)
        vel[bounce] *= -1
        pos = np.clip(pos, 0, np.array(size) - dim)
        gt = [(i + 1, *pos[i], *dim[i]) for i in range(objects)]
        truth.append(gt)
        row = []
        for i in range(objects):
            weak = occlusion is not None and occlusion[0] <= f < occlusion[1]
            if not weak and rng.random() < miss:
                continue
            b = np.r_[pos[i], dim[i]] + rng.normal(0, jitter, 4)
            row.append((*b, rng.uniform(0.2, 0.5) if weak else rng.uniform(0.7, 1.0)))
        for _ in range(rng.poisson(false_rate)):
            row.append((*rng.uniform(0, 500, 2), 50, 50, rng.uniform(0.1, 0.65)))
        dets.append(row)
    return truth, dets


def run_tracker(tracker, detections):
    return [tracker.update(d) for d in detections]


def clear_mot(truth, tracks, threshold=0.5):
    """CLEAR MOT metrics. Keeps last frame's matches where still valid, then assigns the rest."""
    gt_total = fp = fn = switches = 0
    overlap = matches = 0
    last = {}
    prev = {}
    for gt, hyp in zip(truth, tracks):
        gt_ids, h_ids = [g[0] for g in gt], [h[0] for h in hyp]
        ious = iou_matrix([g[1:5] for g in gt], [h[1:5] for h in hyp])
        gt_total += len(gt)
        current = {}
        used = set()
        for gi, g in enumerate(gt_ids):
            p = prev.get(g)
            if p in h_ids and ious[gi, h_ids.index(p)] >= threshold:
                current[g] = p
                used.add(p)
        rest_g = [i for i, g in enumerate(gt_ids) if g not in current]
        rest_h = [j for j, h in enumerate(h_ids) if h not in used]
        if rest_g and rest_h:
            sub = ious[np.ix_(rest_g, rest_h)]
            for r, c in zip(*hungarian(-sub)):
                if sub[r, c] >= threshold:
                    current[gt_ids[rest_g[r]]] = h_ids[rest_h[c]]
        for g, h in current.items():
            if g in last and last[g] != h:
                switches += 1
            last[g] = h
            overlap += ious[gt_ids.index(g), h_ids.index(h)]
            matches += 1
        fn += len(gt) - len(current)
        fp += len(hyp) - len(current)
        prev = current
    return {"MOTA": 1 - (fn + fp + switches) / gt_total if gt_total else 0.0, "MOTP": overlap / matches if matches else 0.0, "FN": fn, "FP": fp, "IDSW": switches, "matches": matches}


def identity_metrics(truth, tracks, threshold=0.5):
    """IDF1 (Ristani et al. 2016): the best one-to-one mapping of whole tracks, by overlap frames."""
    g_ids = sorted({g[0] for f in truth for g in f})
    h_ids = sorted({h[0] for f in tracks for h in f})
    gi, hi = {g: i for i, g in enumerate(g_ids)}, {h: i for i, h in enumerate(h_ids)}
    n, m = len(g_ids), len(h_ids)
    if n == 0 or m == 0:
        return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0, "IDTP": 0}
    overlap = np.zeros((n, m))
    for gt, hyp in zip(truth, tracks):
        ious = iou_matrix([g[1:5] for g in gt], [h[1:5] for h in hyp])
        for a, g in enumerate(gt):
            for b, h in enumerate(hyp):
                if ious[a, b] >= threshold:
                    overlap[gi[g[0]], hi[h[0]]] += 1
    r, c = hungarian(-overlap)
    idtp = overlap[r, c].sum()
    ng = sum(len(f) for f in truth)
    nh = sum(len(f) for f in tracks)
    return {"IDF1": 2 * idtp / (ng + nh), "IDP": idtp / nh, "IDR": idtp / ng, "IDTP": int(idtp)}


def hota(truth, tracks):
    """Higher Order Tracking Accuracy (Luiten et al. 2021): the balance of detection and association."""
    alphas = np.arange(0.05, 0.96, 0.05)
    g_ids = sorted({g[0] for f in truth for g in f})
    h_ids = sorted({h[0] for f in tracks for h in f})
    gi, hi = {g: i for i, g in enumerate(g_ids)}, {h: i for i, h in enumerate(h_ids)}
    n, m = len(g_ids), len(h_ids)
    if n == 0 or m == 0:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0}
    potential = np.zeros((n, m))
    frames = []
    gc, hc = np.zeros(n), np.zeros(m)
    for gt, hyp in zip(truth, tracks):
        ious = iou_matrix([g[1:5] for g in gt], [h[1:5] for h in hyp])
        g_idx = [gi[g[0]] for g in gt]
        h_idx = [hi[h[0]] for h in hyp]
        frames.append((g_idx, h_idx, ious))
        for g in g_idx:
            gc[g] += 1
        for h in h_idx:
            hc[h] += 1
        if ious.size:
            denom = ious.sum(axis=0, keepdims=True) + ious.sum(axis=1, keepdims=True) - ious
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(denom > 0, ious / denom, 0.0)
            for a, g in enumerate(g_idx):
                for b, h in enumerate(h_idx):
                    potential[g, h] += s[a, b]
    global_score = potential / np.maximum(gc[:, None] + hc[None] - potential, 1e-10)
    rows = []
    ng, nh = gc.sum(), hc.sum()
    for alpha in alphas:
        count = np.zeros((n, m))
        tp = 0
        for g_idx, h_idx, ious in frames:
            if ious.size == 0:
                continue
            score = global_score[np.ix_(g_idx, h_idx)] * ious
            r, c = hungarian(-score)
            for a, b in zip(r, c):
                if ious[a, b] >= alpha - 1e-10:
                    tp += 1
                    count[g_idx[a], h_idx[b]] += 1
        fn, fp = ng - tp, nh - tp
        det = tp / max(tp + fn + fp, 1e-10)
        if tp:
            a_score = count / np.maximum(gc[:, None] + hc[None] - count, 1e-10)
            ass = (count * a_score).sum() / tp
        else:
            ass = 0.0
        rows.append((np.sqrt(det * ass), det, ass))
    a = np.mean(rows, axis=0)
    return {"HOTA": float(a[0]), "DetA": float(a[1]), "AssA": float(a[2])}


def evaluate(truth, tracks):
    return {**clear_mot(truth, tracks), **identity_metrics(truth, tracks), **hota(truth, tracks)}
