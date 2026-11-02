# Track Lab

Multi-object tracking from scratch in NumPy: the Hungarian algorithm, a Kalman filter, SORT and ByteTrack, together with MOTA, IDF1 and HOTA. It suits anyone who counts or follows objects in video, from people in drone footage to cells under a microscope.

```python
from track_lab import ByteTrack, evaluate, run_tracker, simulate

truth, detections = simulate(frames=120, objects=6, occlusion=(50, 62))
tracks = run_tracker(ByteTrack(), detections)
print(evaluate(truth, tracks))
```

With your own detector, call the tracker once per frame:

```python
tracker = ByteTrack()
for frame in video:
    rows = detect(frame)                # [(x, y, w, h, score), ...]
    for track_id, x, y, w, h in tracker.update(rows):
        ...
```

To compare the trackers on a simulated scene:

```
python -m track_lab
```

## What is inside

| Part | Details |
|---|---|
| `hungarian` | Minimum-cost assignment, with the same output form as SciPy's |
| `KalmanBox` | A constant-velocity filter on centre, area and aspect ratio |
| `SORT` | Kalman prediction plus IoU matching (Bewley et al. 2016), keeping only detections above a score threshold, as is usual |
| `ByteTrack` | Keeps low-score detections for a second round of matching, so tracks survive occlusion (Zhang et al. 2022) |
| `clear_mot` | MOTA, MOTP, misses, false alarms and identity switches |
| `identity_metrics` | IDF1, IDP and IDR (Ristani et al. 2016) |
| `hota` | HOTA, DetA and AssA (Luiten et al. 2021) |

## Which metric?

MOTA mostly measures detection. Two tracks that swap identities halfway barely change it, while IDF1 falls to 0.5, and a test shows exactly that. HOTA balances the two and reports them separately.

## How it is tested

The assignment solver is checked against brute force on 200 small matrices and, when SciPy is installed, against `linear_sum_assignment` on 100 larger ones. Metrics are checked on hand-worked cases with known answers. On simulated scenes with an occlusion period, ByteTrack must beat SORT on IDF1 in at least 7 of 8 runs.

## Limits

Trackers use box overlap only, with no appearance features, so long occlusions and fast motion across frames will cause identity switches. The simulator is simple, and results on it do not predict results on real video. The metrics were not compared with the official TrackEval or py-motmetrics tools, so use those for published numbers.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.
