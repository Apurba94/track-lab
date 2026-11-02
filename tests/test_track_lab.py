import itertools
import unittest

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from track_lab import SORT, ByteTrack, KalmanBox, clear_mot, evaluate, hota, hungarian, identity_metrics, iou_matrix, run_tracker, simulate


class TestHungarian(unittest.TestCase):
    def test_small(self):
        r, c = hungarian([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        self.assertEqual(np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])[r, c].sum(), 5)

    def test_rectangular_both_ways(self):
        wide = np.array([[1.0, 9, 9, 2], [9, 1, 9, 2]])
        r, c = hungarian(wide)
        self.assertEqual(wide[r, c].sum(), 2)
        r, c = hungarian(wide.T)
        self.assertEqual(wide.T[r, c].sum(), 2)

    def test_empty(self):
        self.assertEqual(len(hungarian(np.zeros((0, 3)))[0]), 0)

    def test_against_brute_force(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
            c = rng.integers(0, 20, (n, m)).astype(float)
            r, k = hungarian(c)
            self.assertEqual(len(r), min(n, m))
            if n <= m:
                best = min(sum(c[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
            else:
                best = min(sum(c[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))
            self.assertAlmostEqual(c[r, k].sum(), best)

    @unittest.skipIf(linear_sum_assignment is None, "SciPy is not installed")
    def test_against_scipy(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            c = rng.normal(size=(rng.integers(1, 60), rng.integers(1, 60)))
            a, b = hungarian(c), linear_sum_assignment(c)
            self.assertAlmostEqual(c[a].sum(), c[b].sum(), places=8)


class TestKalman(unittest.TestCase):
    def test_learns_velocity_and_predicts(self):
        k = KalmanBox((0, 0, 50, 50))
        for t in range(1, 30):
            k.predict()
            k.update((t * 4.0, t * 2.0, 50, 50))
        p = k.predict()
        self.assertAlmostEqual(p[0], 120, delta=1.5)
        self.assertAlmostEqual(p[1], 60, delta=1.5)

    def test_smooths_noise(self):
        rng = np.random.default_rng(0)
        k = KalmanBox((0, 0, 50, 50))
        raw, filtered = [], []
        for t in range(1, 80):
            noisy = (t * 3.0 + rng.normal(0, 4), rng.normal(0, 4), 50, 50)
            k.predict()
            k.update(noisy)
            raw.append(abs(noisy[0] - t * 3))
            filtered.append(abs(k.box()[0] - t * 3))
        self.assertLess(np.mean(filtered[20:]), np.mean(raw[20:]))

    def test_box_round_trip(self):
        self.assertTrue(np.allclose(KalmanBox((3, 4, 20, 10)).box(), (3, 4, 20, 10)))


class TestTrackers(unittest.TestCase):
    def test_single_object_keeps_one_id(self):
        for tr in (SORT(), ByteTrack()):
            frames = [[(t * 5.0, 10, 40, 40, 0.9)] for t in range(30)]
            ids = {row[0] for out in run_tracker(tr, frames) for row in out}
            self.assertEqual(len(ids), 1)

    def test_track_survives_a_short_gap(self):
        frames = [[(t * 3.0, 0, 50, 50, 0.9)] if not 10 <= t < 12 else [] for t in range(25)]
        out = run_tracker(SORT(max_age=3), frames)
        self.assertEqual(len({r[0] for f in out for r in f}), 1)

    def test_long_gap_starts_a_new_track(self):
        frames = [[(t * 3.0, 0, 50, 50, 0.9)] if not 10 <= t < 20 else [] for t in range(30)]
        out = run_tracker(SORT(max_age=3), frames)
        self.assertEqual(len({r[0] for f in out for r in f}), 2)

    def test_min_hits_holds_back_a_lone_false_alarm(self):
        tr = SORT(min_hits=3)
        for t in range(5):
            tr.update([(0, 0, 50, 50, 0.9)])
        out = tr.update([(0, 0, 50, 50, 0.9), (300, 300, 50, 50, 0.9)])
        self.assertEqual(len(out), 1)

    def test_bytetrack_ignores_weak_new_objects(self):
        tr = ByteTrack()
        self.assertEqual(tr.update([(0, 0, 50, 50, 0.3)]), [])
        self.assertEqual(tr.tracks, [])

    def test_bytetrack_keeps_tracks_alive_on_weak_detections(self):
        tr = ByteTrack()
        for t in range(5):
            tr.update([(0, 0, 50, 50, 0.9)])
        ids = {r[0] for r in tr.update([(0, 0, 50, 50, 0.3)])}
        self.assertEqual(len(ids), 1)
        self.assertEqual(tr.tracks[0].lost, 0)

    def test_empty_frames(self):
        self.assertEqual(SORT().update([]), [])
        self.assertEqual(ByteTrack().update([]), [])

    def test_clean_scene_is_nearly_perfect(self):
        truth, dets = simulate(frames=80, objects=4, seed=1, miss=0, false_rate=0, jitter=0.5)
        r = evaluate(truth, run_tracker(SORT(min_hits=1), dets))
        self.assertGreater(r["MOTA"], 0.97)
        self.assertGreater(r["IDF1"], 0.95)
        self.assertGreater(r["HOTA"], 0.75)

    def test_bytetrack_beats_sort_through_occlusion(self):
        wins = 0
        for seed in range(8):
            truth, dets = simulate(frames=100, objects=6, seed=seed, occlusion=(40, 55))
            a = evaluate(truth, run_tracker(SORT(), dets))
            b = evaluate(truth, run_tracker(ByteTrack(), dets))
            wins += b["IDF1"] > a["IDF1"]
        self.assertGreaterEqual(wins, 7)


def track(*rows):
    return [list(r) for r in rows]


BOX = (0, 0, 100, 100)
FAR = (500, 500, 100, 100)


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        gt = [[(1, *BOX), (2, *FAR)] for _ in range(10)]
        r = evaluate(gt, gt)
        self.assertEqual((r["MOTA"], r["IDF1"], r["IDSW"]), (1, 1, 0))
        self.assertAlmostEqual(r["MOTP"], 1)
        self.assertAlmostEqual(r["HOTA"], 1)

    def test_misses_and_false_alarms(self):
        gt = [[(1, *BOX)] for _ in range(10)]
        hyp = [[(7, *BOX)] for _ in range(6)] + [[] for _ in range(4)]
        r = clear_mot(gt, hyp)
        self.assertEqual((r["FN"], r["FP"], r["IDSW"]), (4, 0, 0))
        self.assertAlmostEqual(r["MOTA"], 0.6)
        extra = [[(7, *BOX), (8, *FAR)] for _ in range(10)]
        self.assertEqual(clear_mot(gt, extra)["FP"], 10)

    def test_id_switch(self):
        gt = [[(1, *BOX)] for _ in range(10)]
        hyp = [[(1, *BOX)] for _ in range(5)] + [[(2, *BOX)] for _ in range(5)]
        r = evaluate(gt, hyp)
        self.assertEqual(r["IDSW"], 1)
        self.assertAlmostEqual(r["MOTA"], 0.9)
        self.assertAlmostEqual(r["IDF1"], 0.5)

    def test_idf1_punishes_what_mota_forgives(self):
        gt = [[(1, *BOX), (2, *FAR)] for _ in range(20)]
        swap = [[(1, *BOX), (2, *FAR)] for _ in range(10)] + [[(2, *BOX), (1, *FAR)] for _ in range(10)]
        r = evaluate(gt, swap)
        self.assertGreater(r["MOTA"], 0.9)
        self.assertAlmostEqual(r["IDF1"], 0.5)

    def test_hota_splits_detection_and_association(self):
        gt = [[(1, *BOX)] for _ in range(10)]
        half_id = [[(1, *BOX)] for _ in range(5)] + [[(2, *BOX)] for _ in range(5)]
        r = hota(gt, half_id)
        self.assertAlmostEqual(r["DetA"], 1)
        self.assertAlmostEqual(r["AssA"], 0.5, delta=1e-6)
        self.assertAlmostEqual(r["HOTA"], np.sqrt(0.5), delta=1e-6)

    def test_hota_of_nothing(self):
        self.assertEqual(hota([[(1, *BOX)]], [[]])["HOTA"], 0)

    def test_iou_threshold_matters(self):
        gt = [[(1, 0, 0, 100, 100)]]
        hyp = [[(1, 40, 0, 100, 100)]]
        self.assertEqual(clear_mot(gt, hyp)["FN"], 1)
        self.assertEqual(clear_mot(gt, hyp, 0.4)["FN"], 0)

    def test_empty_hypothesis(self):
        r = identity_metrics([[(1, *BOX)]], [[]])
        self.assertEqual(r["IDF1"], 0)

    def test_iou(self):
        self.assertAlmostEqual(iou_matrix([[0, 0, 10, 10]], [[5, 0, 10, 10]])[0, 0], 1 / 3)


if __name__ == "__main__":
    unittest.main()
