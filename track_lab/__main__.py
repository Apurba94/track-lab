"""Compare SORT and ByteTrack on a simulated scene with a period of occlusion: python -m track_lab [seed]"""
import sys

from . import SORT, ByteTrack, evaluate, run_tracker, simulate


def main(argv):
    seed = int(argv[1]) if len(argv) > 1 else 0
    truth, dets = simulate(frames=120, objects=6, seed=seed, occlusion=(50, 62))
    for name, tracker in (("SORT", SORT()), ("ByteTrack", ByteTrack())):
        r = evaluate(truth, run_tracker(tracker, dets))
        print(f"{name:<10} MOTA {r['MOTA']:.3f}  IDF1 {r['IDF1']:.3f}  HOTA {r['HOTA']:.3f}  ID switches {r['IDSW']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
