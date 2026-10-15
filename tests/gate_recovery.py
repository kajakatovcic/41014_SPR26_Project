"""Gate recovery for the EKF, plus a before/after comparison on the robustness scenarios.

Problem (seen in robustness_tests.py): with an unmodelled odometry bias, the estimate drifts between detections,
P stays small, the next (good) detection looks like an outlier, the gate rejects it, and the filter never recovers.

Fix: count CONSECUTIVE rejected detections. A genuine outlier is a one-off, so the streak resets at the next accepted
update. If the streak reaches `gate_streak`, the filter is probably lost rather than the detections being bad, so P is
inflated (position and heading) and the gate opens enough to let the next detection pull the estimate back.

Trade-off: a burst of gate_streak bad detections in a row (e.g. a persistent mirror image) would also trigger the
inflation, and the next detection after that is judged against a looser gate. Keep gate_streak high enough that this
is rare. Both knobs are below.

This is a subclass so it can be tested without editing estimation.py. If it helps, move the logic into EKF.update.

Run from the project root:  python -m tests.gate_recovery   (needs sweep_recordings.pkl from tests/tune_filter.py)
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.estimation.estimation import EKF


class RecoveringEKF(EKF):
    def __init__(self, x0, P0, gate_streak=5, inflate=(0.2, 0.2, np.deg2rad(15.0)), **kw):
        super().__init__(x0, P0, **kw)
        self.gate_streak = gate_streak                 # consecutive rejections that trigger recovery
        self.inflate = np.diag(np.square(inflate))     # added to P on recovery: (m, m, rad) standard deviations
        self.streak = 0
        self.n_recoveries = 0

    def update(self, marker_id, z, R):
        accepted = super().update(marker_id, z, R)
        if accepted:
            self.streak = 0
        else:
            self.streak += 1
            if self.streak >= self.gate_streak:
                self.P = self.P + self.inflate
                self.streak = 0
                self.n_recoveries += 1
        return accepted


if __name__ == "__main__":
    import tests.robustness_tset as rt

    if not rt.REC_FILE.exists():
        sys.exit(f"{rt.REC_FILE} not found: run tests/tune_filter.py first to record the data")
    recs = pickle.load(open(rt.REC_FILE, "rb"))

    keep = {"nominal (2% noise)", "heavy noise (5%)", "left wheel +2% scale", "both wheels +3% scale",
            "5% outliers, gate ON", "5% outliers, gate OFF", "camera delay 100 ms", "wrong start", "combined"}
    rt.SCENARIOS = [s for s in rt.SCENARIOS if s[0] in keep]
    K = 0.05

    print("######## BEFORE: plain EKF ########")
    rt.EKF = EKF
    rt.run_all(recs, K)
    print("\n######## AFTER: EKF with gate recovery ########")
    rt.EKF = RecoveringEKF
    rt.run_all(recs, K)

    # What to check:
    #  - 'left wheel +2%' and 'both wheels +3%' and 'combined': error should fall a lot and gate% should drop
    #  - nominal, heavy noise, outliers (gate ON), wrong start: should be unchanged (no regression)
    #  - 'outliers, gate OFF' is unaffected by design (nothing is ever rejected)
    #  - if the wheel rows improve only partly, the remaining error is drift between detections: the principled fix is
    #    to estimate the bias (an extra state), which is a model change to agree with part A
    #  - try gate_streak 3 and 10, and inflate heading 10 / 25 deg, to see how sensitive the result is