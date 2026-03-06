# trivial_estimator.py
import numpy as np
import stim
import pymatching as pm
from typing import Optional, Tuple


class SyndromeWeightEstimator:
    """
    Decoder-agnostic filter based on normalized syndrome weight.

    Score: q(s) = wt(s) / n_det  ∈ [0, 1]
    Policy: accept iff q(s) <= tau

    This implements Algorithm "Syndrome-Weight Ratio Filter":
      - Smaller tau -> stricter filtering (lower conditional LER, lower throughput)
      - Larger  tau -> more acceptance (higher throughput, potentially higher LER)
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        dem: Optional[stim.DetectorErrorModel] = None,
    ):
        self.circuit = circuit
        self.dem = dem or circuit.detector_error_model(decompose_errors=True)

        # Baseline decoder for recovery on accepted shots (same as your exact estimator)
        self.matcher_std = pm.Matching.from_detector_error_model(self.dem)

        # Number of detectors = length of syndrome bitstring
        self.n_det = self._num_detectors(self.dem)
        if self.n_det <= 0:
            raise RuntimeError("DEM has zero detectors; cannot use syndrome-weight filter.")

    # ---------- public API ----------

    def score_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Return normalized weights q(s) = wt(s) / n_det for each shot.
        """
        # syndromes is shape (shots, n_det) with {0,1} uint8/boolean entries
        # Use uint32 accumulation to avoid small overhead
        w = np.sum(syndromes, axis=1, dtype=np.uint32).astype(np.float64)
        return w / float(self.n_det)

    def accept_mask(self, scores: np.ndarray, tau: float) -> np.ndarray:
        """
        Accept iff normalized weight <= tau.
        """
        return scores <= float(tau)

    def evaluate_batch(
        self,
        syndromes: np.ndarray,
        actual_observables: np.ndarray,
        tau: float,
    ) -> Tuple[float, float]:
        """
        Return (acceptance_rate, conditional_logical_error_on_accepted).

        Recovery uses the STANDARD PyMatching decoder on the original DEM,
        identical to your ZCosetGapEstimator.evaluate_batch.
        """
        pred = self.matcher_std.decode_batch(syndromes)
        scores = self.score_batch(syndromes)
        mask = self.accept_mask(scores, tau)

        if mask.sum() == 0:
            return 0.0, 1.0

        cond_logical_err = np.mean((pred[mask] ^ actual_observables[mask]).astype(np.bool_))
        acc_rate = mask.mean()
        return float(acc_rate), float(cond_logical_err)

    # ---------- threshold helper ----------

    @staticmethod
    def quantile_threshold(scores: np.ndarray, target_accept_p: float) -> float:
        """
        Choose tau so that P(score <= tau) ~= target_accept_p, tie-aware.

        Intended usage:
          scores = estimator.score_batch(synd_calib)  # normalized weights
          tau    = estimator.quantile_threshold(scores, target_accept_p)
        """
        s = np.sort(scores)  # ascending
        N = len(s)
        # Index for target_accept_p quantile (shots with score <= tau are accepted)
        k = int(np.floor(target_accept_p * N)) - 1
        k = max(0, min(k, N - 1))
        v = s[k]

        # Move forward through any ties at v to place tau BETWEEN unique values
        j = k
        while j + 1 < N and s[j + 1] == v:
            j += 1

        if j + 1 < N:
            # Put tau halfway between v and next larger distinct value
            return float(0.5 * (v + s[j + 1]))
        else:
            # v is the maximum; nudge a hair above so <= tau accepts exactly those == max
            return float(v) + 1e-12

    # ---------- DEM / detector utilities ----------

    @staticmethod
    def _num_detectors(dem: stim.DetectorErrorModel) -> int:
        if hasattr(dem, "num_detectors"):
            return int(dem.num_detectors)
        count = 0
        for inst in dem.flattened():
            if inst.type == "detector":
                count += 1
        if count > 0:
            return count
        # Last-resort scan of targets (older Stim)
        max_det = -1
        for inst in dem:
            if inst.type in ("error", "detector_error", "detector"):
                for trg in inst.targets_copy():
                    if hasattr(trg, "is_relative_detector_id") and trg.is_relative_detector_id():
                        max_det = max(max_det, int(getattr(trg, "val", trg.val)))
        return max_det + 1
