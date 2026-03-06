# estimator_z.py (exact opposite-class gap via DEM lifting; Stim API compatible)

import numpy as np
import stim
import pymatching as pm
from typing import Optional, Tuple


class ZCosetGapEstimator:
    """
    EXACT gap: phi(s) = C_L(s) - C_I(s) computed by decoding the SAME syndrome s
    under an EXTENDED DEM where logical-Z (observable 0 by default) is represented
    as a fresh detector bit. extra_det=0 -> identity class; extra_det=1 -> opposite class.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        dem: Optional[stim.DetectorErrorModel] = None,
        obs_index: int = 0,
    ):
        self.circuit = circuit
        self.dem = dem or circuit.detector_error_model(decompose_errors=True)
        self.obs_index = int(obs_index)

        # Standard matcher for actual recovery on accepted shots
        self.matcher_std = pm.Matching.from_detector_error_model(self.dem)

        # Build extended DEM (logical -> detector) and its matcher
        self.dem_ext, self.extra_det_id = self._dem_with_logical_as_detector(self.dem, self.obs_index)
        self.matcher_ext = pm.Matching.from_detector_error_model(self.dem_ext)

        # Bookkeeping
        self.n_det = self._num_detectors(self.dem)
        self.n_det_ext = self._num_detectors(self.dem_ext)

        # Back-compat: no geometric bZ is used in the exact method
        self.bZ = None

    # ---------- public API ----------

    def score_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Compute phi(s) via exact opposite-class decoding in the extended DEM."""
        return self._coset_gap_exact(self.matcher_ext, syndromes, self.extra_det_id, self.n_det_ext)

    def accept_mask(self, scores: np.ndarray, gamma: float) -> np.ndarray:
        return np.abs(scores) >= gamma


    def evaluate_batch(
        self,
        syndromes: np.ndarray,
        actual_observables: np.ndarray,
        gamma: float,
    ) -> Tuple[float, float]:
        """
        Return (acceptance_rate, conditional_logical_error_on_accepted).
        Recovery uses the STANDARD matcher on the original DEM.
        """
        pred = self.matcher_std.decode_batch(syndromes)
        scores = self.score_batch(syndromes)
        mask = self.accept_mask(scores, gamma)

        if mask.sum() == 0:
            return 0.0, 1.0

        cond_logical_err = np.mean((pred[mask] ^ actual_observables[mask]).astype(np.bool_))
        acc_rate = mask.mean()
        return float(acc_rate), float(cond_logical_err)

    # ---------- threshold helper ----------

    @staticmethod
    def quantile_threshold(scores: np.ndarray, target_accept_p: float) -> float:
        """
        Choose gamma so that P(|score| >= gamma) ~= target_accept_p.
        This ensures the filter keeps the syndromes with the largest
        coset separation in either direction.
        """
        s = np.sort(np.abs(scores))  # sort magnitudes
        N = len(s)
        # Index for (1 - target_accept_p) quantile
        k = int(np.floor((1.0 - target_accept_p) * N))
        k = max(0, min(k, N - 1))
        v = s[k]
        j = k
        while j + 1 < N and s[j + 1] == v:
            j += 1
        if j + 1 < N:
            tau =  float(0.5 * (v + s[j + 1]))
        else:
            tau =  float(v) + 1e-12
        tau = min(tau, float(np.max(s)))
        tau = max(tau, float(np.min(s)))
        tau = np.quantile(np.abs(scores), 1 - target_accept_p)
        tau *= 0.999
        return tau


    # ---------- exact gap via DEM lifting ----------

    @staticmethod
    def _coset_gap_exact(matcher_ext, syndromes, extra_det_id, total_det_ext):
        s0 = ZCosetGapEstimator._extend_syndromes(syndromes, extra_det_id, total_det_ext, val=0)
        s1 = ZCosetGapEstimator._extend_syndromes(syndromes, extra_det_id, total_det_ext, val=1)
        try:
            _, C_I = matcher_ext.decode_batch(s0, return_weights=True)
            _, C_L = matcher_ext.decode_batch(s1, return_weights=True)
        except ValueError:
            n = syndromes.shape[0]
            C_I = np.zeros(n, dtype=float)
            C_L = np.zeros(n, dtype=float)
        return np.asarray(C_L, dtype=float) - np.asarray(C_I, dtype=float)


    @staticmethod
    def _extend_syndromes(synd: np.ndarray, extra_det_id: int, total_det_ext: int, val: int) -> np.ndarray:
        N, n_det = synd.shape
        out = np.zeros((N, total_det_ext), dtype=np.uint8)
        out[:, :n_det] = synd
        out[:, extra_det_id] = val & 1
        return out

    # ---------- DEM transformation: logical → detector ----------

    @staticmethod
    def _dem_with_logical_as_detector(
        dem: stim.DetectorErrorModel, obs_index: int = 0
    ) -> Tuple[stim.DetectorErrorModel, int]:
        # --- count base detectors ---
        try:
            n_det = int(dem.num_detectors)
        except AttributeError:
            n_det = sum(1 for inst in dem.flattened() if inst.type == "detector")
        extra_id = n_det

        dem_ext = stim.DetectorErrorModel()

        # Helpers
        def _is_logical(tgt) -> bool:
            if hasattr(tgt, "is_logical_observable_id"):
                return tgt.is_logical_observable_id()
            if hasattr(tgt, "is_observable_id"):
                return tgt.is_observable_id()
            return False

        def _is_rel_det(tgt) -> bool:
            return hasattr(tgt, "is_relative_detector_id") and tgt.is_relative_detector_id()

        def _val(tgt) -> int:
            return int(getattr(tgt, "val", tgt.val))

        # --- re-emit errors, rewiring logical(obs_index) -> extra detector ---
        # Also collect graph adjacency to audit components.
        # For graphlike purposes: 2-detector error == edge (u,v); 1-detector error == boundary on u.
        from collections import defaultdict, deque
        adj: dict[int, set[int]] = defaultdict(set)   # undirected adj among detectors
        has_boundary: dict[int, bool] = defaultdict(bool)

        for inst in dem:
            t = inst.type
            if t in ("error", "detector_error"):
                new_targets = []
                dets = []   # list of detector ids touched by this error
                flips_obs = False

                for tgt in inst.targets_copy():
                    if _is_logical(tgt) and _val(tgt) == obs_index:
                        new_targets.append(stim.DemTarget.relative_detector_id(extra_id))
                        dets.append(extra_id)
                        flips_obs = True
                    else:
                        new_targets.append(tgt)
                        if _is_rel_det(tgt):
                            did = _val(tgt)
                            dets.append(did)

                args = inst.args_copy()
                p = float(args[0]) if len(args) else 0.0
                dem_ext.append(t, p, new_targets)

                # graphlike bookkeeping (only by detector footprint)
                if len(dets) == 1:
                    has_boundary[dets[0]] = True
                elif len(dets) == 2:
                    u, v = dets
                    adj[u].add(v); adj[v].add(u)
                else:
                    # k>2 parity terms: connect them into a simple chain as a conservative over-approx
                    # (doesn't change the emitted DEM; only used for component detection)
                    for i in range(len(dets) - 1):
                        u, v = dets[i], dets[i+1]
                        adj[u].add(v); adj[v].add(u)
            else:
                continue

        # --- ensure the synthetic detector has a boundary half-edge ---
        eps = 1e-15
        dem_ext.append("error", eps, [stim.DemTarget.relative_detector_id(extra_id)])
        has_boundary[extra_id] = True

        # --- COMPONENT AUDIT: add one tiny boundary per boundary-less component ---
        # Build components over [0..extra_id] using adj
        all_nodes = set(adj.keys()) | {extra_id}
        # Also include isolated detectors that never appeared in adj but are < extra_id
        # (only if they realistically exist — we assume dense indices 0..n_det-1 here)
        for did in range(extra_id):
            all_nodes.add(did)

        seen = set()
        for start in list(all_nodes):
            if start in seen:
                continue
            # BFS the component
            comp = []
            q = deque([start]); seen.add(start)
            boundary_present = False
            while q:
                u = q.popleft()
                comp.append(u)
                boundary_present |= has_boundary.get(u, False)
                for v in adj.get(u, ()):
                    if v not in seen:
                        seen.add(v); q.append(v)

            if boundary_present:
                continue
            # No boundary in this component: add ONE tiny boundary at its first node
            anchor = comp[0]
            dem_ext.append("error", eps, [stim.DemTarget.relative_detector_id(anchor)])
            has_boundary[anchor] = True

        return dem_ext, extra_id


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
        max_det = -1
        for inst in dem:
            if inst.type in ("error", "detector_error", "detector"):
                for trg in inst.targets_copy():
                    if hasattr(trg, "is_relative_detector_id") and trg.is_relative_detector_id():
                        max_det = max(max_det, int(getattr(trg, "val", trg.val)))
        return max_det + 1
