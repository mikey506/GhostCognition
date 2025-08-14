import numpy as np
from typing import List, Dict, Any

EPS = 1e-12
DEFAULT_QUBIT = np.array([1.0+0.0j, 0.0+0.0j], dtype=np.complex128)

class ArchetypeEngine:
    def __init__(self, core=None, logger=None):
        self.core = core
        self.logger = logger
        self._last_good_state = DEFAULT_QUBIT.copy()
        # Simple built-ins
        self.archetypes: Dict[str, Dict[str, Any]] = {
            "hero":   {"polarity": "yang",  "strength": 0.7},
            "mentor": {"polarity": "yin",   "strength": 0.6},
            "shadow": {"polarity": "yin",   "strength": 0.8},
        }

    def _normalize_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.complex128).ravel()
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < EPS:
            # fall back to last good state instead of |0>
            return self._last_good_state.copy()
        out = (q / n).astype(np.complex128, copy=False)
        self._last_good_state = out
        return out

    def _create_archetype_qubit(self, name: str, strength: float) -> np.ndarray:
        # Map polarity/strength to a qubit on the X-Z plane
        info = self.archetypes.get(name.lower(), {"polarity":"neutral","strength":0.5})
        s = np.clip(strength if np.isfinite(strength) else info["strength"], 0.0, 1.0)
        theta = s * (np.pi/2)  # 0 -> |0>, 1 -> |+>
        cos = np.cos(theta); sin = np.sin(theta)
        return np.array([cos, sin], dtype=np.complex128)

    def _emergence_entropy(self, q: np.ndarray) -> float:
        psi = self._normalize_q(q)
        if psi.size == 2:
            p = np.clip([abs(psi[0])**2, abs(psi[1])**2], EPS, 1.0)
            H = float(-(p[0]*np.log2(p[0]) + p[1]*np.log2(p[1])))
            return H # in [0,1]
        elif psi.size == 4:
            a00,a01,a10,a11 = psi
            p0 = abs(a00)**2 + abs(a01)**2
            p1 = abs(a10)**2 + abs(a11)**2
            p = np.clip([p0,p1], EPS, 1.0)
            H = float(-(p[0]*np.log2(p[0]) + p[1]*np.log2(p[1])))
            return H
        return 0.0

    def process_archetype(self, names: List[str], weights: List[float]) -> np.ndarray:
        if not names or not weights or len(names) != len(weights):
            raise ValueError("names and weights must be same-length non-empty lists")
        # Combine as weighted sum of archetype qubits
        qs = []
        for name, w in zip(names, weights):
            info = self.archetypes.get(name.lower(), {"polarity":"neutral","strength":0.5})
            qi = self._create_archetype_qubit(name, info["strength"])
            qs.append(self._normalize_q(qi) * float(w))
        combined = np.sum(qs, axis=0)
        combined = self._normalize_q(combined)
        # Optional: error-correct-like nudge towards legitimate qubit
        emergence = self._emergence_entropy(combined)
        if self.logger:
            self.logger.info(f"[archetype_engine] Archetypal superposition complete | Emergence={emergence:.4f} | ψ={combined}")
        return combined