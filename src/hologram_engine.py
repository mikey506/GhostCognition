import numpy as np
from typing import Any, Sequence, Union

EPS = 1e-12
DEFAULT_QUBIT = np.array([1.0+0.0j, 0.0+0.0j], dtype=np.complex128)

def _as_complex_array(x: Any) -> np.ndarray:
    """
    Robust coercion of CLI- or code-provided inputs to complex numpy arrays.
    Accepts: list/tuple/ndarray/strings like "[0.707, 0.707]" or "[0.5+0.5j, 0.5-0.5j]".
    """
    if isinstance(x, np.ndarray):
        arr = x.astype(np.complex128, copy=False)
        return arr
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.complex128)
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        s = x.strip()
        # Try JSON-ish quick normalization: replace single quotes, ensure commas between spaces if missing
        try:
            # Allow something like "[0.707, 0.707]" or "[0.5+0.5j, 0.5-0.5j]"
            # ast.literal_eval can parse complex literals like 1+2j, but we avoid importing ast here;
            # implement a light parser: strip brackets and split by comma
            if s.startswith('[') and s.endswith(']'):
                inner = s[1:-1]
                parts = [p.strip() for p in inner.split(',') if p.strip()]
                vals = []
                for p in parts:
                    # handle possible complex like "0.5+0.5j"
                    try:
                        vals.append(complex(p))
                    except Exception:
                        # last resort: float
                        vals.append(float(p))
                return np.asarray(vals, dtype=np.complex128)
            else:
                # single scalar
                try:
                    return np.asarray([complex(s)], dtype=np.complex128)
                except Exception:
                    return np.asarray([float(s)], dtype=np.complex128)
        except Exception:
            pass
    # Fallback: best-effort
    return np.atleast_1d(np.asarray(x, dtype=np.complex128))

def _normalize(vec: Union[np.ndarray, Sequence, str], target_dim: int = None) -> np.ndarray:
    """
    Coerce to complex array, optionally coerce to a specific dimension:
    - If target_dim==4 and len(vec)==2, auto-tensor with |0> (vec ⊗ |0>).
    - If target_dim==2 and len(vec)==4, attempt partial trace to qubit by dropping last qubit (simple heuristic).
    Always renormalizes. Falls back to DEFAULT_QUBIT on zero norm.
    """
    arr = _as_complex_array(vec).ravel()
    if target_dim is not None:
        if target_dim == 4 and arr.size == 2:
            arr = np.kron(arr, DEFAULT_QUBIT)
        elif target_dim == 2 and arr.size == 4:
            # simple reduce: sum over last-qubit basis states (|00>+|10>) -> two amplitudes
            arr = np.array([arr[0] + arr[2], arr[1] + arr[3]], dtype=np.complex128)
        # other sizes: keep as is
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm < EPS:
        return DEFAULT_QUBIT.copy()
    return (arr / norm).astype(np.complex128, copy=False)

class HologramEngine:
    def __init__(self, core=None):
        self.core = core

    def phi_calculation(self, state) -> float:
        psi = _normalize(state)
        # Simple proxy: single-qubit entropy if 2 amps; if 4 amps, entropy of marginal of first qubit
        if psi.size == 2:
            p0 = abs(psi[0])**2
            p1 = abs(psi[1])**2
            ps = np.clip([p0, p1], EPS, 1.0)
            H = float(-(ps[0]*np.log2(ps[0]) + ps[1]*np.log2(ps[1])))
            return H
        elif psi.size == 4:
            # marginalize last qubit
            a00,a01,a10,a11 = psi
            p0 = abs(a00)**2 + abs(a01)**2
            p1 = abs(a10)**2 + abs(a11)**2
            ps = np.clip([p0, p1], EPS, 1.0)
            H = float(-(ps[0]*np.log2(ps[0]) + ps[1]*np.log2(ps[1])))
            return H
        return 0.0

    def temporal_entangler(self, state_t1, state_t2):
        s1 = _normalize(state_t1, target_dim=2)
        s2 = _normalize(state_t2, target_dim=2)
        ent = np.kron(s1, s2)
        return _normalize(ent, target_dim=4)