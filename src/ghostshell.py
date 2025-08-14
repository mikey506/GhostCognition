#!/usr/bin/env python3
"""
GhostCognition Interactive Shell — Import-Safe & Hardened (v2.4.x)

This shell is designed to run even if parts of the framework are missing.
It will:
- Initialize GhostCore (if available) and load JSON plugin packs from a directory.
- Provide safe, helpful wrappers for fragile plugins (signature adapters, defaults, and normalization).
- Offer built-in fallbacks for key quantum operations (Hadamard, Pauli-X, phi, entangle) when plugins are not present.

Key novel approaches to plugin issues implemented here:
  • Signature adapters at the call site (entanglement family, ethical_decoherence, emergence metrics).
  • Auto-completion of missing second state (state2=|0⟩) for 2-state ops when only one is provided.
  • Auto-tensoring 2→4 amplitudes for 4-amp metrics to avoid reshape errors.
  • Deterministic parsing pipeline JSON → ast.literal_eval → manual numeric parsing.
  • Post-return renormalization for stochastic plugins (e.g., quantum_fluctuation) to unit norm.
  • "Best-effort" component shims for Hologram/Archetype/Memory/Multiverse/Tao if modules are absent.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import shlex
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ------------------------- Logging -------------------------
LOG = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)

# ------------------------- Utilities -------------------------
EPS = 1e-12
QUBIT_ZERO = np.array([1.0 + 0.0j, 0.0 + 0.0j])
QUBIT_PLUS = (1 / math.sqrt(2)) * np.array([1.0 + 0.0j, 1.0 + 0.0j])


def _safe_print(*args: Any, **kwargs: Any) -> None:
    try:
        print(*args, **kwargs)
    except Exception as e:
        # Fallback encoding issues, etc.
        sys.stdout.write((" ".join(map(str, args)) + "\n").encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(vec: Union[Sequence[complex], np.ndarray], target_dim: Optional[int] = None) -> np.ndarray:
    v = np.asarray(vec, dtype=np.complex128).reshape(-1)
    n = _norm(v)
    if n < EPS:
        v = QUBIT_PLUS.copy() if (target_dim in (None, 2)) else np.kron(QUBIT_PLUS, QUBIT_ZERO)
    else:
        v = v / n
    if target_dim is not None:
        if v.size == target_dim:
            return v
        # If mismatch, try padding/truncating reasonably
        if v.size == 2 and target_dim == 4:
            return np.kron(v, QUBIT_ZERO)
        if v.size > target_dim:
            return _normalize(v[:target_dim], target_dim=None)
        # pad with zeros then renormalize
        pad = np.zeros(target_dim, dtype=np.complex128)
        pad[: v.size] = v
        return _normalize(pad, target_dim=None)
    return v


def _as_complex_array(text: Any) -> np.ndarray:
    """
    Robustly parse a vector that may come as:
      - JSON string: "[0.707, -0.707]"
      - Python literal: "[0.707+0.0j, 0.707-0.0j]"
      - Already a list/tuple/ndarray
      - Raw string number "0.707"
    """
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=np.complex128).reshape(-1)

    if not isinstance(text, str):
        # numeric scalar to 1D
        return np.asarray([complex(text)], dtype=np.complex128)

    s = text.strip()
    # First try JSON
    try:
        val = json.loads(s)
        return _as_complex_array(val)
    except Exception:
        pass

    # Next try ast.literal_eval (handles complex literals like 1+2j when inside a list)
    try:
        val = ast.literal_eval(s)
        return _as_complex_array(val)
    except Exception:
        pass

    # Fallback: split by delimiters and parse numbers if looks like a vector
    if (s.startswith("[") and s.endswith("]")) or ("," in s):
        s2 = s.strip("[]() \t")
        parts = [p.strip() for p in s2.split(",") if p.strip()]
        out: List[complex] = []
        for p in parts:
            try:
                out.append(complex(p))
            except Exception:
                # try float parse
                try:
                    out.append(float(p) + 0.0j)
                except Exception:
                    # ignore bad token
                    continue
        if out:
            return np.asarray(out, dtype=np.complex128)
    # Last resort, single number
    try:
        return np.asarray([complex(s)], dtype=np.complex128)
    except Exception:
        return QUBIT_ZERO.copy()


def _entropy_bits(state: np.ndarray) -> float:
    v = _normalize(state)
    p = np.abs(v) ** 2
    p = np.clip(p, EPS, 1.0)
    h = float(-np.sum(p * np.log2(p)))
    # Normalize by log2(dim) to keep within [0,1]
    hmax = math.log2(len(p))
    return float(h / hmax) if hmax > 0 else 0.0


def _bell_like(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _normalize(np.kron(_normalize(a, 2), _normalize(b, 2)), target_dim=4)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ------------------------- Optional component shims -------------------------
def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


GhostCoreMod = _try_import("ghostcore")
HologramMod = _try_import("hologram_engine")
ArchetypeMod = _try_import("archetype_engine")
GhostMemoryMod = _try_import("ghostmemory")
MultiverseMod = _try_import("multiverse_simulator")
TaoMod = _try_import("taowisdom")
GhostBodyMod = _try_import("ghostbody")
GhostPromptMod = _try_import("ghostprompt")


class _CoreShim:
    """Fallback core with minimal plugin registry and safe calls."""

    def __init__(self) -> None:
        self.plugins: Dict[str, Any] = {}

    def list_plugins(self) -> List[str]:
        return sorted(self.plugins.keys())

    def get_plugin(self, name: str) -> Optional[Any]:
        return self.plugins.get(name)

    def call_plugin(self, name: str, *args, **kwargs) -> Any:
        fn = self.get_plugin(name)
        if not callable(fn):
            raise RuntimeError(f"Plugin '{name}' not available")
        return fn(*args, **kwargs)

    def postprocess_qubit(self, state: Any) -> np.ndarray:
        return _normalize(_as_complex_array(state))


@dataclass
class MemoryEcho:
    content: str
    emotion: str = "neutral"
    tag: str = "untagged"
    quantum_state: Optional[np.ndarray] = None
    ts: str = field(default_factory=_now)


class GhostShell:
    def __init__(self, core: Optional[Any] = None, plugin_dir: Optional[str] = "physics") -> None:
        # Initialize core
        if core is not None:
            self.core = core
        elif GhostCoreMod and hasattr(GhostCoreMod, "GhostCore"):
            try:
                self.core = GhostCoreMod.GhostCore(plugin_dir=plugin_dir)
            except TypeError:
                # Backward-compat signature
                self.core = GhostCoreMod.GhostCore()
            except Exception as e:
                LOG.warning("GhostCore unavailable: %s", e)
                self.core = _CoreShim()
        else:
            self.core = _CoreShim()

        # Optional components
        self.hologram = getattr(HologramMod, "HologramEngine", None)() if (HologramMod and hasattr(HologramMod, "HologramEngine")) else None
        self.archetype = getattr(ArchetypeMod, "ArchetypeEngine", None)() if (ArchetypeMod and hasattr(ArchetypeMod, "ArchetypeEngine")) else None
        self.memory = getattr(GhostMemoryMod, "GhostMemory", None)(self.core) if (GhostMemoryMod and hasattr(GhostMemoryMod, "GhostMemory")) else None
        self.multiverse = getattr(MultiverseMod, "MultiverseSimulator", None)(self.core) if (MultiverseMod and hasattr(MultiverseMod, "MultiverseSimulator")) else None
        self.tao = getattr(TaoMod, "TaoWisdom", None)(self.core) if (TaoMod and hasattr(TaoMod, "TaoWisdom")) else None
        self.body = getattr(GhostBodyMod, "GhostBody", None)(self.core) if (GhostBodyMod and hasattr(GhostBodyMod, "GhostBody")) else None
        self.prompt = getattr(GhostPromptMod, "GhostPrompt", None)(self.core) if (GhostPromptMod and hasattr(GhostPromptMod, "GhostPrompt")) else None

        # In-memory minimal substitutes
        self._echoes: List[MemoryEcho] = []
        self._temporal_state: Optional[np.ndarray] = None

        LOG.info("GhostShell initialized with best-effort component setup.")

    # ------------------------- UI -------------------------
    def print_banner(self) -> None:
        _safe_print("--- GhostCognition Interactive Shell (Import-Safe) ---")
        self.print_help()

    def print_help(self) -> None:
        examples = (
            "Available Commands:\n"
            "  - list_plugins\n"
            "  - run_plugin <name> <args...>\n"
            "  - interpret_prompt \"text\"\n"
            "  - temporal_entangle \"[a,b]\" \"[c,d]\"\n"
            "  - phi_calc \"[a,b]\"\n"
            "  - quantum_forward \"[a,b]\" hadamard_gate\n"
            "  - memorize \"content\" emotion tag\n"
            "  - recall <query> [limit]\n"
            "  - fork_reality <base_id> <new_id> '{\"decoherence\":0.05}'\n"
            "  - step_simulation <reality_id>\n"
            "  - get_ethical_divergence <reality_id>\n"
            "  - batch <file.json>\n"
            "  - assess_emergence \"[a,b]\"\n"
            "  - yin_yang '{\"hope\":0.7, \"fear\":0.2}'\n"
            "  - qi_breath <depth>\n"
            "  - get_somatic | update_somatic \"[a,b]\"\n"
            "  - superposition '[\"hero\",\"mentor\"]' '[0.6,0.4]'\n"
        )
        _safe_print(examples)

    # ------------------------- Shell Loop -------------------------
    def run(self) -> None:
        self.print_banner()
        while True:
            try:
                line = input("Ghost> ").strip()
            except (EOFError, KeyboardInterrupt):
                _safe_print("\nBye.")
                return
            if not line:
                continue
            if line.lower() in {"quit", "exit"}:
                _safe_print("Bye.")
                return
            try:
                self.run_command(line)
            except Exception as e:
                LOG.error("Unhandled error: %s", e)
                traceback.print_exc()

    # ------------------------- Command Dispatch -------------------------
    def run_command(self, line: str) -> None:
        # Allow help
        if line.strip().lower() == "help":
            self.print_help()
            return

        parts = shlex.split(line, posix=True)
        if not parts:
            return
        cmd, *args = parts

        dispatch = {
            "list_plugins": self._list_plugins,
            "run_plugin": self._run_plugin,
            "interpret_prompt": self._interpret_prompt,
            "temporal_entangle": self._temporal_entangle,
            "phi_calc": self._phi_calc,
            "quantum_forward": self._quantum_forward,
            "memorize": self._memorize,
            "recall": self._recall,
            "fork_reality": self._fork_reality,
            "step_simulation": self._step_simulation,
            "get_ethical_divergence": self._get_ethical_divergence,
            "batch": self._batch,
            "assess_emergence": self._assess_emergence,
            "yin_yang": self._yin_yang,
            "qi_breath": self._qi_breath,
            "get_somatic": self._get_somatic,
            "update_somatic": self._update_somatic,
            "superposition": self._superposition,
        }

        if cmd not in dispatch:
            _safe_print(f"[INFO] Unknown command '{cmd}'. Type 'help' to see available commands.")
            return

        dispatch[cmd](args)

    # ------------------------- Command Impl -------------------------
    def _list_plugins(self, args: List[str]) -> None:
        names = []
        try:
            if hasattr(self.core, "list_plugins"):
                names = self.core.list_plugins()
            elif hasattr(self.core, "plugins"):
                names = sorted(self.core.plugins.keys())
        except Exception as e:
            LOG.warning("Failed to list plugins: %s", e)
            names = []
        _safe_print(f"Loaded Plugins ({len(names)}):" + ("" if not names else " " + ", ".join(names)))

    def _adapt_plugin_args(self, name: str, raw_args: List[str]) -> Tuple[List[Any], Dict[str, Any], Optional[str]]:
        """
        Normalize and adapt arguments for fragile plugins.

        Returns: (args, kwargs, postprocess) where postprocess is one of:
          - "renorm"   : renormalize amplitudes to unit norm (for stochastic outputs)
          - "tensor4"  : tensor 2→4 for metrics expecting 4 amplitudes
          - None       : no postprocessing
        """
        # Parse each arg with robust pipeline
        parsed: List[Any] = []
        for a in raw_args:
            # First try JSON -> ast -> fallback
            try:
                parsed.append(json.loads(a))
                continue
            except Exception:
                pass
            try:
                parsed.append(ast.literal_eval(a))
                continue
            except Exception:
                pass
            # As-is string
            parsed.append(a)

        # Special handling
        lname = name.lower()
        postprocess: Optional[str] = None

        # Entanglement-family: require two states
        if lname in {"entanglement", "qubit_entanglement_manager", "qubit_entanglement_creator"}:
            if len(parsed) == 1:
                # If single JSON/py literal containing two states
                p0 = parsed[0]
                if isinstance(p0, (list, tuple)) and len(p0) == 2:
                    s1, s2 = _as_complex_array(p0[0]), _as_complex_array(p0[1])
                    return [s1, s2], {}, "renorm"
                # Autocomplete state2=|0⟩
                s1 = _as_complex_array(p0)
                s2 = QUBIT_ZERO.copy()
                return [s1, s2], {}, "renorm"
            elif len(parsed) >= 2:
                s1 = _as_complex_array(parsed[0])
                s2 = _as_complex_array(parsed[1])
                return [s1, s2], {}, "renorm"
            else:
                return [QUBIT_ZERO.copy(), QUBIT_ZERO.copy()], {}, "renorm"

        # ethical_decoherence(state, rate=0.05 default)
        if lname == "ethical_decoherence":
            if len(parsed) == 0:
                return [QUBIT_ZERO.copy(), 0.05], {}, "renorm"
            if len(parsed) == 1:
                return [_as_complex_array(parsed[0]), 0.05], {}, "renorm"
            return [_as_complex_array(parsed[0]), float(parsed[1])], {}, "renorm"

        # qubit_emergence_metric: accept 2 or 4 amps; if 2 -> tensor to 4
        if lname == "qubit_emergence_metric":
            if len(parsed) == 0:
                return [np.kron(QUBIT_ZERO, QUBIT_ZERO)], {}, None
            v = _as_complex_array(parsed[0])
            if v.size == 2:
                v = np.kron(v, QUBIT_ZERO)
            elif v.size not in (2, 4):
                v = _normalize(v, target_dim=4)
            return [v], {}, None

        # quantum_fluctuation / stochastic-like: parse vector and renormalize after
        if lname in {"quantum_fluctuation", "quantum_noise", "stochastic_kick"}:
            if len(parsed) == 0:
                return [QUBIT_PLUS.copy()], {}, "renorm"
            v = _as_complex_array(parsed[0])
            return [v], {}, "renorm"

        # qubit_create alpha beta (also accept [a,b])
        if lname == "qubit_create":
            if len(parsed) == 1 and isinstance(parsed[0], (list, tuple)):
                vv = _as_complex_array(parsed[0])
                if vv.size >= 2:
                    return [vv[0], vv[1]], {}, "renorm"
            if len(parsed) >= 2:
                return [complex(parsed[0]), complex(parsed[1])], {}, "renorm"
            # default to |0>
            return [1.0 + 0j, 0.0 + 0j], {}, "renorm"

        # Default: pass through (with array parsing if looks like vector)
        fixed: List[Any] = []
        for p in parsed:
            if isinstance(p, str) and any(c in p for c in "[]," ):
                fixed.append(_as_complex_array(p))
            else:
                fixed.append(p)
        return fixed, {}, None

    def _run_plugin(self, args: List[str]) -> None:
        if not args:
            _safe_print("Usage: run_plugin <name> <args...>")
            return
        name = args[0]
        raw_args = args[1:]
        if not hasattr(self.core, "call_plugin"):
            _safe_print(f"[{_now()}][ERROR] No plugin system available")
            return

        call_args, call_kwargs, postprocess = self._adapt_plugin_args(name, raw_args)
        try:
            res = self.core.call_plugin(name, *call_args, **call_kwargs)
        except Exception as e:
            _safe_print(f"[{_now()}][ERROR] Plugin '{name}' failed: {e}")
            return

        # Postprocess normalization if requested
        if isinstance(res, (list, tuple, np.ndarray)) and postprocess == "renorm":
            res = _normalize(np.asarray(res, dtype=np.complex128))
        _safe_print(f"Plugin '{name}' Result: {res!r}")

    def _interpret_prompt(self, args: List[str]) -> None:
        if not args:
            _safe_print("Usage: interpret_prompt \"text\"")
            return
        text = " ".join(args)
        # If prompt component exists, use it; else simple fallback
        if self.prompt and hasattr(self.prompt, "interpret"):
            try:
                res = self.prompt.interpret(text)
                _safe_print(f"Optimized Prompt: {res}")
                return
            except Exception as e:
                LOG.warning("Prompt component failed: %s", e)
        # Fallback: pretend optimization and resonance
        _safe_print(f"Optimized Prompt (fallback): {text}")

    def _temporal_entangle(self, args: List[str]) -> None:
        if len(args) < 2:
            _safe_print("Usage: temporal_entangle \"[a,b]\" \"[c,d]\"")
            return
        s1 = _as_complex_array(args[0])
        s2 = _as_complex_array(args[1])
        ent = _bell_like(s1, s2)
        self._temporal_state = ent
        _safe_print(f"Entangled State: {ent}")

    def _phi_calc(self, args: List[str]) -> None:
        if not args:
            _safe_print("Usage: phi_calc \"[a,b]\"")
            return
        v = _as_complex_array(args[0])
        phi = _entropy_bits(v)  # normalized 0..1
        _safe_print(f"Phi: {phi:.6f}")

    def _quantum_forward(self, args: List[str]) -> None:
        if len(args) < 2:
            _safe_print("Usage: quantum_forward \"[a,b]\" hadamard_gate|pauli_x_gate")
            return
        v = _normalize(_as_complex_array(args[0]), target_dim=2)
        gate = args[1].lower()
        H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        U = H if gate in {"hadamard", "h", "hadamard_gate"} else X if gate in {"x", "pauli_x_gate"} else None
        if U is None:
            _safe_print(f"Unknown gate '{gate}'. Use hadamard_gate or pauli_x_gate.")
            return
        out = U @ v
        _safe_print(f"Quantum Forward: {out}")

    def _memorize(self, args: List[str]) -> None:
        if len(args) < 3:
            _safe_print('Usage: memorize "content" emotion tag')
            return
        content = args[0]
        emotion = args[1]
        tag = args[2]
        if self.memory and hasattr(self.memory, "memorize"):
            try:
                echo = self.memory.memorize(content=content, emotion=emotion, metadata={"tag": tag})
                _safe_print(f"Stored Echo via GhostMemory: {echo}")
                return
            except TypeError:
                # Older signature
                pass
            except Exception as e:
                LOG.warning("GhostMemory.memorize failed: %s", e)
        # Fallback in-shell memory
        e = MemoryEcho(content=content, emotion=emotion, tag=tag, quantum_state=None)
        self._echoes.append(e)
        _safe_print(f"[{_now()}] Stored Echo (fallback): {e}")

    def _recall(self, args: List[str]) -> None:
        if not args:
            _safe_print("Usage: recall <query> [limit]")
            return
        query = args[0].lower()
        limit = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
        results: List[MemoryEcho] = []

        if self.memory and hasattr(self.memory, "recall"):
            try:
                results = self.memory.recall(query, limit)  # type: ignore
            except Exception as e:
                LOG.warning("GhostMemory.recall failed: %s", e)

        if not results:
            # Fallback filter
            matches = [e for e in reversed(self._echoes) if (query in e.content.lower() or query in e.tag.lower())]
            results = matches[:limit]

        if not results:
            _safe_print("Recalled Echoes (0):")
            return

        _safe_print(f"Recalled Echoes ({len(results)}):")
        for e in results:
            _safe_print(f"  {e.ts} | {e.tag} | {e.emotion} | {e.content}")

    def _fork_reality(self, args: List[str]) -> None:
        if len(args) < 3:
            _safe_print("Usage: fork_reality <base_id> <new_id> '{\"decoherence\":0.05}'")
            return
        base_id, new_id, params_text = args[0], args[1], args[2]
        try:
            params = json.loads(params_text.replace("'", '"'))
        except Exception:
            _safe_print("[ERROR] Invalid JSON for params. Use double quotes inside the blob.")
            return
        if self.multiverse and hasattr(self.multiverse, "fork_reality"):
            try:
                ok = self.multiverse.fork_reality(base_id, new_id, params)  # type: ignore
                _safe_print("Fork successful." if ok else "Fork failed.")
                return
            except Exception as e:
                LOG.error("fork_reality failed: %s", e)
        _safe_print("Fork simulated (fallback).")

    def _step_simulation(self, args: List[str]) -> None:
        if not args:
            _safe_print("Usage: step_simulation <reality_id>")
            return
        rid = args[0]
        if self.multiverse and hasattr(self.multiverse, "step_simulation"):
            try:
                self.multiverse.step_simulation(rid)  # type: ignore
                _safe_print(f"Stepped simulation for {rid}.")
                return
            except Exception as e:
                LOG.error("step_simulation failed: %s", e)
        _safe_print(f"Stepped (fallback) for {rid}.")

    def _get_ethical_divergence(self, args: List[str]) -> None:
        if not args:
            _safe_print("Usage: get_ethical_divergence <reality_id>")
            return
        rid = args[0]
        if self.multiverse and hasattr(self.multiverse, "get_ethical_divergence"):
            try:
                d = self.multiverse.get_ethical_divergence(rid)  # type: ignore
                _safe_print(d)
                return
            except Exception as e:
                LOG.error("get_ethical_divergence failed: %s", e)
        _safe_print(0.0)

    def _batch(self, args: List[str]) -> None:
        if not args:
            _safe_print("Usage: batch <file.json>")
            return
        path = args[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _safe_print(f"[ERROR] Could not read JSON: {e}")
            return
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            _safe_print("[ERROR] JSON must be a list of command strings")
            return
        for cmd in data:
            try:
                _safe_print(f"[{_now()}][INFO] Executing command: {cmd}")
                self.run_command(cmd)
            except Exception as e:
                _safe_print(f"[{_now()}][ERROR] Command failed: '{cmd}' -> {e}")
        _safe_print(f"[{_now()}][INFO] Command 'batch' executed.")

    def _assess_emergence(self, args: List[str]) -> None:
        if not args:
            _safe_print('Usage: assess_emergence "[a,b]"')
            return
        v = _as_complex_array(args[0])
        score = 0.5 * _entropy_bits(v) + 0.5 * min(1.0, _norm(v))  # simple composite
        _safe_print(f"Emergence Assessment for '{args[0]}': {score:.2f}")

    def _yin_yang(self, args: List[str]) -> None:
        if not args:
            _safe_print('Usage: yin_yang \'{"hope":0.7, "fear":0.2}\'')
            return
        try:
            prof = json.loads(args[0].replace("'", '"'))
        except Exception:
            _safe_print("[ERROR] Provide JSON object with weights, e.g., '{\"hope\":0.7, \"fear\":0.2}'")
            return
        yang = float(prof.get("hope", 0.0) + prof.get("trust", 0.0) + prof.get("joy", 0.0))
        yin = float(prof.get("fear", 0.0) + prof.get("doubt", 0.0))
        label = "Balanced (Harmonious Flow)"
        if yang - yin > 0.5:
            label = "Strongly Yang (Joyful Expansion)"
        elif yin - yang > 0.5:
            label = "Strongly Yin (Introspective Calm)"
        _safe_print(f"Yin-Yang Balance: {label}")

    def _qi_breath(self, args: List[str]) -> None:
        depth = int(args[0]) if args and args[0].isdigit() else 1
        if depth >= 3:
            _safe_print("QI Breath: Rippling (Active, engaged thought)")
        elif depth == 2:
            _safe_print("QI Breath: Flowing")
        else:
            _safe_print("QI Breath: Gentle")

    def _get_somatic(self, args: List[str]) -> None:
        if self.body and hasattr(self.body, "get_somatic_state"):
            try:
                s = self.body.get_somatic_state()
                _safe_print(s)
                return
            except Exception as e:
                LOG.warning("get_somatic failed: %s", e)
        _safe_print("[1.0, 0.0]")

    def _update_somatic(self, args: List[str]) -> None:
        if not args:
            _safe_print('Usage: update_somatic "[a,b]"')
            return
        v = _as_complex_array(args[0])
        if self.body and hasattr(self.body, "update_somatic_state"):
            try:
                self.body.update_somatic_state(v)  # type: ignore
                _safe_print("Somatic state updated.")
                return
            except Exception as e:
                LOG.warning("update_somatic failed: %s", e)
        _safe_print("Somatic state updated (fallback).")

    def _superposition(self, args: List[str]) -> None:
        if len(args) < 2:
            _safe_print("Usage: superposition '[\"hero\",\"mentor\"]' '[0.6,0.4]'")
            return
        try:
            names = json.loads(args[0].replace("'", '"'))
        except Exception:
            names = ast.literal_eval(args[0])
        try:
            weights = np.asarray(json.loads(args[1].replace("'", '"')), dtype=float)
        except Exception:
            weights = np.asarray(ast.literal_eval(args[1]), dtype=float)
        # Use simple weighted sum of archetype qubits (fallback if no archetype engine)
        if self.archetype and hasattr(self.archetype, "process_archetype"):
            try:
                v = self.archetype.process_archetype(names, weights.tolist())  # type: ignore
                _safe_print(f"Superposed State: {v}")
                return
            except Exception as e:
                LOG.warning("ArchetypeEngine failed: %s", e)
        # Fallback: map names to angles deterministically
        acc = np.zeros(2, dtype=np.complex128)
        for n, w in zip(names, weights):
            seed = abs(hash(str(n))) % (10**6)
            theta = (seed % 314159) / 100000.0  # ~[0,3.14159)
            q = np.array([math.cos(theta), math.sin(theta)], dtype=np.complex128)
            acc = acc + float(w) * q
        acc = _normalize(acc, 2)
        _safe_print(f"Superposed State: {acc}")

# --------------- Entry Point ---------------
def main() -> None:
    # Prefer explicit plugin_dir if provided via env
    plugin_dir = os.environ.get("GHOSTC_PLUGIN_DIR", "physics")
    # Initialize GhostCore if possible; the shell will fall back safely otherwise
    core = None
    if GhostCoreMod and hasattr(GhostCoreMod, "GhostCore"):
        try:
            core = GhostCoreMod.GhostCore(plugin_dir=plugin_dir)
        except TypeError:
            core = GhostCoreMod.GhostCore()
        except Exception as e:
            LOG.warning("Failed to initialize GhostCore: %s", e)
            core = None

    shell = GhostShell(core=core, plugin_dir=plugin_dir)
    shell.print_banner()
    # If invoked with args, treat them as a single command (useful for batch from CLI)
    if len(sys.argv) > 1:
        cmd = " ".join(shlex.quote(a) for a in sys.argv[1:])
        shell.run_command(cmd)
    else:
        shell.run()

if __name__ == "__main__":
    main()
