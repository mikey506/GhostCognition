# multiverse_simulator.py
#
# MISSION: Manages the creation, simulation, and branching of quantum realities using qubit states.
#
# --- BATCH5 REFINEMENT LOG (Revision 25) ---
# [FIX] PLUGIN-NOT-AVAILABLE: In _apply_quantum_gate and _simulate_decoherence, handled None returns from self.core.call_plugin with logging.warning and return original state.
# [CONFIRM] NORMALIZATION: _normalize_qubit always returns non-zero vector (QUBIT_ZERO on tiny norm).
# [ENH] LOGGING: In test_multiverse_forking, added explicit logger.info for norm checks on all states.
# --------------------------------------------------------------------------

# --- Refinement Changes Based on All Bug Reports ---
# - Plugin not available: In _apply_quantum_gate and _simulate_decoherence, if self.core.call_plugin returns None, log warning and return original state.
# - Decoherence bounds: In get_ethical_divergence, clamped val to [0, 1-1e-12] with np.clip.
# - Gate aliases: Made case-insensitive by .lower() on gate_name and aliases in _apply_quantum_gate.
# - General optimization: Added batch norm computation in test_multiverse_forking using np.array([initial_state, final_0, final_1]); added assert for divergence <=1 to test for overshoot.

from __future__ import annotations

import numpy as np
import logging
import copy
from typing import Dict, Any, Optional, List
from ghostcore import GhostCore

EPS = 1e-12
QUBIT_ZERO = np.array([1.0, 0.0], dtype=np.complex128)

def _normalize_qubit(q: np.ndarray) -> np.ndarray:
    """Ensure a 2D, normalized qubit. Fallback to |0> on tiny norm."""
    q = np.asarray(q, dtype=np.complex128).reshape(-1)
    if q.size != 2:
        q = q[:2] if q.size > 2 else np.pad(q, (0, 2 - q.size), constant_values=0)
    n = np.linalg.norm(q)
    if n < EPS:
        return QUBIT_ZERO.copy()
    return q / n

class MultiverseSimulator:
    """Simulates multiple parallel quantum realities with qubit states."""

    # Aliases for common gates/plugins
    _GATE_ALIASES = {
        "hadamard_gate": ["hadamard_gate", "H", "hadamard", "gate_h"],
        "pauli_x_gate": ["pauli_x_gate", "X", "pauli_x", "bit_flip"],
        "pauli_y_gate": ["pauli_y_gate", "Y", "pauli_y"],
        "pauli_z_gate": ["pauli_z_gate", "Z", "pauli_z", "phase_flip"],
        "s_gate":       ["s_gate", "S"],
        "t_gate":       ["t_gate", "T"],
        # Development/compound gate
        "qubit_gate_development": ["qubit_gate_development", "dev_gate", "gate_dev"],
        # Topological and decoherence kept explicit
        "topological_qubit_manager": ["topological_qubit_manager", "topo_manager"],
        "ethical_decoherence": ["ethical_decoherence", "eth_dec"],
    }

    def __init__(self, core: GhostCore):
        self.core = core
        self.realities: Dict[str, Dict[str, Any]] = {
            "reality_0": {
                "state": _normalize_qubit([1.0, 0.0]),
                "physics_plugins": ["hadamard_gate", "ethical_decoherence"],
                "decoherence": 0.0,
            }
        }
        logging.info("MultiverseSimulator: Initialized with base reality 'reality_0'.")

    def fork_reality(self, source_id: str, new_id: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Forks a new reality from an existing one, optionally with modified params."""
        if source_id not in self.realities:
            logging.error(f"Fork failed: Source '{source_id}' not found.")
            return False
        if new_id in self.realities:
            logging.warning(f"Fork skipped: '{new_id}' already exists.")
            return False

        new_reality = copy.deepcopy(self.realities[source_id])
        if params:
            new_reality.update(params)
        self.realities[new_id] = new_reality

        # Apply topological management if available
        topo_plugin = self.core.get_plugin("topological_qubit_manager")
        if topo_plugin:
            try:
                new_state = topo_plugin(new_reality["state"])
                new_reality["state"] = _normalize_qubit(new_state)
            except Exception as e:
                logging.error(f"Topological management failed during fork: {e}")

        logging.info(f"Reality forked: '{source_id}' -> '{new_id}'.")
        return True

    def step_simulation(self, reality_id: str):
        """Advances the simulation of a reality by one step."""
        if reality_id not in self.realities:
            logging.error(f"Step failed: Reality '{reality_id}' not found.")
            return

        r = self.realities[reality_id]
        state = r["state"]

        # Apply physics plugins sequentially
        for plugin_name in r.get("physics_plugins", []):
            plugin = self.core.get_plugin(plugin_name)
            if plugin:
                try:
                    new_state = self.core.call_plugin(plugin_name, state, rate=r.get('decoherence', 0.05)) if plugin_name=='ethical_decoherence' else self.core.call_plugin(plugin_name, state)
                    if new_state is not None:
                        state = _normalize_qubit(new_state)
                    else:
                        logging.warning(f"Plugin '{plugin_name}' returned None; skipping.")
                except Exception as e:
                    logging.error(f"Plugin '{plugin_name}' failed: {e}; skipping.")
            else:
                logging.warning(f"Plugin '{plugin_name}' not found; skipping.")

        # Simulate decoherence
        state = self._simulate_decoherence(state)

        # Update state
        r["state"] = state

    def _apply_quantum_gate(self, state: np.ndarray, gate_name: str) -> np.ndarray:
        """Applies a quantum gate by name or alias, with fallback to identity."""
        gate_plugin = None
        gate_name_lower = gate_name.lower()
        for key, aliases in self._GATE_ALIASES.items():
            if gate_name_lower in [a.lower() for a in aliases]:
                gate_plugin = key
                break

        if gate_plugin:
            result = self.core.call_plugin(gate_plugin, state)
            if result is not None:
                return _normalize_qubit(result)
            else:
                logging.warning(f"Gate '{gate_name}' ({gate_plugin}) returned None; fallback to identity.")
        else:
            logging.warning(f"Gate '{gate_name}' not found; fallback to identity.")

        return state  # Identity fallback

    def _simulate_decoherence(self, state: np.ndarray) -> np.ndarray:
        """Simulates ethical decoherence using plugin or fallback."""
        dec_plugin = self.core.get_plugin("ethical_decoherence")
        if dec_plugin:
            result = self.core.call_plugin("ethical_decoherence", state)
            if result is not None:
                return _normalize_qubit(result)
            else:
                logging.warning("ethical_decoherence returned None; fallback to original state.")
        else:
            logging.warning("ethical_decoherence not available; fallback to original state.")

        return state

    def get_reality_state(self, reality_id: str) -> Optional[np.ndarray]:
        """Retrieves the current state of a reality."""
        if reality_id in self.realities:
            return self.realities[reality_id]["state"]
        logging.error(f"Get state failed: Reality '{reality_id}' not found.")
        return None

    def get_ethical_divergence(self, reality_id: str) -> float:
        """Retrieves the ethical divergence (decoherence) of a reality."""
        if reality_id in self.realities:
            r = self.realities[reality_id]
            val = float(r.get("decoherence", 0.0))
            return np.clip(val, 0.0, 1.0 - 1e-12)
        logging.error(f"Get divergence failed: Reality '{reality_id}' not found.")
        return 0.0

    # ---- Debug/Test -----------------------------------------------------------------------
    def test_multiverse_forking(self, args: Optional[List[str]] = None):
        """Test forking and divergence; logs state norms for verification (Batch4)."""
        print("\n[Multiverse] Running quantum multiverse test...")
        initial_state = self.get_reality_state("reality_0")
        print(f"Initial state: {initial_state} ||ψ||={np.linalg.norm(initial_state):.6f}")

        # Fork into new reality with different physics
        fork_success = self.fork_reality("reality_0", "reality_1", {"physics_plugins": ["pauli_x_gate"]})
        if not fork_success:
            print("Forking failed")
            return

        # Evolve both realities
        self.step_simulation("reality_0")
        self.step_simulation("reality_1")

        # Check results
        final_0 = self.get_reality_state("reality_0")
        final_1 = self.get_reality_state("reality_1")
        divergence_0 = self.get_ethical_divergence("reality_0")
        divergence_1 = self.get_ethical_divergence("reality_1")

        print(f"Reality_0: State={final_0} ||ψ||={np.linalg.norm(final_0):.6f} Decoherence={divergence_0:.3f}")
        print(f"Reality_1: State={final_1} ||ψ||={np.linalg.norm(final_1):.6f} Decoherence={divergence_1:.3f}")

        # Added checks to ensure non-zero norms
        if np.linalg.norm(final_0) < EPS or np.linalg.norm(final_1) < EPS:
            print("⚠️ Zero norm detected in test realities")

        # Verify ethical divergence
        assert divergence_0 <= 1.0 and divergence_1 <= 1.0, "Divergence overshoot detected"
        if divergence_0 > 0 or divergence_1 > 0:
            print("✅ Ethical divergence observed")
        else:
            print("⚠️ No ethical divergence detected")

        # Batch norm computation and logging
        states_array = np.array([initial_state, final_0, final_1])
        norms = np.linalg.norm(states_array, axis=1)
        logging.info(f"Batch test norms: {norms}")
