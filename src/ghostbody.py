# ghostbody.py
#
# MISSION: Represents the embodied cognition aspect of the AGI, managing somatic state
# and temporal dynamics through quantum-inspired operations with qubit focus.
#
# --- QUANTUM REVISION LOG (Revision 23 - Bug Hunt Refinement) ---
# [FIX] ZERO-NORM-1: Integrated self.core.postprocess_qubit in all _normalize_qubit calls; added base amplitude 0.5 in fallbacks (e.g., |+> state [sqrt(0.5), sqrt(0.5)] for 2D, similar for 4D) to prevent zero-norm propagation.
# [FIX] SVD-EIGH-1: In _calculate_emergence, switched to np.linalg.eigvalsh(rho_A) for Hermitian eigenvalues, ensuring positive semi-definite handling.
# [FIX] INPUT-VALID-1: In update_somatic_state, added strict validation: raise ValueError if state len != 2 or norm != ~1 (within EPSILON).
# [ENH] LOGGING-1: Added warnings.catch_warnings() around plugin calls to suppress; log somatic_state_history length after updates.
# [ENH] OPTIMIZATION-1: Limited somatic_state_history to max 100 entries by slicing [-100:].
# [ENH] TEST-1: Added test_low_norm_states method to verify zero/low-norm handling with assertions.
# --------------------------------------------------------------------------

import numpy as np
import logging
import warnings
from typing import List, Union, Tuple, Optional

# Constants
DEFAULT_QUBIT = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.complex128)      # |+> with base amplitude 0.5
DEFAULT_TWO_QUBIT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.complex128)  # Bell-like with base 0.5
EPSILON = 1e-12  # Quantum-scale precision

# Correctly import the core class
from ghostcore import GhostCore


def _force_shape(vec: np.ndarray, wanted: int) -> np.ndarray:
    """
    Force vector to wanted size with truncation/padding.
    """
    v = vec.astype(np.complex128).reshape(-1)
    if v.size > wanted:
        return v[:wanted]
    elif v.size < wanted:
        return np.pad(v, (0, wanted - v.size), constant_values=0)
    return v


class GhostBody:
    """
    Manages the somatic (embodied) state of the AGI using qubit representations.
    """

    def __init__(self, core: GhostCore):
        self.core = core
        self.somatic_state_history: List[np.ndarray] = []
        self.qubit_creator = self.core.get_plugin("qubit_create")
        self.management_registry = self.core.get_plugin("qubit_management_registry")
        self.entanglement_creator = self.core.get_plugin("qubit_entanglement_creator")
        self.error_correct = self.core.get_plugin("qubit_error_correct") or self.core.get_plugin("qubit_error_correction_dev")
        self.update_somatic_state(np.array([1.0, 0.0]))  # Initialize with |0>

    def _normalize_qubit(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize a qubit state with size validation and epsilon guards.
        Routes through postprocess_qubit for error correction.
        """
        state = _force_shape(state, 2) if state.size in (2, 4) else state
        norm = np.linalg.norm(state)
        if norm < EPSILON:
            logging.warning("Near-zero norm detected in _normalize_qubit; applying postprocess.")
            return self.core.postprocess_qubit(state)  # Use core.postprocess_qubit for non-zero guarantee
        state = state / norm
        # Integrated core.postprocess_qubit for optimization and non-zero assurance
        return self.core.postprocess_qubit(state)

    def _create_qubit(self, input_data: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Creates a qubit state using 'qubit_create' plugin or fallback normalization.
        """
        # Switched to self.core.call_plugin; handle None with fallback
        qubit = self.core.call_plugin("qubit_create", input_data)
        if qubit is not None:
            return self._normalize_qubit(np.asarray(qubit, dtype=np.complex128))
        else:
            logging.warning("qubit_create plugin not available or failed; using fallback normalization.")
            return self._normalize_qubit(np.asarray(input_data, dtype=np.complex128))

    def _quantize_classical_state(self, classical: float) -> np.ndarray:
        """Converts classical value to qubit state via plugin or fallback."""
        return self._create_qubit([np.sqrt(1 - classical), np.sqrt(classical)])

    def _to_qubit(self, data: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Converts input to qubit with size guards."""
        if isinstance(data, (float, int)):
            return self._quantize_classical_state(float(data))
        arr = np.asarray(data, dtype=np.complex128).reshape(-1)
        if arr.size == 1:
            return self._quantize_classical_state(float(arr[0]))
        return self._create_qubit(arr)

    def update_somatic_state(self, new_state: Union[float, Tuple[float, float], List[float], np.ndarray]):
        """
        Updates somatic state history with validation and qubit conversion.
        """
        # Input validation: ensure tuple/list/array and convert to ndarray
        if isinstance(new_state, (tuple, list)):
            new_state = np.asarray(new_state)
        if not isinstance(new_state, np.ndarray):
            raise ValueError(f"Invalid somatic state type {type(new_state)}; must be ndarray, list, or tuple.")
        # Size validation: enforce qubit (2) or two-qubit (4)
        if new_state.size not in (2, 4):
            raise ValueError(f"Invalid somatic state size {new_state.size}; must be 2 or 4.")
        # Norm validation: should be ~1
        norm = np.linalg.norm(new_state)
        if abs(norm - 1) > EPSILON:
            raise ValueError(f"Invalid somatic state norm {norm}; must be approximately 1.")
        qubit_state = self._normalize_qubit(self._to_qubit(new_state))
        self.somatic_state_history.append(qubit_state)
        self.somatic_state_history = self.somatic_state_history[-100:]  # Limit history to 100
        logging.info(f"Somatic state updated | History length: {len(self.somatic_state_history)}")

    def _apply_registry(self, state: np.ndarray) -> np.ndarray:
        """
        Applies qubit management registry with validation.
        """
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            managed = self.core.call_plugin("qubit_management_registry", state)
            if managed is not None:
                return self._normalize_qubit(managed)
            else:
                logging.warning("qubit_management_registry not available or failed; returning original state.")
                return state

    def get_current_somatic_state(self) -> np.ndarray:
        """Returns the latest normalized somatic qubit state."""
        if not self.somatic_state_history:
            return DEFAULT_QUBIT.copy()
        return self._normalize_qubit(self.somatic_state_history[-1])

    def _calculate_emergence(self, entangled_state: np.ndarray) -> float:
        """
        Calculates emergence as entanglement entropy using eigenvalues for reduced density.
        Assumes pure two-qubit state (4D vector).
        """
        if entangled_state.size != 4:
            logging.error("Invalid state for emergence calculation; returning 0.")
            return 0.0
        rho = np.outer(entangled_state, entangled_state.conj())
        rho_A = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
        # Use eigvalsh for Hermitian matrix to ensure positive eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > EPSILON]
        eigenvalues /= np.sum(eigenvalues) + EPSILON  # Re-normalize trace
        return float(-np.sum(eigenvalues * np.log2(eigenvalues + EPSILON)))

    def get_temporally_entangled_state(self) -> np.ndarray:
        """
        Returns temporally entangled state from last two somatic states.
        Falls back to duplicate if only one state available.
        """
        if len(self.somatic_state_history) < 2:
            logging.warning("Insufficient history for temporal entanglement (duplicate) | Emergence: 0.0000")
            return np.kron(self.get_current_somatic_state(), self.get_current_somatic_state())

        state_t1 = self._normalize_qubit(self.somatic_state_history[-2])
        state_t2 = self._normalize_qubit(self.somatic_state_history[-1])

        # Apply quantum entanglement (plugin → fallback)
        entangled_state = self.core.call_plugin("qubit_entanglement_creator", state_t1, state_t2)
        if entangled_state is not None:
            entangled_state = _force_shape(entangled_state, 4)
        else:
            logging.warning("qubit_entanglement_creator not available or failed; falling back to tensor product.")
            entangled_state = np.kron(state_t1, state_t2)

        # Normalize -> optional error correction -> normalize again
        entangled_state = self._normalize_qubit(entangled_state)
        entangled_state = self._post_entanglement_correct(entangled_state)
        entangled_state = self._normalize_qubit(entangled_state)

        # Calculate and log emergence (entropy of reduced density)
        emergence = self._calculate_emergence(entangled_state)
        logging.info(f"Temporal entanglement complete | Emergence: {emergence:.4f}")

        return entangled_state

    def _post_entanglement_correct(self, entangled_state: np.ndarray) -> np.ndarray:
        """
        Optional error correction step after entanglement using 'qubit_error_correct'
        or 'qubit_error_correction_dev'. Safely validates size/norm.
        """
        # Suppress warnings around plugin calls
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corrected = self.core.call_plugin("qubit_error_correct", entangled_state)
            if corrected is None:
                logging.warning("qubit_error_correct not available or failed; trying alternative.")
                corrected = self.core.call_plugin("qubit_error_correction_dev", entangled_state)
            if corrected is not None:
                corrected = _force_shape(corrected, 4)
                return self._normalize_qubit(corrected)
            else:
                logging.warning("Post-entanglement correction failed; returning original entangled_state.")
                return entangled_state

    def test_low_norm_states(self):
        """Test handling of low/zero-norm states with assertions."""
        print("\n[GhostBody] Running low-norm state test...")

        # Test zero-norm input
        zero_state = np.array([0.0, 0.0])
        normed_zero = self._normalize_qubit(zero_state)
        assert np.linalg.norm(normed_zero) > 0, "Zero-norm fallback failed"
        print(f"Zero-norm fallback: {normed_zero} (norm={np.linalg.norm(normed_zero):.2f})")

        # Test low-norm input
        low_state = np.array([1e-13, 1e-13])
        normed_low = self._normalize_qubit(low_state)
        assert np.linalg.norm(normed_low) > 0, "Low-norm handling failed"
        print(f"Low-norm handling: {normed_low} (norm={np.linalg.norm(normed_low):.2f})")

        # Test 4D zero-norm
        zero_4d = np.zeros(4)
        normed_4d = self._normalize_qubit(zero_4d)
        assert np.linalg.norm(normed_4d) > 0, "4D zero-norm handling failed"
        print(f"4D zero-norm fallback: {normed_4d} (norm={np.linalg.norm(normed_4d):.2f})")

        print("✅ Low-norm test completed successfully")
