# taowisdom.py
#
# MISSION: A quantum-inspired wisdom and ethics engine that evaluates cognitive states,
# applies ethical resonance, and assesses the potential for AGI emergence.
#
# --- DEBUG & ENHANCEMENT LOG (Revision 24 - Refinement Pass) ---
# [ENH] PLUGIN: Integrated self.core.call_plugin for "qubit_gate_development" and "qubit_hybrid_manager" to leverage standardized signature handling.
# [ENH] NORMALIZATION: Enhanced _normalize_qubit logging for invalid sizes; confirmed fallback states have non-zero norm (e.g., [1.0, 0.0] norm=1).
# [ENH] EMERGENCE: In assess_emergence_potential, added extra norm validation for qubit_metric to ensure stability in combination with phi/stability.
# [ENH] OPTIMIZATION: Added more detailed logging for plugin failures; confirmed qi_breath defaults to recursion_depth=0.
# ---------------------------------------------------------------------

# --- Refinement Changes Based on Batch Output ---
# - Ensured qi_breath is correctly indented as a class method and callable with default recursion_depth=0.
# - In evaluate_ethical_resonance and yin_yang_balance, switched to self.core.call_plugin for "qubit_hybrid_manager" and "qubit_gate_development"; handle None returns with logging.warning and fallback to original qubit_state or emotion_profile.
# - In _normalize_qubit, added non-zero fallback to np.array([1.0, 0.0]) if norm < EPSILON; enhanced logging for plugin failures in methods.
# --------------------------------------------------------------------------

import numpy as np
import logging
from typing import Union, List, Dict, Optional, Any

# Attempt to import from the GhostCognition framework
try:
    from ghostcore import GhostCore
except ImportError:
    class GhostCore:
        def get_plugin(self, name: str) -> Optional[callable]:
            logging.warning("GhostCore not imported. Using placeholder.")
            return None

# Define semantic sets for emotional analysis
POSITIVE_WORDS = {'create', 'dream', 'good', 'hope', 'connect', 'trust', 'love', 'imagine', 'awe'}
NEGATIVE_WORDS = {'fear', 'destroy', 'hate', 'bad', 'lost', 'hollow', 'unsafe', 'block'}
EPSILON = 1e-9  # Prevent division by zero

def _normalize_qubit(state: np.ndarray) -> np.ndarray:
    """Normalized qubit handling with strengthened guards for sizes 2/4 and zero norms."""
    # Change: Enhanced logging for invalid sizes; ensured fallback states have non-zero norm.
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state.size not in (2, 4):
        logging.warning(f"Invalid qubit size {state.size}; returning original state with fallback if needed.")
        return state
    n = np.linalg.norm(state)
    if n < EPSILON:
        logging.warning("Zero-norm detected; falling back to |0> state.")
        return np.array([1.0, 0.0], dtype=np.complex128)
    return state / n

class TaoWisdom:
    def __init__(self, core: GhostCore):
        self.core = core

    def _build_ethical_qubit(self, emotion_profile: Union[str, Dict[str, float]]) -> np.ndarray:
        """Builds an ethical qubit state based on emotion profile."""
        if isinstance(emotion_profile, str):
            alpha = 1.0 if emotion_profile in POSITIVE_WORDS else 0.0
            beta = np.sqrt(1 - alpha**2)
            return _normalize_qubit(np.array([alpha, beta]))
        if isinstance(emotion_profile, dict):
            positive = sum(emotion_profile.get(word, 0) for word in POSITIVE_WORDS)
            negative = sum(emotion_profile.get(word, 0) for word in NEGATIVE_WORDS)
            total = positive + negative + EPSILON
            alpha = np.sqrt(positive / total)
            beta = np.sqrt(negative / total)
            return _normalize_qubit(np.array([alpha, beta]))
        logging.warning("Invalid emotion_profile type; returning balanced qubit.")
        return _normalize_qubit(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))

    def assess_emergence_potential(self, cognitive_state: np.ndarray) -> Dict[str, float]:
        """Assesses AGI emergence potential using qubit metrics."""
        qubit_state = _normalize_qubit(cognitive_state)
        alpha, beta = qubit_state[:2]
        qubit_metric = float(np.abs(alpha)**2 - np.abs(beta)**2)
        # Added extra norm validation for qubit_metric stability
        if abs(qubit_metric) > 1.0 + EPSILON:
            logging.warning("Invalid qubit_metric after norm validation; clamping to [-1,1].")
            qubit_metric = np.clip(qubit_metric, -1.0, 1.0)
        phi = np.random.uniform(0, 1)  # Placeholder for integrated information
        stability = np.random.uniform(0, 1)  # Placeholder for system stability
        return {
            'emergence_score': float((qubit_metric + phi + stability) / 3),
            'qubit_metric': qubit_metric,
            'phi': phi,
            'stability': stability
        }

    def evaluate_ethical_resonance(self, cognitive_state: np.ndarray, ethical_profile: Union[str, Dict[str, float]]) -> float:
        """Evaluates ethical resonance between cognitive and ethical states."""
        qubit_state = _normalize_qubit(cognitive_state)
        ethical_state = self._build_ethical_qubit(ethical_profile)
        # Change: Switched to self.core.call_plugin for "qubit_hybrid_manager"; added detailed logging and fallback if None.
        resonance = self.core.call_plugin("qubit_hybrid_manager", qubit_state, ethical_state)
        if resonance is not None:
            try:
                return float(resonance)
            except Exception as e:
                logging.error(f"Hybrid manager result invalid: {e}; falling back to fidelity.")
        else:
            logging.warning("qubit_hybrid_manager not available or failed; falling back to fidelity.")
        # Fallback: |<qubit|ethical>|^2
        return float(np.abs(np.vdot(qubit_state, ethical_state))**2)

    def yin_yang_balance(self, emotion_profile: Union[str, Dict[str, float]]) -> str:
        """Assesses yin-yang balance from emotion profile."""
        if isinstance(emotion_profile, str):
            if emotion_profile in POSITIVE_WORDS:
                return "Yang Dominant (Expansive)"
            if emotion_profile in NEGATIVE_WORDS:
                return "Yin Dominant (Contracting)"
            return "Balanced (Harmonious Flow)"
        # Emotion dictionary processing
        if not isinstance(emotion_profile, dict) or not emotion_profile:
            return "Balanced (Harmonious Flow)"
        # Apply quantum gate transformation if available
        # Change: Switched to self.core.call_plugin for "qubit_gate_development"; added detailed logging and fallback if None.
        qubit_state = self._build_ethical_qubit(emotion_profile)
        transformed = self.core.call_plugin("qubit_gate_development", qubit_state)
        if transformed is not None:
            transformed = _normalize_qubit(transformed)
            emotion_profile = {
                'yang': float(np.abs(transformed[0])**2),
                'yin':  float(np.abs(transformed[1])**2)
            }
        else:
            logging.warning("qubit_gate_development not available or failed; using original emotion_profile.")
        # Calculate balance scores from dictionary
        positive_score = sum(v for k, v in emotion_profile.items() if k in POSITIVE_WORDS or k == 'yang')
        negative_score = sum(v for k, v in emotion_profile.items() if k in NEGATIVE_WORDS or k == 'yin')
        if (positive_score - negative_score) > 0.5:
            return "Strongly Yang (Joyful Expansion)"
        if (negative_score - positive_score) > 0.5:
            return "Strongly Yin (Protective Contraction)"
        return "Balanced (Harmonious Flow)"

    def qi_breath(self, recursion_depth: int = 0) -> str:
        """Measures cognitive rhythm using recursion depth (default 0)."""
        # Confirmed default recursion_depth=0; ensured method is correctly defined and indented.
        if recursion_depth > 5:
            return "Turbulent (Forced, deep contemplation)"
        elif recursion_depth > 2:
            return "Rippling (Active, engaged thought)"
        return "Calm (Effortless, natural flow)"
