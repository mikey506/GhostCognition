import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger("ghostmemory")

@dataclass
class MemoryEcho:
    content: str
    strength: float
    emotion: str
    tag: str
    quantum_state: Any = None

class GhostMemory:
    def __init__(self, core):
        self.core = core
        self.echo_log: List[MemoryEcho] = []

    def memorize(self, echo: MemoryEcho):
        # Minimal sanitization
        if not isinstance(echo.content, str) or not (0.0 <= echo.strength <= 1.0):
            raise ValueError("Invalid MemoryEcho")
        # Normalize quantum_state if present
        if echo.quantum_state is not None:
            echo.quantum_state = self.core.postprocess_qubit(echo.quantum_state)
        self.echo_log.append(echo)
        logger.info("[ghostmemory] Memorized echo with tag=%s", echo.tag)

    def recall(self, query: str, limit: int = 5):
        # Basic semantic similarity proxy: substring match count + strength
        matches = []
        for e in self.echo_log:
            score = (query.lower() in e.content.lower()) * 1.0 + e.strength
            matches.append((score, e))
        matches.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in matches[:limit]]
        # Try quantum measurement plugin if present (support 1-arg or 2-arg variants)
        meas = getattr(self.core, "plugins", {}).get("qubit_measurement_manager")
        if meas and any(e.quantum_state is not None for e in results):
            try:
                states = np.stack([self.core.postprocess_qubit(e.quantum_state) for e in results])
                try:
                    # Try (query, states)
                    meas(query, states)  # if signature allows, it will run
                except TypeError:
                    # Try (states,) only
                    meas(states)
            except Exception as ex:
                logger.warning("[ghostmemory] Quantum measurement failed: %s. Using classical fallback.", ex)
        return results