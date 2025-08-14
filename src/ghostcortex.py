import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional
from ghostcore import GhostCore

logging.basicConfig(level=logging.INFO, format='[%(levelname)s][%(name)s] %(message)s')
log = logging.getLogger(__name__)

class MemoryEcho:
    def __init__(self, content: str, strength: float, emotion: str, metadata: Dict[str, Any], timestamp: Optional[float] = None):
        self.content = content
        self.strength = strength
        self.emotion = emotion
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()

    def __repr__(self) -> str:
        return f"MemoryEcho(content='{self.content[:30]}...', strength={self.strength:.2f}, emotion='{self.emotion}')"

class GhostMemory:
    def __init__(self, core: GhostCore):
        self.core = core
        self.quantum_storage: Dict[str, np.ndarray] = {}
        self.echo_log: List[MemoryEcho] = []
        log.info("GhostMemory initialized with quantum storage.")

    def _create_quantum_state(self, echo: MemoryEcho) -> np.ndarray:
        creation_plugin = self.core.get_plugin("qubit_ethical_creation")
        state = None
        if creation_plugin:
            try:
                state = self.core.call_plugin(creation_plugin, echo.content, echo.emotion, echo.strength)
            except Exception as e:
                log.error(f"Quantum ethical creation failed: {e}")
        if state is None:
            seed = int.from_bytes(echo.content.encode(), 'little') % (2**32 - 1)
            np.random.seed(seed)
            state = np.array([np.random.rand(), np.random.rand()], dtype=np.complex128)
        return self.core.postprocess_qubit(state)

    def memorize(self, echo: MemoryEcho):
        if not isinstance(echo, MemoryEcho):
            log.error(f"Memorize failed: input must be MemoryEcho, not {type(echo)}.")
            return
        self.echo_log.append(echo)
        tag = echo.metadata.get('tag', f"untagged_{len(self.echo_log)}")
        quantum_state = self._create_quantum_state(echo)
        self.quantum_storage[tag] = quantum_state
        log.info(f"Memorized echo with tag '{tag}' as quantum state: {quantum_state}")
