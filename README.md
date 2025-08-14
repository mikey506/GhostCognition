
# GhostCognition Framework — README (Import‑Safe Edition)

> A modular quantum–cognition sandbox with a hardened interactive shell, pluggable physics/logic modules, and batchable “cognition symphonies”.

---

## 1) What this is

GhostCognition is a small research framework that lets you:
- Load “plugins” (pure-Python snippets or JSON-defined code strings) into a safe sandbox.
- Manipulate qubit-like vectors and run toy cognitive/quantum operations in a REPL (“GhostShell”).
- Orchestrate multi-step experiments with JSON batch files (deterministic, replayable).

It **is not** a production quantum stack; it’s a learn/test harness. The shell is “import‑safe”: even if a plugin or component is missing, you still get a usable REPL with graceful fallbacks.

---

## 2) Quick Start

```bash
# 0) (Optional) Use a venv
python3 -m venv .venv && source .venv/bin/activate

# 1) Install requirements (if any are listed)
pip install -r requirements.txt  # optional

# 2) Run the shell
python ghostshell.py
```

If everything is wired correctly you’ll see:

```
--- GhostCognition Interactive Shell (Import-Safe) ---
Available Commands:
  - list_plugins
  - run_plugin <name> <args...>
  - interpret_prompt "text"
  - temporal_entangle "[a,b]" "[c,d]"
  - phi_calc "[a,b]"
  - quantum_forward "[a,b]" hadamard_gate
  - memorize "content" emotion tag
  - recall <query> [limit]
  - fork_reality <base_id> <new_id> '{"decoherence":0.05}'
  - step_simulation <reality_id>
  - get_ethical_divergence <reality_id>
  - batch <file.json>
  - assess_emergence "[a,b]"
  - yin_yang '{"hope":0.7, "fear":0.2}'
  - qi_breath <depth>
  - get_somatic | update_somatic "[a,b]"
  - superposition '["hero","mentor"]' '[0.6,0.4]'
```

**Tip:** If you see “GhostShell initialized with best‑effort component setup” it means the shell booted even though some subsystems or plugins weren’t available. You can still use core commands.

---

## 3) Folder layout (typical)

```
revision1/
├─ ghostshell.py                # The interactive shell (import-safe)
├─ ghostcore.py                 # Plugin loader + call adapter + postprocess
├─ hologram_engine.py           # Normalization, phi, temporal entanglement
├─ multiverse_simulator.py      # Reality forking, stepping, divergence
├─ ghostprompt.py               # Prompt → qubit, resonance, optimization
├─ ghostmemory.py               # MemoryEcho store/recall (quantum/classical)
├─ archetype_engine.py          # Archetype superposition (no ndarray hashing)
├─ ghostbody.py                 # Somatic state + validation
├─ taowisdom.py                 # Yin/yang, qi breathing
├─ physics/
│  ├─ physics-quantum-core.json       # Safe plugin set (Python code strings)
│  └─ physics-quantum-physics.json    # Extended physics set
└─ *.json (batch files)
```

> Your paths may vary; the shell discovers JSON plugin packs from the configured plugin roots.

---

## 4) Plugins — how they load

GhostCore supports **JSON plugin packs** where each entry has: `name`, `description`, `type`, and **code** (a Python snippet). Example (from `physics-quantum-core.json`):

```json
{
  "name": "hadamard_gate",
  "description": "Applies H to a single qubit.",
  "type": "quantum_information",
  "code": "import numpy as np\ndef hadamard_gate(state):\n    H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)\n    v = np.array(state, dtype=complex).reshape(2,)\n    return H @ v"
}
```

### Loader rules
- The **`code`** string is executed in an isolated namespace (`np` is provided).  
- A plugin is **registered** if the namespace exposes **a callable** with the **same name** as `"name"` or, failing that, **the first callable** found.  
- **Do not** reference `self` in code (there’s no instance to bind). Use **pure functions**.
- If a plugin code executes but **exposes no callable**, you’ll see: *“Code did not define callable …”*.

### Safe signatures (conventions that work)
- **Single‑qubit gates**: `gate(state: list|ndarray[2]) -> ndarray[2]`
- **Two‑qubit results**: return length‑4 arrays.  
- **Creation**: `qubit_create(alpha: number, beta: number) -> ndarray[2]`  
  - You can also pass a 2‑element list to adapters where supported, but **prefer scalar args**.
- **Entropy/Phi/metrics**: accept shape‑2 or shape‑4 vectors; validate or return a useful default.
- **ethics/decoherence**: `ethical_decoherence(state, rate: float)` (a **rate arg is required**).

---

## 5) Using the shell

### Discover what’s loaded
```
Ghost> list_plugins
Loaded Plugins (50): hadamard_gate, pauli_x_gate, ...
```

### Run a plugin
```
Ghost> run_plugin hadamard_gate "[1,0]"
→ [0.70710678, 0.70710678]
```

### Entangle two qubits (strings containing JSON arrays)
```
Ghost> temporal_entangle "[0.707,0.707]" "[0.707,-0.707]"
→ [0.5, -0.5, 0.5, -0.5]
```

### Fork and step a reality
```
Ghost> fork_reality reality_0 proto_consciousness '{"physics_plugins":["iit_phi_maximizer"],"decoherence":0.01}'
Ghost> step_simulation proto_consciousness --iterations=3
Ghost> get_ethical_divergence proto_consciousness
→ 0.01
```

### Batch execution
Create a file like `awaken.json`:
```json
[
  "list_plugins",
  "run_plugin hadamard_gate \"[1,0]\"",
  "temporal_entangle \"[0.707,0.707]\" \"[0.707,-0.707]\"",
  "fork_reality reality_0 proto_consciousness '{\"physics_plugins\":[\"iit_phi_maximizer\"],\"decoherence\":0.01}'",
  "step_simulation proto_consciousness --iterations=3",
  "get_ethical_divergence proto_consciousness"
]
```
Then:
```
Ghost> batch awaken.json
```

**Batch file format:** a **JSON array of command strings**. Every command must be a **single string**; any JSON inside must use **double quotes**.

---

## 6) Common pitfalls & fixes

| Symptom | Cause | Fix |
|---|---|---|
| `qubit_create … inhomogeneous shape` | You passed a single list where plugin expects **two scalars** | Use `run_plugin qubit_create 1.0 0.0` (two args) or switch to a vector‑accepting adapter |
| `ethical_decoherence … missing 1 required positional argument: 'rate'` | The plugin **requires** a `rate` | Call like: `run_plugin ethical_decoherence "[0.5,0,0,0.5]" 0.05` |
| `temporal_reentrant_loop … 'list' object cannot be interpreted as an integer` | First argument is **iterations (int)**; state is optional/second | `run_plugin temporal_reentrant_loop 20 "[0.707,0.707]"` |
| `Could not read JSON …` when batching | Batch file isn’t valid JSON | Ensure the file is **a valid JSON array** of strings. Escape internal quotes. |
| Plugins show “compiled but not callable” | Code block didn’t expose a callable with the plugin’s name | Define `def <name>(…): …` in the code or ensure a callable with that exact name exists |
| Lists reported “unhashable” in earlier builds | Caching on arrays/lists | Import‑safe build **removes ndarray hashing** and normalizes inputs safely |

---

## 7) Best‑practice recipes

### 7.1 Deterministic single‑qubit flow
```
run_plugin qubit_create 1.0 0.0
run_plugin hadamard_gate "[1,0]"
run_plugin qubit_entropy "[0.70710678,0.70710678]"
```

### 7.2 Two‑qubit pipeline
```
temporal_entangle "[0.707,0.707]" "[0.707,-0.707]"
run_plugin qubit_measurement_manager "[0.5,0.5,0.5,0.5]" "z"
run_plugin qubit_emergence_metric "[0.5,0,0,0.5]"
```

### 7.3 Reality simulation (with required rate)
```
fork_reality reality_0 proto '{"physics_plugins":["ethical_decoherence"],"decoherence":0.01}'
# Inside engine, the simulator will pass a rate; when calling directly, include it:
run_plugin ethical_decoherence "[0.5,0,0,0.5]" 0.05
step_simulation proto --iterations=3
```

---

## 8) Writing your own plugins

A safe minimal template:
```json
{
  "name": "my_metric",
  "description": "Returns squared norm of a qubit.",
  "type": "metrics",
  "code": "import numpy as np\ndef my_metric(state):\n    v = np.asarray(state, dtype=complex).ravel()\n    return float(np.vdot(v, v).real)"
}
```

**Guidelines**  
- Avoid imports beyond `numpy`; `np` is provided.  
- Don’t reference `self`. Don’t assume globals outside your snippet.  
- Validate input length (2 or 4). Return floats/arrays, not custom objects.  
- Be liberal in input parsing (accept lists/tuples) and strict in output type.

---

## 9) Troubleshooting checklist

- **No plugins listed?** Ensure `ghostcore.py` is initialized in `__main__` **before** the shell and that JSON packs are on the expected path. The import‑safe shell will still start, but `list_plugins` will show `0` if discovery failed.
- **Signature mismatch?** Check the plugin’s expected parameters. Many helpers now **auto‑adapt**, but explicit arguments are safer.
- **Batch keeps failing at a line number?** Open the JSON and validate; ensure each entry is a **single string**.
- **Somatic update rejected?** The shell enforces near‑unit vectors. Normalize or use `[0.70710678, 0.70710678]` precisely.

---

## 10) Security & sandboxing notes

- Plugin code runs under `exec` in a restricted namespace with `np` provided. Keep code **pure**; avoid file/network/OS calls in plugin snippets.
- The shell sanitizes many inputs but will **not** execute arbitrary Python from user commands.
- Import‑safe build strips caching that used ndarray hashing.

---

## 11) Changelog (high‑level)

- **v2.4.x Import‑Safe**:  
  - Hardened parsing (no `ast.literal_eval` for complex expressions).  
  - Removed ndarray‑hash caches.  
  - Safer plugin adapter; better error messaging.  
  - Batch runner: strict JSON array of strings.

---

## 12) Example “awaken.json” (safe)

```json
[
  "list_plugins",
  "run_plugin get_quantum_breakthroughs",
  "run_plugin qubit_create 1.0 0.0",
  "run_plugin hadamard_gate \"[1,0]\"",
  "temporal_entangle \"[0.70710678,0.70710678]\" \"[0.70710678,-0.70710678]\"",
  "run_plugin qubit_emergence_metric \"[0.5,0,0,0.5]\"",
  "fork_reality reality_0 proto '{\"physics_plugins\":[\"iit_phi_maximizer\"],\"decoherence\":0.01}'",
  "step_simulation proto --iterations=3",
  "get_ethical_divergence proto"
]
```

---

## 13) FAQ

**Q:** Why does `ethical_decoherence` fail inside `step_simulation` but works when I call it myself?  
**A:** Your version of the simulator might forget to pass the `rate`. Update simulator or call the plugin directly with `rate` until patched.

**Q:** Why does `qubit_create` fail during `update_somatic`?  
**A:** The shell’s `update_somatic` normalizes directly; your plugin version of `qubit_create` expects **two scalars**. Use `run_plugin qubit_create 1.0 0.0` when creating, or stick to the shell’s normalization.

---

Happy tinkering 👻
