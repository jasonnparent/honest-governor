# Honest Governor

**Reduces oscillation in threshold-based control systems under noisy signals.**

A simple, memory-based stability controller that stops adaptive systems from oscillating between safe and unsafe modes.

---

## The Problem

Adaptive systems often fail in a predictable loop:

1. Something goes wrong → system escalates (safer mode)
2. The system adapts internally (models adjust, filters stabilize)
3. Error signal drops → system assumes problem is gone
4. **It de-escalates too early** ← mistaking adaptation for recovery
5. The original instability is still present → system re-escalates
6. **Result**: oscillation, instability, wasted compute, unreliable behavior

The core issue: **You can't tell the difference between "the model learned" and "the environment fixed itself."**

---

## The Solution

Instead of only asking:

> "Is the signal safe right now?"

We also track:

> "If I reduce caution, will I immediately regret it?"

This is implemented using **counterfactual regret tracking**: a short-term memory of whether past de-escalations would have been mistakes.

---

## Core Mechanism

The governor maintains:

- A **state machine**:
  - `fast` → normal operation
  - `cautious` → reduced risk (70% action magnitude)
  - `halt` → safe / disabled mode

- A **signal score** computed from:
  - prediction error (how wrong was I?)
  - drift detection (is the environment changing?)

- A **regret memory buffer**:
  - tracks whether past de-escalations would have been mistakes

---

## Decision Logic

### Escalation (fast → cautious → halt)

If signal is high:
- **Escalate immediately** (no delay)

### De-escalation (halt → cautious → fast)

Only allowed if **BOTH** are true:
- signal is low
- recent regret is low

This prevents false recovery.

---

## Typical Impact

In simulated environments:
- **60–80% reduction** in state switching vs naive thresholding
- Faster reaction to real shocks
- More stable recovery behavior

---

## 📚 When to Use

✅ **Good fit**:
- Adaptive systems with internal model changes (LLM agents, control loops)
- Noisy signal environments
- High cost of false recovery (cascading failures)

❌ **Bad fit**:
- Deterministic systems (no noise)
- One-shot decisions (no history to track)
- Already have perfect signal classification

---

## Quick Start

### Installation

```bash
# Just copy honest_governor.py to your project
# Requires: numpy
pip install numpy
```

### Basic Usage

```python
import numpy as np
from honest_governor import HonestGovernorAgent

# Create agent
agent = HonestGovernorAgent()

# Each step: propose action, observe outcome
result = agent.step(
    identity=np.array([...]),      # Proposed action
    outcome_vec=np.array([...])    # What actually happened
)

# Result contains:
print(result["state"])             # Current state (fast/cautious/halt)
print(result["signal"])            # Instability score
print(result["regret"])            # Average regret (0-1)
print(result["modified_action"])   # Action after governor filtering
```

---

## Common Use Cases

### 1. LLM Agents with Tool Retry

```python
agent = HonestGovernorAgent(
    rise_threshold=0.5,    # Only escalate on real failure
    drop_threshold=0.2,    # Conservative recovery
)

# Choose model based on governor state
if agent.gov.state == "halt":
    model = "fallback"     # Most conservative
elif agent.gov.state == "cautious":
    model = "gpt3.5"       # Medium
else:
    model = "gpt4"         # Best performance
```

### 2. API Retry with Adaptive Backoff

```python
backoff_map = {
    "fast": 0,      # No delay
    "cautious": 5,  # 5 seconds
    "halt": 30,     # 30 seconds
}

backoff_seconds = backoff_map[agent.gov.state]
time.sleep(backoff_seconds)
```

### 3. Robotics / Control Systems

```python
# Dynamically reduce action magnitude
action = compute_action()
modified_action = agent.gov.act(action)
robot.apply(modified_action)
```

---

## Configuration

```python
agent = HonestGovernorAgent(
    # Memory
    memory_size=500,
    memory_decay=0.99,
    
    # Clustering
    cluster_threshold=0.2,
    max_clusters=10,
    
    # Drift detection
    drift_window=20,
    
    # Governor thresholds
    regret_window=30,
    rise_threshold=0.4,
    drop_threshold=0.15,
    regret_de_escalation_threshold=0.1,
)
```

### Tuning Guide

**More Conservative** (react fast, recover slow):
- Lower `rise_threshold` (e.g., 0.3)
- Lower `drop_threshold` (e.g., 0.1)
- Lower `regret_de_escalation_threshold` (e.g., 0.05)

**More Aggressive** (tolerate noise, trust recovery):
- Higher `rise_threshold` (e.g., 0.6)
- Higher `drop_threshold` (e.g., 0.4)
- Higher `regret_de_escalation_threshold` (e.g., 0.3)

## ⚠️ Limitations

This system is a lightweight control primitive, not a calibrated safety system.

Aggressive configurations (high rise thresholds, high regret tolerance, or low drop thresholds) can:
- delay recovery from instability
- create false-stable states
- amplify oscillation under noisy signals

Use caution in production or safety-critical environments.
---

## Monitoring

```python
# Get current status
status = agent.get_status()
print(status)
# {
#   'step': 42,
#   'state': 'cautious',
#   'memory_size': 42,
#   'num_clusters': 3,
#   'drift_buffer_size': 20,
#   'regret_buffer_size': 30,
# }

# Reset between episodes
agent.reset()
```

---

## Testing

Run the full test suite:

```bash
python test_honest_governor.py
```

Output:
```
============================================================
HONEST GOVERNOR TEST SUITE
============================================================

Testing DecayMemory... ✅
Testing SimpleClusterer... ✅
Testing DriftDetector... ✅
Testing CounterfactualRegretGovernor... ✅
Testing HonestGovernorAgent... ✅
Testing error handling... ✅

Testing oscillation prevention...
  With governor: 8 state changes
  Without governor: 22 state changes
  Reduction: 63.6%
  ✅ Oscillation prevention verified

============================================================
✅ ALL TESTS PASSED
============================================================
```

See `examples.py` for 7 detailed integration patterns.

---

## 📈 Performance

- **Time**: O(1) per step
- **Space**: O(n) where n = max buffer size
- **CPU**: Very lightweight (single-digit millisecond per step)
- **Memory**: ~50KB for default config

---

## 🔧 How It Works

### Signal Computation

```python
signal = 0.5 * prediction_error + 0.5 * drift_score
```

Weights can be tuned depending on your system.

- **prediction_error**: `||outcome - action||`
  - High when system behaves unexpectedly
- **drift_score**: Distance from rolling baseline
  - High when distribution shifts

### Regret Tracking

```python
would_regret = signal > rise_threshold
regret_memory.append(1 if would_regret else 0)
avg_regret = mean(regret_memory)
```

If `avg_regret` is high:
- Recent history shows signals spiking after de-escalation attempts
- The system is still unstable
- Don't de-escalate yet

---

## 📝 API Reference

### HonestGovernorAgent

```python
class HonestGovernorAgent:
    def __init__(
        memory_size: int = 500,
        memory_decay: float = 0.99,
        cluster_threshold: float = 0.2,
        max_clusters: int = 10,
        drift_window: int = 20,
        regret_window: int = 30,
        rise_threshold: float = 0.4,
        drop_threshold: float = 0.15,
        regret_de_escalation_threshold: float = 0.1,
    )
    
    def step(
        identity: np.ndarray,
        outcome_vec: np.ndarray
    ) -> Dict[str, Any]
    
    def reset() -> None
    
    def get_status() -> Dict[str, Any]
```

### CounterfactualRegretGovernor

```python
class CounterfactualRegretGovernor:
    def update(signal: float) -> Tuple[str, float]
    def act(identity: np.ndarray) -> np.ndarray
```

---

## 🤝 Contributing

Areas for improvement:
- Adaptive thresholds (learn rise/drop based on domain)
- Multi-signal fusion (combine multiple error sources)
- Explainability (visualize decision history)

---

**Made for systems that need to stay stable.**
