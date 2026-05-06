"""
Honest Governor - Memory-based stability controller for adaptive systems.

Prevents premature de-escalation by tracking counterfactual regret,
distinguishing between model adaptation and environment recovery.
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Tuple


class DecayMemory:
    """Time-decayed memory for recent outcomes.

    Stores outcome vectors with exponential decay weighting,
    giving more importance to recent history.
    """

    def __init__(self, max_size: int = 500, decay: float = 0.99) -> None:
        """Initialize decay memory.

        Args:
            max_size: Maximum number of vectors to store
            decay: Exponential decay factor (0.99 means 1% decay per step)
        """
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")

        self.max_size = int(max_size)
        self.decay = float(decay)
        self.data: list = []

    def add(self, vec: np.ndarray) -> None:
        """Add a vector to memory.

        Args:
            vec: 1D or 2D numpy array to store
        """
        if vec is None:
            raise ValueError("Cannot add None to memory")

        arr = np.asarray(vec, dtype=np.float32)
        # Ensure stored items are 1D vectors when possible
        if arr.ndim == 0:
            arr = arr.reshape(1)
        self.data.append(arr.copy())
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def get_weighted(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get stored vectors with decay weights.

        Returns:
            Tuple of (vectors array, normalized weights array)
        """
        if not self.data:
            return np.empty((0,)), np.array([])

        # Stack into (N, D) array (consistent shape)
        try:
            vecs = np.vstack(self.data)
        except Exception:
            # Fallback: return as object array
            vecs = np.array(self.data, dtype=object)

        # Weight older entries lower (exponential decay)
        n = len(self.data)
        weights = np.array([self.decay ** i for i in range(n)])[::-1]
        weights = weights.astype(np.float32)
        weights /= weights.sum() + 1e-9
        return vecs, weights

    def __len__(self) -> int:
        """Return number of stored vectors."""
        return len(self.data)


class SimpleClusterer:
    """Simple online clustering for outcome patterns.

    Maintains running centroids of observed outcomes,
    useful for pattern recognition in adaptive systems.
    """

    def __init__(self, threshold: float = 0.2, max_clusters: int = 10) -> None:
        """Initialize clusterer.

        Args:
            threshold: Distance threshold for assigning to existing cluster
            max_clusters: Maximum number of clusters to maintain
        """
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        if max_clusters <= 0:
            raise ValueError("max_clusters must be > 0")

        self.threshold = float(threshold)
        self.max_clusters = int(max_clusters)
        self.centroids: list = []

    def assign(self, vec: np.ndarray) -> int:
        """Assign vector to nearest cluster, creating new if needed.

        Args:
            vec: Vector to assign

        Returns:
            Cluster ID (0-indexed)
        """
        if vec is None:
            raise ValueError("Cannot assign None")

        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        if not self.centroids:
            self.centroids.append(arr.copy())
            return 0

        # Find nearest centroid
        dists = [np.linalg.norm(arr - c) for c in self.centroids]
        idx = int(np.argmin(dists))

        if dists[idx] < self.threshold:
            # Update centroid with online mean (keep centroid shape)
            self.centroids[idx] = 0.9 * self.centroids[idx] + 0.1 * arr
            return idx
        else:
            # Create new cluster if space available
            if len(self.centroids) < self.max_clusters:
                self.centroids.append(arr.copy())
                return len(self.centroids) - 1
            return idx

    def __len__(self) -> int:
        """Return number of active clusters."""
        return len(self.centroids)


class DriftDetector:
    """Simple anchor-based drift detection.

    Tracks deviation from a rolling baseline to detect
    distribution shifts in the signal.
    """

    def __init__(self, window: int = 20) -> None:
        """Initialize drift detector.

        Args:
            window: Size of rolling window for baseline
        """
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window = int(window)
        self.history = deque(maxlen=window)

    def update(self, vec: np.ndarray) -> None:
        """Update detector with new observation.

        Args:
            vec: New observation vector
        """
        if vec is None:
            raise ValueError("Cannot update with None")
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        self.history.append(arr.copy())

    def score(self, vec: np.ndarray) -> float:
        """Compute drift score (distance from baseline).

        Args:
            vec: Vector to score

        Returns:
            Drift score (0 = no drift, higher = more drift)
        """
        if vec is None:
            raise ValueError("Cannot score None")
        if len(self.history) < 5:
            return 0.0

        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        baseline = np.mean(np.stack(list(self.history), axis=0), axis=0)
        # Ensure same shape
        if baseline.shape != arr.shape:
            # If shapes differ, compute norm on flattened difference
            return float(np.linalg.norm(baseline.flatten() - arr.flatten()))
        return float(np.linalg.norm(arr - baseline))

    def __len__(self) -> int:
        """Return window size."""
        return len(self.history)


class CounterfactualRegretGovernor:
    """Core Honest Governor - prevents premature de-escalation.

    Maintains a state machine (fast → cautious → halt) and tracks
    counterfactual regret to distinguish model adaptation from
    environment recovery.
    """

    def __init__(
        self,
        regret_window: int = 30,
        rise_threshold: float = 0.4,
        drop_threshold: float = 0.15,
        regret_de_escalation_threshold: float = 0.1,
    ) -> None:
        """Initialize governor.

        Args:
            regret_window: Size of regret history buffer
            rise_threshold: Signal level to trigger escalation
            drop_threshold: Signal level to allow de-escalation
            regret_de_escalation_threshold: Max average regret to de-escalate
        """
        if regret_window <= 0:
            raise ValueError("regret_window must be > 0")
        if not (0.0 <= drop_threshold < rise_threshold):
            raise ValueError("Require 0 <= drop_threshold < rise_threshold")
        if not (0.0 <= regret_de_escalation_threshold <= 1.0):
            raise ValueError("regret_de_escalation_threshold must be in [0,1]")

        self.regret_window = int(regret_window)
        self.rise_threshold = float(rise_threshold)
        self.drop_threshold = float(drop_threshold)
        self.regret_de_escalation_threshold = float(regret_de_escalation_threshold)

        self.regret_memory = deque(maxlen=regret_window)
        self.state = "fast"  # fast → cautious → halt

    def act(self, identity: np.ndarray) -> np.ndarray:
        """Modify action based on current safety state.

        Args:
            identity: Proposed action vector

        Returns:
            Modified action vector
        """
        if identity is None:
            raise ValueError("Cannot act on None")

        arr = np.asarray(identity, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        if self.state == "fast":
            return arr.copy()
        elif self.state == "cautious":
            # Reduce action magnitude by 30%
            return (arr * 0.7).copy()
        elif self.state == "halt":
            # Stop all action
            return np.zeros_like(arr)
        else:
            raise ValueError(f"Unknown state: {self.state}")

    def update(self, signal: float) -> Tuple[str, float]:
        """Update state with counterfactual regret logic.

        Escalation is immediate on high signal.
        De-escalation only allowed if both signal AND regret are low.

        Args:
            signal: Current instability signal (0-1 scale recommended)

        Returns:
            Tuple of (new_state, avg_regret)
        """
        if not isinstance(signal, (int, float, np.floating)):
            raise ValueError(f"Signal must be numeric, got {type(signal)}")

        signal = float(signal)

        # Escalate quickly on high signal
        if signal > self.rise_threshold:
            if self.state == "fast":
                self.state = "cautious"
            elif self.state == "cautious":
                self.state = "halt"

        # Counterfactual regret: would we regret de-escalating?
        # (i.e., would signal spike immediately after?)
        would_regret = signal > self.rise_threshold
        self.regret_memory.append(1 if would_regret else 0)

        avg_regret = float(np.mean(self.regret_memory)) if self.regret_memory else 0.0

        # Only de-escalate if BOTH signal is low AND regret is low
        if (signal < self.drop_threshold and avg_regret < self.regret_de_escalation_threshold):
            if self.state == "halt":
                self.state = "cautious"
            elif self.state == "cautious":
                self.state = "fast"

        return self.state, avg_regret

    def __len__(self) -> int:
        """Return regret buffer size."""
        return len(self.regret_memory)


class HonestGovernorAgent:
    """Full combined agent with memory, clustering, drift, and governor.

    Integrates all components to provide a complete stability control system
    for adaptive environments.
    """

    def __init__(
        self,
        memory_size: int = 500,
        memory_decay: float = 0.99,
        cluster_threshold: float = 0.2,
        max_clusters: int = 10,
        drift_window: int = 20,
        regret_window: int = 30,
        rise_threshold: float = 0.4,
        drop_threshold: float = 0.15,
        regret_de_escalation_threshold: float = 0.1,
    ) -> None:
        """Initialize Honest Governor Agent.

        Args:
            memory_size: Size of outcome memory buffer
            memory_decay: Decay factor for memory weighting
            cluster_threshold: Distance threshold for clustering
            max_clusters: Maximum number of clusters
            drift_window: Window size for drift detection
            regret_window: Size of regret history
            rise_threshold: Escalation signal threshold
            drop_threshold: De-escalation signal threshold
            regret_de_escalation_threshold: Regret threshold for de-escalation
        """
        # Normalize and validate basic inputs early
        if memory_size <= 0:
            raise ValueError("memory_size must be > 0")
        if not (0.0 < memory_decay <= 1.0):
            raise ValueError("memory_decay must be in (0,1]")

        self.memory = DecayMemory(max_size=memory_size, decay=memory_decay)
        self.clusterer = SimpleClusterer(threshold=cluster_threshold, max_clusters=max_clusters)
        self.drift = DriftDetector(window=drift_window)

        # Creating governor may raise ValueError for invalid thresholds; let it propagate
        self.gov = CounterfactualRegretGovernor(
            regret_window=regret_window,
            rise_threshold=rise_threshold,
            drop_threshold=drop_threshold,
            regret_de_escalation_threshold=regret_de_escalation_threshold,
        )
        self.step_count = 0

    def step(self, identity: np.ndarray, outcome_vec: np.ndarray) -> Dict[str, Any]:
        """Run one step of the agent.

        Args:
            identity: Proposed action vector
            outcome_vec: Observed result from environment

        Returns:
            Dictionary with keys:
                - state: Current safety state (fast/cautious/halt)
                - signal: Computed instability signal
                - regret: Average counterfactual regret
                - cluster: Cluster ID of outcome
                - drift: Drift score from baseline
                - modified_action: Action after governor modification
                - step: Step counter
        """
        # Validate inputs
        if identity is None or outcome_vec is None:
            raise ValueError("identity and outcome_vec cannot be None")

        identity = np.asarray(identity, dtype=np.float32)
        outcome_vec = np.asarray(outcome_vec, dtype=np.float32)

        # Allow scalar-to-vector promotion
        if identity.ndim == 0:
            identity = identity.reshape(1)
        if outcome_vec.ndim == 0:
            outcome_vec = outcome_vec.reshape(1)

        if identity.shape != outcome_vec.shape:
            raise ValueError(
                f"identity {identity.shape} and outcome_vec {outcome_vec.shape} must have same shape"
            )

        # Store outcome
        self.memory.add(outcome_vec)

        # Clustering
        cluster_id = self.clusterer.assign(outcome_vec)

        # Drift detection
        drift_score = self.drift.score(outcome_vec)
        self.drift.update(outcome_vec)

        # Compute instability signal: weighted combination of error and drift
        error = float(np.linalg.norm(outcome_vec - identity))
        signal = 0.5 * error + 0.5 * drift_score

        # Governor decision
        state, regret = self.gov.update(signal)

        # Apply governor to action
        modified_action = self.gov.act(identity)

        self.step_count += 1

        return {
            "state": state,
            "signal": signal,
            "regret": regret,
            "cluster": cluster_id,
            "drift": drift_score,
            "modified_action": modified_action,
            "step": self.step_count,
        }

    def reset(self) -> None:
        """Reset agent to initial state.

        Clears memory, clustering, drift history, and governor state.
        """
        self.memory = DecayMemory(max_size=self.memory.max_size, decay=self.memory.decay)
        self.clusterer = SimpleClusterer(threshold=self.clusterer.threshold, max_clusters=self.clusterer.max_clusters)
        self.drift = DriftDetector(window=self.drift.window)
        self.gov = CounterfactualRegretGovernor(
            regret_window=self.gov.regret_window,
            rise_threshold=self.gov.rise_threshold,
            drop_threshold=self.gov.drop_threshold,
            regret_de_escalation_threshold=self.gov.regret_de_escalation_threshold,
        )
        self.step_count = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics.

        Returns:
            Dictionary with status information
        """
        return {
            "step": self.step_count,
            "state": self.gov.state,
            "memory_size": len(self.memory),
            "num_clusters": len(self.clusterer),
            "drift_buffer_size": len(self.drift),
            "regret_buffer_size": len(self.gov),
        }
