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
        self.max_size = max_size
        self.decay = decay
        self.data: list = []

    def add(self, vec: np.ndarray) -> None:
        """Add a vector to memory.
        
        Args:
            vec: 1D or 2D numpy array to store
        """
        if vec is None:
            raise ValueError("Cannot add None to memory")
        self.data.append(vec.copy())
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def get_weighted(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get stored vectors with decay weights.
        
        Returns:
            Tuple of (vectors array, normalized weights array)
        """
        if not self.data:
            return np.array([]), np.array([])
        
        # Weight older entries lower (exponential decay)
        weights = np.array([self.decay ** i for i in range(len(self.data))])[::-1]
        weights /= weights.sum() + 1e-9
        return np.array(self.data), weights

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
        self.threshold = threshold
        self.max_clusters = max_clusters
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
            
        if not self.centroids:
            self.centroids.append(vec.copy())
            return 0

        # Find nearest centroid
        dists = [np.linalg.norm(vec - c) for c in self.centroids]
        idx = int(np.argmin(dists))

        if dists[idx] < self.threshold:
            # Update centroid with online mean
            self.centroids[idx] = 0.9 * self.centroids[idx] + 0.1 * vec
            return idx
        else:
            # Create new cluster if space available
            if len(self.centroids) < self.max_clusters:
                self.centroids.append(vec.copy())
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
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, vec: np.ndarray) -> None:
        """Update detector with new observation.
        
        Args:
            vec: New observation vector
        """
        if vec is None:
            raise ValueError("Cannot update with None")
        self.history.append(vec.copy())

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
        
        baseline = np.mean(list(self.history), axis=0)
        return float(np.linalg.norm(vec - baseline))

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
        if drop_threshold >= rise_threshold:
            raise ValueError("drop_threshold must be less than rise_threshold")
            
        self.regret_window = regret_window
        self.rise_threshold = rise_threshold
        self.drop_threshold = drop_threshold
        self.regret_de_escalation_threshold = regret_de_escalation_threshold

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
            
        if self.state == "fast":
            return identity.copy()
        elif self.state == "cautious":
            # Reduce action magnitude by 30%
            modified = identity.copy()
            modified *= 0.7
            return modified
        elif self.state == "halt":
            # Stop all action
            return np.zeros_like(identity)
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
        if not isinstance(signal, (int, float)):
            raise ValueError(f"Signal must be numeric, got {type(signal)}")

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

        avg_regret = (
            float(np.mean(self.regret_memory)) if self.regret_memory else 0.0
        )

        # Only de-escalate if BOTH signal is low AND regret is low
        if (signal < self.drop_threshold and 
            avg_regret < self.regret_de_escalation_threshold):
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
        self.memory = DecayMemory(max_size=memory_size, decay=memory_decay)
        self.clusterer = SimpleClusterer(
            threshold=cluster_threshold, max_clusters=max_clusters
        )
        self.drift = DriftDetector(window=drift_window)
        self.gov = CounterfactualRegretGovernor(
            regret_window=regret_window,
            rise_threshold=rise_threshold,
            drop_threshold=drop_threshold,
            regret_de_escalation_threshold=regret_de_escalation_threshold,
        )
        self.step_count = 0

    def step(
        self, identity: np.ndarray, outcome_vec: np.ndarray
    ) -> Dict[str, Any]:
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
        
        if len(identity) != len(outcome_vec):
            raise ValueError(
                f"identity ({len(identity)}) and outcome_vec "
                f"({len(outcome_vec)}) must have same length"
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
        self.memory = DecayMemory(
            max_size=self.memory.max_size, decay=self.memory.decay
        )
        self.clusterer = SimpleClusterer(
            threshold=self.clusterer.threshold,
            max_clusters=self.clusterer.max_clusters,
        )
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