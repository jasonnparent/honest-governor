"""
Comprehensive test suite for Honest Governor.

Run with: python test_honest_governor.py
"""

import numpy as np
from honest_governor import (
    DecayMemory,
    SimpleClusterer,
    DriftDetector,
    CounterfactualRegretGovernor,
    HonestGovernorAgent,
)

np.random.seed(42)


def test_decay_memory():
    """Test DecayMemory functionality."""
    print("Testing DecayMemory...", end=" ")
    
    mem = DecayMemory(max_size=5, decay=0.99)
    assert len(mem) == 0
    
    # Add vectors
    v1 = np.array([1.0, 2.0, 3.0])
    mem.add(v1)
    assert len(mem) == 1
    
    # Get weighted
    vecs, weights = mem.get_weighted()
    assert vecs.shape == (1, 3)
    assert np.isclose(weights.sum(), 1.0)
    
    # Test max size enforcement
    for i in range(10):
        mem.add(np.random.randn(3))
    assert len(mem) == 5  # Max size respected
    
    print("✅")


def test_simple_clusterer():
    """Test SimpleClusterer functionality."""
    print("Testing SimpleClusterer...", end=" ")
    
    clust = SimpleClusterer(threshold=0.5, max_clusters=5)
    
    # First vector creates first cluster
    v1 = np.array([1.0, 0.0])
    id1 = clust.assign(v1)
    assert id1 == 0
    assert len(clust) == 1
    
    # Very close vector assigned to same cluster
    v2 = np.array([1.1, 0.1])
    id2 = clust.assign(v2)
    assert id2 == 0
    assert len(clust) == 1
    
    # Far vector creates new cluster
    v3 = np.array([10.0, 10.0])
    id3 = clust.assign(v3)
    assert id3 == 1
    assert len(clust) == 2
    
    print("✅")


def test_drift_detector():
    """Test DriftDetector functionality."""
    print("Testing DriftDetector...", end=" ")
    
    drift = DriftDetector(window=10)
    
    # Build baseline
    baseline = np.array([0.0, 0.0])
    for _ in range(10):
        drift.update(baseline + np.random.randn(2) * 0.01)
    
    assert len(drift) == 10
    
    # Test score near baseline
    score_near = drift.score(baseline)
    assert score_near < 0.1
    
    # Test score far from baseline
    score_far = drift.score(baseline + np.array([5.0, 5.0]))
    assert score_far > score_near
    
    print("✅")


def test_governor():
    """Test CounterfactualRegretGovernor state machine."""
    print("Testing CounterfactualRegretGovernor...", end=" ")
    
    gov = CounterfactualRegretGovernor(
        regret_window=10,
        rise_threshold=0.5,
        drop_threshold=0.2,
        regret_de_escalation_threshold=0.2,
    )
    
    # Initial state
    assert gov.state == "fast"
    
    # Escalation
    state, regret = gov.update(0.6)  # High signal
    assert state == "cautious"
    
    state, regret = gov.update(0.7)  # Even higher
    assert state == "halt"
    
    # De-escalation blocked by regret
    state, regret = gov.update(0.1)  # Low signal
    assert state == "halt"  # Still in halt due to regret
    
    # Eventually recovers
    for _ in range(15):
        state, regret = gov.update(0.1)
    assert state == "fast"
    
    # Test action modification
    identity = np.array([10.0, 10.0])
    gov.state = "fast"
    assert np.allclose(gov.act(identity), identity)
    
    gov.state = "cautious"
    modified = gov.act(identity)
    assert np.allclose(modified, identity * 0.7)
    
    gov.state = "halt"
    assert np.allclose(gov.act(identity), np.zeros_like(identity))
    
    print("✅")


def test_full_agent():
    """Test full HonestGovernorAgent integration."""
    print("Testing HonestGovernorAgent...", end=" ")
    
    agent = HonestGovernorAgent()
    
    # Single step
    identity = np.random.randn(10)
    outcome = np.random.randn(10)
    result = agent.step(identity, outcome)
    
    assert "state" in result
    assert "signal" in result
    assert "regret" in result
    assert "cluster" in result
    assert "drift" in result
    assert "modified_action" in result
    assert "step" in result
    
    # Multiple steps
    for _ in range(50):
        identity = np.random.randn(10)
        outcome = np.random.randn(10)
        agent.step(identity, outcome)
    
    # Check status
    status = agent.get_status()
    assert status["step"] == 51
    assert status["memory_size"] > 0
    
    # Test reset
    agent.reset()
    assert agent.step_count == 0
    assert status["step"] == 51  # Old status unchanged
    
    print("✅")


def test_oscillation_prevention():
    """Demonstrate oscillation prevention benefit."""
    print("\nTesting oscillation prevention...")
    
    agent = HonestGovernorAgent(
        regret_window=30,
        rise_threshold=0.4,
        drop_threshold=0.15,
        regret_de_escalation_threshold=0.1,
    )
    
    # Scenario: Signal drops after adaptation, then spikes again
    states_with_gov = []
    signals = [0.5] * 10 + [0.2] * 10 + [0.5] * 10 + [0.1] * 50
    
    for signal in signals:
        identity = np.ones(10)
        outcome = np.ones(10) + signal
        result = agent.step(identity, outcome)
        states_with_gov.append(result["state"])
    
    # Naive approach (no regret tracking)
    states_without_gov = []
    for signal in signals:
        if signal > 0.4:
            state = "halt"
        elif signal > 0.15:
            state = "cautious"
        else:
            state = "fast"
        states_without_gov.append(state)
    
    # Count state changes
    changes_with = sum(1 for i in range(1, len(states_with_gov))
                       if states_with_gov[i] != states_with_gov[i-1])
    changes_without = sum(1 for i in range(1, len(states_without_gov))
                          if states_without_gov[i] != states_without_gov[i-1])
    
    reduction = 100 * (changes_without - changes_with) / max(1, changes_without)
    
    print(f"  With governor: {changes_with} state changes")
    print(f"  Without governor: {changes_without} state changes")
    print(f"  Reduction: {reduction:.1f}%")
    
    assert changes_with < changes_without
    print("  ✅ Oscillation prevention verified\n")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("Testing error handling...", end=" ")
    
    agent = HonestGovernorAgent()
    
    # None inputs
    try:
        agent.step(None, np.array([1.0]))
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Mismatched lengths
    try:
        agent.step(np.array([1.0, 2.0]), np.array([1.0]))
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Invalid config (drop > rise)
    try:
        HonestGovernorAgent(drop_threshold=0.5, rise_threshold=0.3)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✅")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HONEST GOVERNOR TEST SUITE")
    print("=" * 60 + "\n")
    
    test_decay_memory()
    test_simple_clusterer()
    test_drift_detector()
    test_governor()
    test_full_agent()
    test_error_handling()
    test_oscillation_prevention()
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60 + "\n")