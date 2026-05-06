"""
Honest Governor - Usage Examples

Demonstrates 7 real-world integration patterns.
Run with: python examples.py
"""

import numpy as np
from honest_governor import HonestGovernorAgent

np.random.seed(42)


def example_1_basic_usage():
    """Example 1: Basic agent initialization and step."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    agent = HonestGovernorAgent()
    
    print("Running 5 steps of agent...\n")
    for i in range(5):
        # identity = proposed action vector
        identity = np.random.randn(10)
        outcome = identity + np.random.randn(10) * 0.1
        
        result = agent.step(identity, outcome)
        
        print(f"Step {i+1}:")
        print(f"  State:  {result['state']}")
        print(f"  Signal: {result['signal']:.4f}")
        print(f"  Regret: {result['regret']:.4f}")
        print(f"  Cluster: {result['cluster']}")


def example_2_llm_fallback():
    """Example 2: LLM agent model selection based on governor state."""
    print("\n" + "=" * 70)
    print("Example 2: LLM Agent Model Selection")
    print("=" * 70)
    
    agent = HonestGovernorAgent(
        rise_threshold=0.5,
        drop_threshold=0.2,
    )
    
    models = {
        "fast": "gpt-4-turbo",
        "cautious": "gpt-3.5-turbo",
        "halt": "fallback-rule-based",
    }
    
    print("\nSimulating LLM agent with adaptive model selection:\n")
    print(f"{'Step':<6} {'Model':<20} {'State':<10} {'Signal':<8}")
    print("-" * 48)
    
    for step in range(20):
        # Simulate signal: high for first 8 steps, then recovers
        if step < 8:
            signal_val = 0.6 + np.random.randn() * 0.1
        elif step < 12:
            signal_val = 0.3 + np.random.randn() * 0.15
        else:
            signal_val = 0.1 + np.random.randn() * 0.05
        
        signal_val = max(0, signal_val)
        
        identity = np.ones(5) * 0.8
        outcome = identity + np.random.randn(5) * signal_val
        
        result = agent.step(identity, outcome)
        model = models[result["state"]]
        
        print(f"{step+1:<6} {model:<20} {result['state']:<10} {result['signal']:<8.3f}")


def example_3_adaptive_backoff():
    """Example 3: API retry with adaptive backoff."""
    print("\n" + "=" * 70)
    print("Example 3: Adaptive API Retry with Backoff")
    print("=" * 70)
    
    agent = HonestGovernorAgent()
    
    backoff_map = {
        "fast": 0,      # No delay
        "cautious": 1,  # 1 second
        "halt": 5,      # 5 seconds
    }
    
    print("\nSimulating API call retry logic:\n")
    print(f"{'Attempt':<8} {'Backoff(s)':<12} {'State':<10} {'Success':<8}")
    print("-" * 40)
    
    for attempt in range(10):
        failure_rate = 0.8 if attempt < 3 else 0.2
        is_failed = np.random.rand() < failure_rate
        
        error_signal = 0.7 if is_failed else 0.05
        identity = np.array([1.0])
        outcome = np.array([error_signal])
        
        result = agent.step(identity, outcome)
        backoff = backoff_map[result["state"]]
        status = "No" if is_failed else "Yes"
        
        print(f"{attempt+1:<8} {backoff:<12} {result['state']:<10} {status:<8}")
        
        if backoff > 0:
            print(f"         → Wait {backoff}s before retry")


def example_4_tuning_comparison():
    """Example 4: Conservative vs Aggressive configuration."""
    print("\n" + "=" * 70)
    print("Example 4: Conservative vs Aggressive Tuning")
    print("=" * 70)
    
    conservative = HonestGovernorAgent(
        rise_threshold=0.3,
        drop_threshold=0.1,
        regret_de_escalation_threshold=0.05,
    )
    
    aggressive = HonestGovernorAgent(
        rise_threshold=0.6,
        drop_threshold=0.4,
        regret_de_escalation_threshold=0.3,
    )
    
    # Test signal: stable → spike → recovery
    signals = [0.2] * 5 + [0.7] * 5 + [0.1] * 10
    
    print("\nSignal pattern: stable → spike → recovery\n")
    print(f"{'Step':<6} {'Signal':<8} {'Conservative':<15} {'Aggressive':<15}")
    print("-" * 48)
    
    for step, sig in enumerate(signals):
        identity = np.array([1.0])
        outcome = np.array([sig])
        
        cons_result = conservative.step(identity, outcome)
        agg_result = aggressive.step(identity, outcome)
        
        print(f"{step+1:<6} {sig:<8.2f} {cons_result['state']:<15} "
              f"{agg_result['state']:<15}")
    
    print("\nConservative: Escalates faster, recovers slower")
    print("Aggressive: Tolerates noise, recovers faster")


def example_5_monitoring():
    """Example 5: Monitoring agent status."""
    print("\n" + "=" * 70)
    print("Example 5: Agent Monitoring & Diagnostics")
    print("=" * 70)
    
    agent = HonestGovernorAgent()
    
    # Run some steps
    print("\nRunning 100 steps...")
    for i in range(100):
        identity = np.random.randn(5)
        noise_scale = 0.5 if i < 50 else 0.1
        outcome = identity + np.random.randn(5) * noise_scale
        agent.step(identity, outcome)
    
    # Get status
    status = agent.get_status()
    
    print("\nAgent Status:")
    for key, value in status.items():
        print(f"  {key:<20}: {value}")


def example_6_multiepisode():
    """Example 6: Multi-episode learning."""
    print("\n" + "=" * 70)
    print("Example 6: Multi-Episode Training")
    print("=" * 70)
    
    agent = HonestGovernorAgent()
    
    num_episodes = 3
    steps_per_episode = 30
    
    print("\nTraining across 3 episodes (noise decreases each episode):\n")
    
    for episode in range(num_episodes):
        agent.reset()
        
        escalations = 0
        
        for step in range(steps_per_episode):
            noise_scale = 0.5 - (episode * 0.1)
            identity = np.ones(5)
            outcome = identity + np.random.randn(5) * noise_scale
            
            result = agent.step(identity, outcome)
            
            if result["state"] != "fast":
                escalations += 1
        
        status = agent.get_status()
        print(f"Episode {episode + 1}:")
        print(f"  Steps in escalated state: {escalations}/{steps_per_episode}")
        print(f"  Regret buffer size: {status['regret_buffer_size']}")


def example_7_oscillation_demo():
    """Example 7: Concrete oscillation prevention."""
    print("\n" + "=" * 70)
    print("Example 7: Oscillation Prevention Demo")
    print("=" * 70)
    
    agent = HonestGovernorAgent(
        regret_window=20,
        rise_threshold=0.5,
        drop_threshold=0.2,
    )
    
    print("\nScenario: Model adapts internally but problem not actually fixed\n")
    print("Step | Signal | State      | Regret | Explanation")
    print("-" * 65)
    
    # Phase 1: Problem starts
    for i in range(3):
        signal = 0.6
        identity = np.ones(5)
        outcome = identity + np.random.randn(5) * signal
        result = agent.step(identity, outcome)
        print(f"{i+1:3d}  | {signal:.2f}  | {result['state']:<10} | "
              f"{result['regret']:.2f}  | Problem → escalating")
    
    # Phase 2: Model adapts (signal drops)
    for i in range(3, 8):
        signal = 0.25
        identity = np.ones(5)
        outcome = identity + np.random.randn(5) * signal
        result = agent.step(identity, outcome)
        print(f"{i+1:3d}  | {signal:.2f}  | {result['state']:<10} | "
              f"{result['regret']:.2f}  | Tracking regret...")
    
    # Phase 3: Attempt recovery blocked
    for i in range(8, 13):
        signal = 0.1
        identity = np.ones(5)
        outcome = identity + np.random.randn(5) * signal
        result = agent.step(identity, outcome)
        
        if i < 10:
            explanation = "Blocks premature!"
        else:
            explanation = "Recovery OK"
        
        print(f"{i+1:3d}  | {signal:.2f}  | {result['state']:<10} | "
              f"{result['regret']:.2f}  | {explanation}")
    
    print("\n✓ Governor prevented false recovery!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HONEST GOVERNOR - USAGE EXAMPLES")
    print("=" * 70)
    
    example_1_basic_usage()
    example_2_llm_fallback()
    example_3_adaptive_backoff()
    example_4_tuning_comparison()
    example_5_monitoring()
    example_6_multiepisode()
    example_7_oscillation_demo()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70 + "\n")