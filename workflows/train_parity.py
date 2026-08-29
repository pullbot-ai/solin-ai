"""
Train parity detection model using Neural Ladder Training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from solin.train import ParityTask, train
from solin.core import Network
from solin.stats import TrainingStats


def main():
    # Configuration
    INPUT_SIZE = 8
    MAX_EXAMPLES = 5000
    MODEL_PATH = "models/parity_nlt.json"
    STATS_PATH = "models/parity_stats.json"
    POLICY_PATH = "models/parity_policy.npz"
    
    print("=" * 60)
    print("Solin AI — Neural Ladder Training")
    print("Task: Parity Detection")
    print("=" * 60)
    
    # Create task
    task = ParityTask(input_size=INPUT_SIZE)
    
    # Train model
    network, stats = train(
        task=task,
        max_examples=MAX_EXAMPLES,
        input_size=INPUT_SIZE,
        output_size=1,
        use_smart_architect=True,
        policy_path=POLICY_PATH
    )
    
    # Print report
    print(stats.report())
    
    # Save model and stats
    os.makedirs("models", exist_ok=True)
    network.save(MODEL_PATH)
    stats.save(STATS_PATH)
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Stats saved to: {STATS_PATH}")
    
    # Final accuracy test
    correct = 0
    total = 1000
    for _ in range(total):
        import numpy as np
        bits = np.random.randint(0, 2, INPUT_SIZE).tolist()
        output = network.forward(bits)[0]
        target = 1.0 if sum(bits) % 2 == 0 else 0.0
        if abs(output - target) < 0.3:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"\nFinal Test Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    return network, stats


if __name__ == "__main__":
    main()
