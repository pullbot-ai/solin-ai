"""
Train binary arithmetic model using Neural Ladder Training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from solin.train import ArithmeticTask, train
from solin.core import Network
from solin.stats import TrainingStats


def main():
    # Configuration
    BITS_PER_NUMBER = 2
    MAX_EXAMPLES = 5000
    MODEL_PATH = "models/arithmetic_nlt.json"
    STATS_PATH = "models/arithmetic_stats.json"
    POLICY_PATH = "models/arithmetic_policy.npz"
    
    print("=" * 60)
    print("Solin AI — Neural Ladder Training")
    print("Task: Binary Addition")
    print("=" * 60)
    
    # Create task
    task = ArithmeticTask(bits_per_number=BITS_PER_NUMBER)
    
    # Input: 2 numbers × BITS_PER_NUMBER bits each
    # Output: BITS_PER_NUMBER + 1 bits for result
    input_size = 2 * BITS_PER_NUMBER
    output_size = BITS_PER_NUMBER + 1
    
    # Train model
    network, stats = train(
        task=task,
        max_examples=MAX_EXAMPLES,
        input_size=input_size,
        output_size=output_size,
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
    
    return network, stats


if __name__ == "__main__":
    main()
