"""
Train sequence prediction model using Neural Ladder Training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from solin.train import SequenceTask, train
from solin.core import Network
from solin.stats import TrainingStats


def main():
    # Configuration
    SEQUENCE_LENGTH = 6
    MAX_EXAMPLES = 5000
    MODEL_PATH = "models/sequence_nlt.json"
    STATS_PATH = "models/sequence_stats.json"
    POLICY_PATH = "models/sequence_policy.npz"
    
    print("=" * 60)
    print("Solin AI — Neural Ladder Training")
    print("Task: Sequence Prediction")
    print("=" * 60)
    
    # Create task
    task = SequenceTask(sequence_length=SEQUENCE_LENGTH)
    
    # Train model
    network, stats = train(
        task=task,
        max_examples=MAX_EXAMPLES,
        input_size=SEQUENCE_LENGTH - 1,
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
    
    return network, stats


if __name__ == "__main__":
    main()
