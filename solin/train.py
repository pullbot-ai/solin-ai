"""
Training loop and task definitions for Neural Ladder Training.
"""

import numpy as np
from typing import List, Tuple, Generator, Optional
from .core import Network, Architect
from .architect_local import SmartArchitect
from .stats import TrainingStats
import time


class Task:
    """Base task interface."""
    
    def generate_examples(self) -> Generator[Tuple[List[float], List[float]], None, None]:
        raise NotImplementedError

    def is_correct(self, output: List[float], target: List[float]) -> bool:
        return all(abs(o - t) < 0.3 for o, t in zip(output, target))


class ParityTask(Task):
    """Detect if number of 1s is even."""
    
    def __init__(self, input_size: int = 4):
        self.input_size = input_size

    def generate_examples(self):
        while True:
            bits = np.random.randint(0, 2, self.input_size).tolist()
            target = [1.0 if sum(bits) % 2 == 0 else 0.0]
            yield bits, target


class SequenceTask(Task):
    """Predict next element in a simple sequence."""
    
    def __init__(self, sequence_length: int = 4):
        self.sequence_length = sequence_length

    def generate_examples(self):
        while True:
            seq = np.random.randint(0, 2, self.sequence_length).tolist()
            target = [float(seq[-1])]  # Predict last element from previous ones
            yield seq[:-1], target


class ArithmeticTask(Task):
    """Simple binary addition."""
    
    def __init__(self, bits_per_number: int = 2):
        self.bits_per_number = bits_per_number

    def generate_examples(self):
        while True:
            a = np.random.randint(0, 2**self.bits_per_number)
            b = np.random.randint(0, 2**self.bits_per_number)
            result = a + b
            
            # Encode as binary
            a_bits = [float(x) for x in format(a, f'0{self.bits_per_number}b')]
            b_bits = [float(x) for x in format(b, f'0{self.bits_per_number}b')]
            result_bits = [float(x) for x in format(result, f'0{self.bits_per_number + 1}b')]
            
            yield a_bits + b_bits, result_bits


def train(task: Task, max_examples: int = 1000, input_size: int = 4,
          output_size: int = 1, use_smart_architect: bool = True,
          policy_path: Optional[str] = None) -> Tuple[Network, TrainingStats]:
    """
    Main Neural Ladder Training loop.
    
    Args:
        task: Task to train on
        max_examples: Maximum number of examples to process
        input_size: Size of input vector
        output_size: Size of output vector
        use_smart_architect: Use learned policy architect vs random
        policy_path: Path to save/load architect policy
    
    Returns:
        (trained_network, training_stats)
    """
    network = Network(input_size, output_size)
    
    if use_smart_architect:
        architect = SmartArchitect(policy_path=policy_path)
    else:
        architect = Architect()
    
    stats = TrainingStats()
    stats.start()
    
    print(f"Starting NLT training on {task.__class__.__name__}")
    print(f"Input size: {input_size}, Output size: {output_size}")
    print("-" * 50)
    
    for i, (example_input, target) in enumerate(task.generate_examples()):
        if i >= max_examples:
            break
        
        start_time = time.time()
        
        # Try current network
        output = network.forward(example_input)
        
        if task.is_correct(output, target):
            stats.record_example(
                'solved',
                len(network.nodes),
                len(network.edges),
                elapsed=time.time() - start_time
            )
            continue
        
        # Architect tries to solve
        solution = architect.solve(network, example_input, target)
        
        if solution:
            temp_network, reuse_stats = solution
            architect.merge(network, temp_network)
            stats.record_example(
                'merged',
                len(network.nodes),
                len(network.edges),
                new_nodes=reuse_stats.get('new_nodes', 0),
                reused_nodes=reuse_stats.get('reused_nodes', 0),
                elapsed=time.time() - start_time
            )
        else:
            stats.record_example(
                'failed',
                len(network.nodes),
                len(network.edges),
                elapsed=time.time() - start_time
            )
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Example {i+1}/{max_examples}: "
                  f"nodes={len(network.nodes)}, "
                  f"edges={len(network.edges)}, "
                  f"solved={stats.solved_immediately}, "
                  f"merged={stats.merged_solutions}, "
                  f"failed={stats.failed_attempts}")
    
    stats.finish()
    stats.main_network_nodes = len(network.nodes)
    
    return network, stats
