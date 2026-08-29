"""
Training statistics tracking and reporting for NLT.
"""

from dataclasses import dataclass, field
from typing import List
import time
import json


@dataclass
class TrainingStats:
    """Full stats for one NLT training run."""
    
    # Core counters
    total_examples: int = 0
    solved_immediately: int = 0
    merged_solutions: int = 0
    failed_attempts: int = 0
    
    # Growth metrics
    nodes_over_time: List[int] = field(default_factory=list)
    edges_over_time: List[int] = field(default_factory=list)
    merge_sizes: List[int] = field(default_factory=list)
    reused_nodes_per_merge: List[int] = field(default_factory=list)
    
    # Memory metrics
    peak_temp_nodes: int = 0
    main_network_nodes: int = 0
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    time_per_example: List[float] = field(default_factory=list)
    
    def start(self):
        self.start_time = time.time()
    
    def finish(self):
        self.end_time = time.time()
    
    def record_example(self, result: str, nodes: int, edges: int,
                       new_nodes: int = 0, reused_nodes: int = 0,
                       elapsed: float = 0.0):
        """Record one training example result."""
        self.total_examples += 1
        
        if result == 'solved':
            self.solved_immediately += 1
        elif result == 'merged':
            self.merged_solutions += 1
            self.merge_sizes.append(new_nodes)
            self.reused_nodes_per_merge.append(reused_nodes)
        elif result == 'failed':
            self.failed_attempts += 1
        
        self.nodes_over_time.append(nodes)
        self.edges_over_time.append(edges)
        self.time_per_example.append(elapsed)
    
    def _pct(self, count: int) -> str:
        if self.total_examples == 0:
            return "0.0%"
        return f"{100 * count / self.total_examples:.1f}%"
    
    def report(self) -> str:
        """Generate human-readable training report."""
        total_time = self.end_time - self.start_time
        avg_time = (sum(self.time_per_example) / len(self.time_per_example)
                    if self.time_per_example else 0)
        
        final_nodes = self.nodes_over_time[-1] if self.nodes_over_time else 0
        final_edges = self.edges_over_time[-1] if self.edges_over_time else 0
        max_nodes = max(self.nodes_over_time) if self.nodes_over_time else 0
        
        avg_merge_size = (sum(self.merge_sizes) / len(self.merge_sizes)
                          if self.merge_sizes else 0)
        avg_reuse = (sum(self.reused_nodes_per_merge) / len(self.reused_nodes_per_merge)
                     if self.reused_nodes_per_merge else 0)
        
        return f"""
╔══════════════════════════════════════════════════════════╗
║         Neural Ladder Training Report                    ║
╚══════════════════════════════════════════════════════════╝

Training Examples: {self.total_examples}
  Solved immediately: {self.solved_immediately} ({self._pct(self.solved_immediately)})
  Architect merged:   {self.merged_solutions} ({self._pct(self.merged_solutions)})
  Architect failed:   {self.failed_attempts} ({self._pct(self.failed_attempts)})

Network Growth:
  Final nodes:    {final_nodes}
  Final edges:    {final_edges}
  Peak nodes:     {max_nodes}
  Growth ratio:   {final_nodes / max(1, self.total_examples):.3f} nodes/example

Reuse Metrics:
  Avg merge size:     {avg_merge_size:.1f} new nodes
  Avg nodes reused:   {avg_reuse:.1f} existing nodes
  Reuse efficiency:   {avg_reuse / max(1, avg_reuse + avg_merge_size) * 100:.1f}%

Performance:
  Total time:      {total_time:.2f}s
  Avg time/example: {avg_time * 1000:.2f}ms
  Examples/sec:    {self.total_examples / max(0.001, total_time):.1f}

Memory Efficiency:
  Main network:    {final_nodes} nodes
  Peak temp:       {self.peak_temp_nodes} nodes
  Memory ratio:    {final_nodes / max(1, self.peak_temp_nodes):.1f}x
"""
    
    def save(self, filepath: str):
        """Save stats to JSON."""
        data = {
            "total_examples": self.total_examples,
            "solved_immediately": self.solved_immediately,
            "merged_solutions": self.merged_solutions,
            "failed_attempts": self.failed_attempts,
            "final_nodes": self.nodes_over_time[-1] if self.nodes_over_time else 0,
            "final_edges": self.edges_over_time[-1] if self.edges_over_time else 0,
            "peak_nodes": max(self.nodes_over_time) if self.nodes_over_time else 0,
            "total_time": self.end_time - self.start_time,
            "avg_time_per_example": (sum(self.time_per_example) / len(self.time_per_example)
                                     if self.time_per_example else 0),
            "avg_merge_size": (sum(self.merge_sizes) / len(self.merge_sizes)
                               if self.merge_sizes else 0),
            "avg_reuse": (sum(self.reused_nodes_per_merge) / len(self.reused_nodes_per_merge)
                          if self.reused_nodes_per_merge else 0)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
