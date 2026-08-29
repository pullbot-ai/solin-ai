"""
Core Neural Ladder Training structures.
Node, Network, and Architect that grows networks by solving examples.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import json


class Node:
    """Single neuron in the neural ladder network."""

    def __init__(self, node_id: int, activation: str = 'relu'):
        self.id = node_id
        self.weights: Dict[int, float] = {}
        self.bias: float = 0.0
        self.activation = activation
        self.value: float = 0.0

    def forward(self, values: Dict[int, float]) -> float:
        total = self.bias
        for in_id, w in self.weights.items():
            total += values.get(in_id, 0.0) * w
        self.value = self._activate(total)
        return self.value

    def _activate(self, x: float) -> float:
        if self.activation == 'relu':
            return max(0.0, x)
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if self.activation == 'tanh':
            return np.tanh(x)
        if self.activation == 'linear':
            return x
        raise ValueError(f"Unknown activation: {self.activation}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "weights": self.weights,
            "bias": self.bias,
            "activation": self.activation
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        node = cls(data["id"], data["activation"])
        node.weights = {int(k): v for k, v in data["weights"].items()}
        node.bias = data["bias"]
        return node


class Network:
    """Directed acyclic graph of neural nodes."""

    def __init__(self, input_size: int, output_size: int):
        self.nodes: Dict[int, Node] = {}
        self.edges: set = set()
        self.input_ids: List[int] = []
        self.output_ids: List[int] = []
        self._next_id = 0

        # Create input nodes
        for _ in range(input_size):
            self.input_ids.append(self._add_node('linear'))
        # Create output nodes
        for _ in range(output_size):
            self.output_ids.append(self._add_node('linear'))

    def _add_node(self, activation: str = 'relu') -> int:
        node = Node(self._next_id, activation)
        self.nodes[node.id] = node
        self._next_id += 1
        return node.id

    def add_edge(self, from_id: int, to_id: int, weight: Optional[float] = None):
        if weight is None:
            weight = np.random.randn() * 0.1
        self.nodes[to_id].weights[from_id] = weight
        self.edges.add((from_id, to_id))

    def forward(self, input_vector: List[float]) -> List[float]:
        values = {}
        for i, val in enumerate(input_vector):
            values[self.input_ids[i]] = val

        # Topological sort by node ID (nodes added in order)
        for node_id in sorted(self.nodes.keys()):
            if node_id not in self.input_ids:
                values[node_id] = self.nodes[node_id].forward(values)

        return [values[out_id] for out_id in self.output_ids]

    def copy(self) -> 'Network':
        """Deep copy of the network."""
        new_net = Network(len(self.input_ids), len(self.output_ids))
        new_net.nodes = {nid: Node(nid, n.activation) for nid, n in self.nodes.items()}
        for nid, node in new_net.nodes.items():
            node.weights = dict(self.nodes[nid].weights)
            node.bias = self.nodes[nid].bias
        new_net.edges = set(self.edges)
        new_net._next_id = self._next_id
        return new_net

    def save(self, filepath: str):
        """Save network to JSON."""
        data = {
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "next_id": self._next_id,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
            "edges": [[a, b] for a, b in self.edges]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Network':
        """Load network from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        net = cls.__new__(cls)
        net.input_ids = data["input_ids"]
        net.output_ids = data["output_ids"]
        net._next_id = data["next_id"]
        net.nodes = {int(k): Node.from_dict(v) for k, v in data["nodes"].items()}
        net.edges = set(tuple(e) for e in data["edges"])
        return net


class Architect:
    """Constructs temporary solutions and merges them into the main network."""

    def __init__(self, max_attempts: int = 20, max_new_nodes: int = 30):
        self.max_attempts = max_attempts
        self.max_new_nodes = max_new_nodes

    def solve(self, network: Network, example_input: List[float],
              example_target: List[float]) -> Optional[Tuple[Network, dict]]:
        """
        Attempt to construct a temporary network that solves the example.
        Returns (temp_network, reuse_stats) or None if failed.
        """
        temp = network.copy()

        for attempt in range(self.max_attempts):
            new_nodes = self._grow(temp)

            output = temp.forward(example_input)
            if self._is_correct(output, example_target):
                reuse_stats = {
                    "new_nodes": new_nodes,
                    "reused_nodes": len(temp.nodes) - new_nodes,
                    "attempts": attempt + 1
                }
                return temp, reuse_stats

        return None

    def _grow(self, network: Network) -> int:
        """Add random structure. Returns number of new nodes added."""
        new_id = network._add_node('relu')

        # Connect from a random existing node
        from_id = np.random.choice(list(network.nodes.keys()))
        network.add_edge(from_id, new_id)

        # Connect to a random output or later node
        to_id = np.random.choice(network.output_ids)
        network.add_edge(new_id, to_id)

        # Sometimes add a second node for depth
        if np.random.random() < 0.3:
            new_id2 = network._add_node('relu')
            network.add_edge(new_id, new_id2)
            network.add_edge(new_id2, to_id)
            return 2

        return 1

    def _is_correct(self, output: List[float], target: List[float]) -> bool:
        return all(abs(o - t) < 0.3 for o, t in zip(output, target))

    def merge(self, main: Network, temp: Network):
        """Merge successful temporary structure into main network."""
        # Add any new nodes
        for nid, node in temp.nodes.items():
            if nid not in main.nodes:
                main.nodes[nid] = node
                main._next_id = max(main._next_id, nid + 1)

        # Add any new edges
        for edge in temp.edges:
            if edge not in main.edges:
                from_id, to_id = edge
                main.edges.add(edge)
                main.nodes[to_id].weights[from_id] = temp.nodes[to_id].weights[from_id]
