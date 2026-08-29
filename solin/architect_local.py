"""
Local Architect AI — no API keys, runs entirely in GitHub Actions.
Uses a simple learned policy for structural decisions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os


class LocalArchitectPolicy:
    """
    A tiny policy network that decides structural actions.
    Trained alongside the main network, cached in GitHub Actions.
    """

    def __init__(self, state_size: int = 10, action_size: int = 4):
        self.state_size = state_size
        self.action_size = action_size

        # Simple policy: linear layer + softmax
        self.weights = np.random.randn(state_size, action_size) * 0.1
        self.bias = np.zeros(action_size)

    def get_action(self, state: np.ndarray) -> int:
        """Return action index from state."""
        logits = state @ self.weights + self.bias
        probs = self._softmax(logits)
        return np.argmax(probs)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def update(self, state: np.ndarray, action: int, reward: float, lr: float = 0.01):
        """Simple policy gradient update."""
        logits = state @ self.weights + self.bias
        probs = self._softmax(logits)

        # One-hot action
        action_onehot = np.zeros(self.action_size)
        action_onehot[action] = 1

        # Policy gradient
        grad = np.outer(state, (action_onehot - probs)) * reward
        self.weights += lr * grad
        self.bias += lr * (action_onehot - probs) * reward

    def save(self, filepath: str):
        np.savez(filepath, weights=self.weights, bias=self.bias)

    def load(self, filepath: str):
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']


class SmartArchitect:
    """
    Architect that uses learned policy for structural decisions.
    Still no API keys — everything local.
    """

    ACTIONS = ['add_node', 'add_edge', 'reuse', 'prune']

    def __init__(self, max_attempts: int = 20, max_new_nodes: int = 30,
                 policy_path: Optional[str] = None):
        self.max_attempts = max_attempts
        self.max_new_nodes = max_new_nodes
        self.policy = LocalArchitectPolicy()

        if policy_path and os.path.exists(policy_path):
            self.policy.load(policy_path)

    def solve(self, network, example_input: List[float],
              example_target: List[float]) -> Optional[Tuple]:
        """Attempt to solve example by growing structure."""
        temp = network.copy()

        for attempt in range(self.max_attempts):
            # Encode current state
            state = self._encode_state(temp, example_input)

            # Get action from policy
            action = self.policy.get_action(state)

            # Execute action
            new_nodes = self._execute_action(temp, action)

            # Check if solved
            output = temp.forward(example_input)
            if self._is_correct(output, example_target):
                reward = 1.0 / (attempt + 1)  # Prefer faster solutions
                self.policy.update(state, action, reward)

                reuse_stats = {
                    "new_nodes": new_nodes,
                    "reused_nodes": len(temp.nodes) - new_nodes,
                    "attempts": attempt + 1,
                    "action": self.ACTIONS[action]
                }
                return temp, reuse_stats

        return None

    def _encode_state(self, network, example_input: List[float]) -> np.ndarray:
        """Encode network state as feature vector."""
        features = [
            len(network.nodes) / 100.0,
            len(network.edges) / 200.0,
            len(example_input) / 10.0,
            np.mean(example_input) if example_input else 0,
            np.std(example_input) if example_input else 0,
            network._next_id / 100.0,
            len(network.output_ids) / 10.0,
            self.max_attempts / 100.0,
            self.max_new_nodes / 100.0,
            1.0  # bias term
        ]
        return np.array(features)

    def _execute_action(self, network, action: int) -> int:
        """Execute structural action. Returns new nodes added."""
        if action == 0:  # add_node
            return self._add_node(network)
        elif action == 1:  # add_edge
            self._add_edge(network)
            return 0
        elif action == 2:  # reuse
            self._reuse_structure(network)
            return 0
        elif action == 3:  # prune
            self._prune(network)
            return 0
        return 0

    def _add_node(self, network) -> int:
        """Add a node and connect it."""
        if len(network.nodes) >= network._next_id + self.max_new_nodes:
            return 0

        new_id = network._add_node('relu')
        from_id = np.random.choice(list(network.nodes.keys()))
        network.add_edge(from_id, new_id)
        to_id = np.random.choice(network.output_ids)
        network.add_edge(new_id, to_id)
        return 1

    def _add_edge(self, network):
        """Add edge between existing nodes."""
        if len(network.nodes) < 2:
            return

        from_id = np.random.choice(list(network.nodes.keys()))
        to_id = np.random.choice(list(network.nodes.keys()))
        if from_id != to_id and (from_id, to_id) not in network.edges:
            network.add_edge(from_id, to_id)

    def _reuse_structure(self, network):
        """Reuse existing structure by adjusting weights."""
        if network.edges:
            edge = list(network.edges)[np.random.randint(len(network.edges))]
            from_id, to_id = edge
            network.nodes[to_id].weights[from_id] *= (1 + np.random.randn() * 0.1)

    def _prune(self, network):
        """Remove weak edges."""
        if network.edges:
            edge = list(network.edges)[np.random.randint(len(network.edges))]
            from_id, to_id = edge
            weight = network.nodes[to_id].weights.get(from_id, 0)
            if abs(weight) < 0.01:
                network.edges.discard(edge)
                network.nodes[to_id].weights.pop(from_id, None)

    def _is_correct(self, output: List[float], target: List[float]) -> bool:
        return all(abs(o - t) < 0.3 for o, t in zip(output, target))

    def merge(self, main, temp):
        """Merge successful temporary structure into main network."""
        for nid, node in temp.nodes.items():
            if nid not in main.nodes:
                main.nodes[nid] = node
                main._next_id = max(main._next_id, nid + 1)

        for edge in temp.edges:
            if edge not in main.edges:
                from_id, to_id = edge
                main.edges.add(edge)
                main.nodes[to_id].weights[from_id] = temp.nodes[to_id].weights[from_id]

    def save_policy(self, filepath: str):
        self.policy.save(filepath)
