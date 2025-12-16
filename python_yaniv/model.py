"""
PyTorch Neural Network for Action-Scoring

The ActionValueNet scores (state, action) pairs.
For each legal action, compute score and pick the best.

Supports both PyTorch (GPU-capable) and NumPy fallback.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import json

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .encoding import StateActionEncoder
from .env import Action, ActionType


class ActionValueNet:
    """
    Action-value network that scores (state, action) pairs.

    Architecture:
    - Input: concatenated [state_features | action_features]
    - Hidden: Linear -> ReLU -> Linear -> ReLU
    - Output: Single scalar score

    Works with PyTorch if available, otherwise uses NumPy.
    """

    def __init__(
        self,
        input_dim: int = 281,  # Default: 219 (state) + 62 (action)
        hidden_dim: int = 256,
        hidden_dim2: int = 128,
        use_torch: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.use_torch = use_torch and TORCH_AVAILABLE

        if self.use_torch:
            self._init_torch_model()
        else:
            self._init_numpy_model()

    def _init_torch_model(self) -> None:
        """Initialize PyTorch model."""
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, 1),
        )
        # Initialize weights
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _init_numpy_model(self) -> None:
        """Initialize NumPy weights for fallback."""
        # Xavier initialization
        def xavier_init(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        self.w1 = xavier_init(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.w2 = xavier_init(self.hidden_dim, self.hidden_dim2)
        self.b2 = np.zeros(self.hidden_dim2, dtype=np.float32)
        self.w3 = xavier_init(self.hidden_dim2, 1)
        self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim] or [input_dim]

        Returns:
            Scores [batch_size] or scalar
        """
        if self.use_torch:
            return self._forward_torch(x)
        else:
            return self._forward_numpy(x)

    def _forward_torch(self, x: np.ndarray) -> np.ndarray:
        """PyTorch forward pass."""
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32)
            if xt.dim() == 1:
                xt = xt.unsqueeze(0)
            scores = self.model(xt).squeeze(-1)
            return scores.numpy()

    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy forward pass."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Layer 1
        h1 = x @ self.w1 + self.b1
        h1 = np.maximum(h1, 0)  # ReLU

        # Layer 2
        h2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(h2, 0)  # ReLU

        # Output layer
        out = h2 @ self.w3 + self.b3
        return out.squeeze(-1)

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights as numpy arrays."""
        if self.use_torch:
            state_dict = self.model.state_dict()
            return {
                "w1": state_dict["0.weight"].numpy().T,  # Transpose for consistency
                "b1": state_dict["0.bias"].numpy(),
                "w2": state_dict["2.weight"].numpy().T,
                "b2": state_dict["2.bias"].numpy(),
                "w3": state_dict["4.weight"].numpy().T,
                "b3": state_dict["4.bias"].numpy()
            }
        else:
            return {
                "w1": self.w1.copy(),
                "b1": self.b1.copy(),
                "w2": self.w2.copy(),
                "b2": self.b2.copy(),
                "w3": self.w3.copy(),
                "b3": self.b3.copy()
            }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set model weights from numpy arrays."""
        if self.use_torch:
            state_dict = {
                "0.weight": torch.tensor(weights["w1"].T),
                "0.bias": torch.tensor(weights["b1"]),
                "2.weight": torch.tensor(weights["w2"].T),
                "2.bias": torch.tensor(weights["b2"]),
                "4.weight": torch.tensor(weights["w3"].T),
                "4.bias": torch.tensor(weights["b3"])
            }
            self.model.load_state_dict(state_dict)
        else:
            self.w1 = weights["w1"].copy()
            self.b1 = weights["b1"].copy()
            self.w2 = weights["w2"].copy()
            self.b2 = weights["b2"].copy()
            self.w3 = weights["w3"].copy()
            self.b3 = weights["b3"].copy()

    def save(self, filepath: str) -> None:
        """Save model weights to JSON file."""
        weights = self.get_weights()
        data = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_dim2": self.hidden_dim2,
            "weights": {k: v.tolist() for k, v in weights.items()}
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load model weights from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.input_dim = data["input_dim"]
        self.hidden_dim = data["hidden_dim"]
        self.hidden_dim2 = data["hidden_dim2"]

        weights = {k: np.array(v, dtype=np.float32) for k, v in data["weights"].items()}

        if self.use_torch:
            self._init_torch_model()

        self.set_weights(weights)

    def copy(self) -> "ActionValueNet":
        """Create a deep copy of the model."""
        new_model = ActionValueNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            hidden_dim2=self.hidden_dim2,
            use_torch=self.use_torch
        )
        new_model.set_weights(self.get_weights())
        return new_model

    def to_device(self, device: str) -> None:
        """Move model to specified device (torch only)."""
        if self.use_torch and device != "cpu":
            self.model = self.model.to(device)


class TorchPolicy:
    """
    Policy wrapper that uses ActionValueNet to choose actions.
    Scores all legal actions and picks the best (or samples with temperature).
    """

    def __init__(
        self,
        model: ActionValueNet,
        encoder: Optional[StateActionEncoder] = None,
        temperature: float = 0.0,
        device: str = "cpu"
    ):
        self.model = model
        self.encoder = encoder or StateActionEncoder()
        self.temperature = temperature
        self.device = device

        if model.use_torch and device != "cpu":
            model.to_device(device)

    def choose_action(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Action]
    ) -> Action:
        """
        Choose action by scoring all legal actions.

        Args:
            observation: Game state observation
            legal_actions: List of legal actions

        Returns:
            Selected action
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Encode all (state, action) pairs
        x = self.encoder.encode_state_action_batch(observation, legal_actions)

        # Get scores for all actions
        scores = self.model.forward(x)

        if self.temperature <= 0:
            # Greedy selection
            best_idx = int(np.argmax(scores))
        else:
            # Softmax sampling with temperature
            probs = self._softmax(scores / self.temperature)
            best_idx = np.random.choice(len(legal_actions), p=probs)

        return legal_actions[best_idx]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        x = x - np.max(x)  # Numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def get_action_scores(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Action]
    ) -> List[float]:
        """Get scores for all legal actions."""
        x = self.encoder.encode_state_action_batch(observation, legal_actions)
        scores = self.model.forward(x)
        return scores.tolist()


# Factory functions
def create_random_model(
    input_dim: int = 281,
    hidden_dim: int = 256,
    hidden_dim2: int = 128,
    use_torch: bool = True
) -> ActionValueNet:
    """Create a randomly initialized model."""
    return ActionValueNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_dim2=hidden_dim2,
        use_torch=use_torch
    )


def create_policy(
    model: ActionValueNet,
    temperature: float = 0.0,
    device: str = "cpu"
) -> TorchPolicy:
    """Create a policy from a model."""
    return TorchPolicy(model=model, temperature=temperature, device=device)


def load_model(filepath: str, use_torch: bool = True) -> ActionValueNet:
    """Load model from file."""
    model = ActionValueNet(use_torch=use_torch)
    model.load(filepath)
    return model
