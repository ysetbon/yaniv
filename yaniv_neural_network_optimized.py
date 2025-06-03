import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import json
from numba import jit, njit, prange
import multiprocessing as mp

class YanivNeuralNetworkOptimized:
    """Optimized neural network for Yaniv game decisions with vectorized operations"""
    
    def __init__(self, network_id: int, hidden_size: int = 64):
        self.network_id = network_id
        self.fitness = 0
        self.wins = 0
        self.games_played = 0
        
        # Input features (same as original)
        input_size = 11  # 5 cards + hand value + deck size + 3 opponents + last discard
        output_size = 16  # Action space
        
        # Initialize weights using He initialization for better convergence
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Pre-allocate arrays for batch processing
        self.batch_size = 0
        self.batch_features = None
        self.batch_hidden = None
        self.batch_output = None
    
    def forward_batch(self, game_states: List[Dict]) -> np.ndarray:
        """Vectorized forward pass for multiple game states"""
        batch_size = len(game_states)
        
        # Pre-allocate if needed
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.batch_features = np.zeros((batch_size, 11))
            self.batch_hidden = np.zeros((batch_size, self.w1.shape[1]))
            self.batch_output = np.zeros((batch_size, self.w2.shape[1]))
        
        # Extract features for all states at once
        self._extract_features_batch(game_states, self.batch_features)
        
        # Vectorized forward pass
        # Hidden layer with ReLU activation
        np.dot(self.batch_features, self.w1, out=self.batch_hidden)
        self.batch_hidden += self.b1
        np.maximum(self.batch_hidden, 0, out=self.batch_hidden)  # In-place ReLU
        
        # Output layer
        np.dot(self.batch_hidden, self.w2, out=self.batch_output)
        self.batch_output += self.b2
        
        # Batch softmax
        return self._softmax_batch(self.batch_output)
    
    def forward(self, game_state: Dict) -> np.ndarray:
        """Single forward pass (kept for compatibility)"""
        features = self._extract_features(game_state)
        
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, np.dot(features, self.w1) + self.b1)
        
        # Output layer with softmax
        output = np.dot(hidden, self.w2) + self.b2
        return self._softmax(output)
    
    def _extract_features_batch(self, game_states: List[Dict], out: np.ndarray):
        """Vectorized feature extraction for multiple game states"""
        for i, state in enumerate(game_states):
            hand = state['hand']
            
            # Encode hand cards
            for j in range(5):
                if j < len(hand):
                    out[i, j] = hand[j]['value'] / 13.0
                else:
                    out[i, j] = 0
            
            # Hand value
            out[i, 5] = state['hand_value'] / 50.0
            
            # Deck size
            out[i, 6] = state['deck_size'] / 54.0
            
            # Opponent cards
            for j, count in enumerate(state['opponent_cards'][:3]):
                out[i, 7 + j] = count / 10.0
            
            # Last discarded card
            if state['last_discard']:
                out[i, 10] = state['last_discard']['value'] / 13.0
            else:
                out[i, 10] = 0
    
    def _extract_features(self, game_state: Dict) -> np.ndarray:
        """Convert game state to neural network input features"""
        features = np.zeros(11)
        
        # Encode hand cards
        hand = game_state['hand']
        for i in range(5):
            if i < len(hand):
                features[i] = hand[i]['value'] / 13.0
        
        # Hand value
        features[5] = game_state['hand_value'] / 50.0
        
        # Deck size
        features[6] = game_state['deck_size'] / 54.0
        
        # Opponent cards
        for i, count in enumerate(game_state['opponent_cards'][:3]):
            features[7 + i] = count / 10.0
        
        # Last discarded card
        if game_state['last_discard']:
            features[10] = game_state['last_discard']['value'] / 13.0
        
        return features
    
    @staticmethod
    @njit
    def _softmax_jit(x: np.ndarray) -> np.ndarray:
        """JIT-compiled softmax for single vector"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        return self._softmax_jit(x)
    
    def _softmax_batch(self, x: np.ndarray) -> np.ndarray:
        """Vectorized softmax for batch processing"""
        # Subtract max for numerical stability
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def get_action(self, game_state: Dict, legal_actions: List[Dict]) -> Dict:
        """Select an action based on neural network output"""
        probabilities = self.forward(game_state)
        
        # Map legal actions to indices
        action_indices = []
        for i, action in enumerate(legal_actions):
            if i < len(probabilities):
                action_indices.append((i, probabilities[i]))
        
        # Sample action based on probabilities
        if action_indices:
            indices, probs = zip(*action_indices)
            probs = np.array(probs)
            probs = probs / probs.sum()  # Renormalize
            
            chosen_index = np.random.choice(indices, p=probs)
            return legal_actions[chosen_index]
        
        # Fallback: random legal action
        return random.choice(legal_actions)
    
    def get_actions_batch(self, game_states: List[Dict], legal_actions_list: List[List[Dict]]) -> List[Dict]:
        """Get actions for multiple game states at once"""
        probabilities_batch = self.forward_batch(game_states)
        
        actions = []
        for i, (probs, legal_actions) in enumerate(zip(probabilities_batch, legal_actions_list)):
            # Map legal actions to indices
            action_indices = []
            for j, action in enumerate(legal_actions):
                if j < len(probs):
                    action_indices.append((j, probs[j]))
            
            # Sample action based on probabilities
            if action_indices:
                indices, action_probs = zip(*action_indices)
                action_probs = np.array(action_probs)
                action_probs = action_probs / action_probs.sum()
                
                chosen_index = np.random.choice(indices, p=action_probs)
                actions.append(legal_actions[chosen_index])
            else:
                actions.append(random.choice(legal_actions))
        
        return actions
    
    def mutate_vectorized(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Vectorized mutation using NumPy operations"""
        # Generate mutation masks
        w1_mask = np.random.random(self.w1.shape) < mutation_rate
        b1_mask = np.random.random(self.b1.shape) < mutation_rate
        w2_mask = np.random.random(self.w2.shape) < mutation_rate
        b2_mask = np.random.random(self.b2.shape) < mutation_rate
        
        # Apply mutations
        self.w1[w1_mask] += np.random.randn(np.sum(w1_mask)) * mutation_strength
        self.b1[b1_mask] += np.random.randn(np.sum(b1_mask)) * mutation_strength
        self.w2[w2_mask] += np.random.randn(np.sum(w2_mask)) * mutation_strength
        self.b2[b2_mask] += np.random.randn(np.sum(b2_mask)) * mutation_strength
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Apply random mutations to network weights"""
        self.mutate_vectorized(mutation_rate, mutation_strength)
    
    def copy(self) -> 'YanivNeuralNetworkOptimized':
        """Create a copy of this network"""
        new_network = YanivNeuralNetworkOptimized(self.network_id, self.w1.shape[1])
        new_network.w1 = self.w1.copy()
        new_network.b1 = self.b1.copy()
        new_network.w2 = self.w2.copy()
        new_network.b2 = self.b2.copy()
        return new_network
    
    def save(self, filename: str):
        """Save network weights to file"""
        data = {
            'network_id': self.network_id,
            'w1': self.w1.tolist(),
            'b1': self.b1.tolist(),
            'w2': self.w2.tolist(),
            'b2': self.b2.tolist(),
            'fitness': self.fitness,
            'wins': self.wins,
            'games_played': self.games_played
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load(self, filename: str):
        """Load network weights from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.network_id = data['network_id']
        self.w1 = np.array(data['w1'])
        self.b1 = np.array(data['b1'])
        self.w2 = np.array(data['w2'])
        self.b2 = np.array(data['b2'])
        self.fitness = data.get('fitness', 0)
        self.wins = data.get('wins', 0)
        self.games_played = data.get('games_played', 0)


# JIT-compiled helper functions for game logic
@njit
def calculate_hand_value_jit(values: np.ndarray, suits: np.ndarray) -> int:
    """JIT-compiled hand value calculation"""
    total = 0
    for i in range(len(values)):
        if suits[i] == -1:  # Joker
            total += 0
        elif values[i] >= 11:  # Face cards
            total += 10
        else:
            total += values[i]
    return total


@njit
def find_pairs_jit(values: np.ndarray) -> List[Tuple[int, int]]:
    """JIT-compiled pair finding"""
    pairs = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] == values[j]:
                pairs.append((i, j))
    return pairs


@njit
def find_runs_jit(values: np.ndarray, suits: np.ndarray) -> List[List[int]]:
    """JIT-compiled run finding"""
    runs = []
    n = len(values)
    
    # Group by suit
    for target_suit in range(4):  # 0-3 for the four suits
        suit_cards = []
        for i in range(n):
            if suits[i] == target_suit:
                suit_cards.append((values[i], i))
        
        if len(suit_cards) >= 3:
            # Sort by value
            suit_cards.sort()
            
            # Find consecutive runs
            for start in range(len(suit_cards) - 2):
                run = [suit_cards[start]]
                for j in range(start + 1, len(suit_cards)):
                    if suit_cards[j][0] == run[-1][0] + 1:
                        run.append(suit_cards[j])
                    else:
                        break
                
                if len(run) >= 3:
                    run_indices = [card[1] for card in run]
                    runs.append(run_indices)
    
    return runs