import numpy as np
import random
from typing import List, Tuple, Dict
import json

class YanivNeuralNetwork:
    """Simple neural network for Yaniv game decisions - CPU optimized"""
    
    def __init__(self, network_id: int, hidden_size: int = 64):
        self.network_id = network_id
        self.fitness = 0
        self.wins = 0
        self.games_played = 0
        
        # Input features: 
        # - 5 cards in hand (value encoded)
        # - Current hand value
        # - Number of cards in deck
        # - Number of cards each opponent has
        # - Last discarded card
        input_size = 5 + 1 + 1 + 3 + 1  # 11 features
        
        # Output: probability for each possible action
        # - Discard single card (5 options)
        # - Discard pairs/sets (simplified to 10 options)
        # - Call Yaniv (1 option)
        output_size = 16
        
        # Initialize weights with small random values
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def forward(self, game_state: Dict) -> np.ndarray:
        """Forward pass through the network"""
        # Extract features from game state
        features = self._extract_features(game_state)
        
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, np.dot(features, self.w1) + self.b1)
        
        # Output layer with softmax
        output = np.dot(hidden, self.w2) + self.b2
        probabilities = self._softmax(output)
        
        return probabilities
    
    def _extract_features(self, game_state: Dict) -> np.ndarray:
        """Convert game state to neural network input features"""
        features = []
        
        # Encode hand cards (normalize card values to 0-1)
        hand = game_state['hand']
        for i in range(5):
            if i < len(hand):
                # Simple encoding: card value / 13
                features.append(hand[i]['value'] / 13.0)
            else:
                features.append(0)
        
        # Hand value (normalized)
        features.append(game_state['hand_value'] / 50.0)
        
        # Deck size (normalized)
        features.append(game_state['deck_size'] / 54.0)
        
        # Opponent card counts (normalized)
        for count in game_state['opponent_cards']:
            features.append(count / 10.0)
        
        # Last discarded card
        if game_state['last_discard']:
            features.append(game_state['last_discard']['value'] / 13.0)
        else:
            features.append(0)
        
        return np.array(features)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
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
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Apply random mutations to network weights"""
        # Mutate first layer
        mask = np.random.random(self.w1.shape) < mutation_rate
        self.w1[mask] += np.random.randn(np.sum(mask)) * mutation_strength
        
        mask = np.random.random(self.b1.shape) < mutation_rate
        self.b1[mask] += np.random.randn(np.sum(mask)) * mutation_strength
        
        # Mutate second layer
        mask = np.random.random(self.w2.shape) < mutation_rate
        self.w2[mask] += np.random.randn(np.sum(mask)) * mutation_strength
        
        mask = np.random.random(self.b2.shape) < mutation_rate
        self.b2[mask] += np.random.randn(np.sum(mask)) * mutation_strength
    
    def copy(self) -> 'YanivNeuralNetwork':
        """Create a copy of this network"""
        new_network = YanivNeuralNetwork(self.network_id)
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