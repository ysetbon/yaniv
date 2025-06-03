import numpy as np
from typing import List, Dict, Tuple, Optional
from numba import jit
import json

class EnhancedYanivNN:
    """Enhanced neural network with deeper architecture and skip connections"""
    
    def __init__(self, input_size: int = 35, hidden_sizes: List[int] = [128, 64, 32], 
                 output_size: int = 16, dropout_rate: float = 0.2):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Initialize layers with He initialization
        self.layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append({
                'weights': np.random.randn(prev_size, hidden_size) * np.sqrt(2.0 / prev_size),
                'bias': np.zeros((1, hidden_size))
            })
            prev_size = hidden_size
        
        # Output layer
        self.layers.append({
            'weights': np.random.randn(prev_size, output_size) * np.sqrt(2.0 / prev_size),
            'bias': np.zeros((1, output_size))
        })
        
        # Skip connection from input to output
        self.skip_weights = np.random.randn(input_size, output_size) * 0.01
        
        # For tracking activations during forward pass
        self.activations = []
        
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass with optional dropout for training"""
        self.activations = [x]
        
        # Hidden layers with ReLU
        for i, layer in enumerate(self.layers[:-1]):
            z = np.dot(self.activations[-1], layer['weights']) + layer['bias']
            a = np.maximum(0, z)  # ReLU
            
            # Apply dropout during training
            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                a = a * mask / (1 - self.dropout_rate)
            
            self.activations.append(a)
        
        # Output layer with skip connection
        output_layer = self.layers[-1]
        z_out = np.dot(self.activations[-1], output_layer['weights']) + output_layer['bias']
        
        # Add skip connection
        z_out += np.dot(x, self.skip_weights) * 0.1  # Scale down skip connection
        
        # Softmax activation
        exp_z = np.exp(z_out - np.max(z_out, axis=-1, keepdims=True))
        output = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        
        return output
    
    def get_weights_flat(self) -> np.ndarray:
        """Flatten all weights for genetic algorithm operations"""
        weights = []
        
        for layer in self.layers:
            weights.extend(layer['weights'].flatten())
            weights.extend(layer['bias'].flatten())
        
        weights.extend(self.skip_weights.flatten())
        
        return np.array(weights)
    
    def set_weights_from_flat(self, flat_weights: np.ndarray):
        """Reconstruct weights from flat array"""
        idx = 0
        
        for layer in self.layers:
            w_size = layer['weights'].size
            b_size = layer['bias'].size
            
            layer['weights'] = flat_weights[idx:idx + w_size].reshape(layer['weights'].shape)
            idx += w_size
            
            layer['bias'] = flat_weights[idx:idx + b_size].reshape(layer['bias'].shape)
            idx += b_size
        
        skip_size = self.skip_weights.size
        self.skip_weights = flat_weights[idx:idx + skip_size].reshape(self.skip_weights.shape)
    
    def save(self, filename: str):
        """Save network to file"""
        network_data = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'layers': [
                {
                    'weights': layer['weights'].tolist(),
                    'bias': layer['bias'].tolist()
                }
                for layer in self.layers
            ],
            'skip_weights': self.skip_weights.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(network_data, f)
    
    @classmethod
    def load(cls, filename: str) -> 'EnhancedYanivNN':
        """Load network from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        network = cls(
            input_size=data['input_size'],
            hidden_sizes=data['hidden_sizes'],
            output_size=data['output_size']
        )
        
        for i, layer_data in enumerate(data['layers']):
            network.layers[i]['weights'] = np.array(layer_data['weights'])
            network.layers[i]['bias'] = np.array(layer_data['bias'])
        
        network.skip_weights = np.array(data['skip_weights'])
        
        return network


class AdvancedFeatureExtractor:
    """Extract rich features from game state"""
    
    @staticmethod
    def extract_features(game_state: Dict, move_history: Optional[List] = None) -> np.ndarray:
        """Extract comprehensive features from game state"""
        features = []
        
        # Basic hand features (5)
        hand_cards = game_state['hand']
        for i in range(5):
            if i < len(hand_cards):
                features.append(hand_cards[i]['value'] / 13.0)
            else:
                features.append(0.0)
        
        # Hand value (1)
        hand_value = sum(card['value'] for card in hand_cards)
        features.append(hand_value / 50.0)
        
        # Suit distribution (4)
        suit_counts = [0, 0, 0, 0]
        for card in hand_cards:
            suit_counts[card['suit']] += 1
        features.extend([c / 5.0 for c in suit_counts])
        
        # Rank distribution (13)
        rank_counts = [0] * 13
        for card in hand_cards:
            rank_counts[card['value'] - 1] += 1
        features.extend([c / 5.0 for c in rank_counts])
        
        # Potential combinations (3)
        features.append(count_potential_sets(hand_cards) / 5.0)
        features.append(count_potential_runs(hand_cards) / 5.0)
        features.append(count_potential_pairs(hand_cards) / 10.0)
        
        # Game state features (4)
        features.append(game_state.get('deck_size', 52) / 52.0)
        features.append(game_state.get('discard_top', 0) / 13.0)
        
        # Opponent information (3)
        for i in range(3):
            if i < len(game_state.get('opponents', [])):
                features.append(game_state['opponents'][i]['cards'] / 10.0)
            else:
                features.append(0.0)
        
        # Game phase indicator
        if game_state.get('deck_size', 52) > 35:
            features.append(1.0)  # Early game
        elif game_state.get('deck_size', 52) > 15:
            features.append(0.5)  # Mid game
        else:
            features.append(0.0)  # Late game
        
        # Risk assessment - can we call Yaniv?
        features.append(1.0 if hand_value <= 7 else 0.0)
        
        return np.array(features)


@jit(nopython=True)
def count_potential_sets(cards) -> int:
    """Count potential sets in hand"""
    value_counts = {}
    for card in cards:
        val = card['value']
        if val not in value_counts:
            value_counts[val] = 0
        value_counts[val] += 1
    
    count = 0
    for val, cnt in value_counts.items():
        if cnt >= 2:
            count += 1
    return count


@jit(nopython=True)
def count_potential_runs(cards) -> int:
    """Count potential runs in hand"""
    # Group by suit
    suit_cards = {0: [], 1: [], 2: [], 3: []}
    for card in cards:
        suit_cards[card['suit']].append(card['value'])
    
    count = 0
    for suit, values in suit_cards.items():
        if len(values) >= 2:
            values.sort()
            # Check for consecutive values
            for i in range(len(values) - 1):
                if values[i+1] - values[i] == 1:
                    count += 1
                    break
    
    return count


@jit(nopython=True)
def count_potential_pairs(cards) -> int:
    """Count pairs in hand"""
    value_counts = {}
    for card in cards:
        val = card['value']
        if val not in value_counts:
            value_counts[val] = 0
        value_counts[val] += 1
    
    pairs = 0
    for cnt in value_counts.values():
        pairs += cnt // 2
    
    return pairs


class HallOfFame:
    """Maintain a collection of best performing networks"""
    
    def __init__(self, size: int = 20):
        self.size = size
        self.champions = []
    
    def update(self, network: EnhancedYanivNN, fitness: float, generation: int, 
               metadata: Optional[Dict] = None):
        """Add a new champion if it's good enough"""
        champion_data = {
            'network': network,
            'fitness': fitness,
            'generation': generation,
            'metadata': metadata or {}
        }
        
        self.champions.append(champion_data)
        
        # Sort by fitness and keep only the best
        self.champions.sort(key=lambda x: x['fitness'], reverse=True)
        self.champions = self.champions[:self.size]
    
    def get_opponents(self, n: int = 5) -> List[EnhancedYanivNN]:
        """Get a diverse set of opponents from the hall of fame"""
        if len(self.champions) <= n:
            return [champ['network'] for champ in self.champions]
        
        # Select champions from different generations for diversity
        selected = []
        
        # Always include the best
        selected.append(self.champions[0]['network'])
        
        # Add others with spacing
        step = len(self.champions) // (n - 1)
        for i in range(1, n):
            idx = min(i * step, len(self.champions) - 1)
            selected.append(self.champions[idx]['network'])
        
        return selected
    
    def save(self, filename: str):
        """Save hall of fame to file"""
        hof_data = {
            'size': self.size,
            'champions': [
                {
                    'fitness': champ['fitness'],
                    'generation': champ['generation'],
                    'metadata': champ['metadata'],
                    'network_file': f"{filename}_network_{i}.json"
                }
                for i, champ in enumerate(self.champions)
            ]
        }
        
        # Save metadata
        with open(filename, 'w') as f:
            json.dump(hof_data, f)
        
        # Save individual networks
        for i, champ in enumerate(self.champions):
            champ['network'].save(f"{filename}_network_{i}.json")
    
    def load(self, filename: str):
        """Load hall of fame from file"""
        with open(filename, 'r') as f:
            hof_data = json.load(f)
        
        self.size = hof_data['size']
        self.champions = []
        
        for champ_data in hof_data['champions']:
            network = EnhancedYanivNN.load(champ_data['network_file'])
            self.champions.append({
                'network': network,
                'fitness': champ_data['fitness'],
                'generation': champ_data['generation'],
                'metadata': champ_data['metadata']
            })


class EnsembleAI:
    """Combine multiple neural networks for better performance"""
    
    def __init__(self, networks: List[EnhancedYanivNN], weights: Optional[np.ndarray] = None):
        self.networks = networks
        self.weights = weights if weights is not None else np.ones(len(networks)) / len(networks)
        self.performance_history = [[] for _ in networks]
    
    def predict(self, game_state: Dict) -> np.ndarray:
        """Get ensemble prediction"""
        features = AdvancedFeatureExtractor.extract_features(game_state)
        predictions = []
        
        for network, weight in zip(self.networks, self.weights):
            pred = network.forward(features)
            predictions.append(pred * weight)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Normalize
        ensemble_pred = ensemble_pred / np.sum(ensemble_pred)
        
        return ensemble_pred
    
    def update_weights(self, network_idx: int, success: bool):
        """Update network weights based on performance"""
        # Track performance
        self.performance_history[network_idx].append(1.0 if success else 0.0)
        
        # Update weights every 10 predictions
        if sum(len(h) for h in self.performance_history) % 10 == 0:
            # Calculate recent performance for each network
            performances = []
            for history in self.performance_history:
                if len(history) >= 10:
                    recent_perf = np.mean(history[-10:])
                else:
                    recent_perf = 0.5  # Default
                performances.append(recent_perf)
            
            # Update weights proportional to performance
            performances = np.array(performances)
            self.weights = performances / np.sum(performances)
            
            # Add small epsilon to prevent zero weights
            self.weights = 0.9 * self.weights + 0.1 / len(self.networks)
    
    def save(self, filename: str):
        """Save ensemble configuration"""
        ensemble_data = {
            'num_networks': len(self.networks),
            'weights': self.weights.tolist(),
            'network_files': [f"{filename}_network_{i}.json" for i in range(len(self.networks))]
        }
        
        # Save metadata
        with open(filename, 'w') as f:
            json.dump(ensemble_data, f)
        
        # Save individual networks
        for i, network in enumerate(self.networks):
            network.save(f"{filename}_network_{i}.json")
    
    @classmethod
    def load(cls, filename: str) -> 'EnsembleAI':
        """Load ensemble from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        networks = []
        for network_file in data['network_files']:
            networks.append(EnhancedYanivNN.load(network_file))
        
        return cls(networks, np.array(data['weights']))