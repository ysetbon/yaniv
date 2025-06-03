"""
Wrapper to make Enhanced Neural Networks compatible with the existing game engine
"""

import numpy as np
from yaniv_neural_network_enhanced import EnhancedYanivNN
from yaniv_neural_network_optimized import YanivNeuralNetworkOptimized


class EnhancedNetworkWrapper:
    """Wrapper to make EnhancedYanivNN compatible with existing game interface"""
    
    def __init__(self, enhanced_network: EnhancedYanivNN):
        self.enhanced_network = enhanced_network
        self.network_id = id(enhanced_network)
    
    def forward(self, game_state):
        """Convert game state and forward through enhanced network"""
        # Convert to enhanced features format
        enhanced_features = self._convert_to_enhanced_features(game_state)
        
        # Forward pass
        return self.enhanced_network.forward(enhanced_features)
    
    def _convert_to_enhanced_features(self, game_state):
        """Convert original game state to enhanced feature format"""
        features = []
        
        # Basic hand features (5)
        hand = game_state.get('hand', [])
        for i in range(5):
            if i < len(hand):
                card_value = hand[i]['value'] if isinstance(hand[i], dict) else hand[i]
                features.append(card_value / 13.0)
            else:
                features.append(0.0)
        
        # Hand value (1)
        hand_value = game_state.get('hand_value', 0)
        features.append(hand_value / 50.0)
        
        # Suit distribution (4) - simplified for compatibility
        features.extend([0.25, 0.25, 0.25, 0.25])  # Assume even distribution
        
        # Rank distribution (13) - simplified
        features.extend([0.08] * 13)  # Default distribution (13 features)
        
        # Potential combinations (3) - simplified
        features.extend([0.1, 0.1, 0.1])  # Default potential
        
        # Game state features (2)
        features.append(game_state.get('deck_size', 52) / 52.0)
        last_discard = game_state.get('last_discard')
        if last_discard:
            discard_value = last_discard['value'] if isinstance(last_discard, dict) else last_discard
            features.append(discard_value / 13.0)
        else:
            features.append(0.0)
        
        # Opponent information (3)
        opponent_cards = game_state.get('opponent_cards', [])
        for i in range(3):
            if i < len(opponent_cards):
                features.append(opponent_cards[i] / 10.0)
            else:
                features.append(0.0)
        
        # Game phase (1)
        deck_size = game_state.get('deck_size', 52)
        if deck_size > 35:
            features.append(1.0)  # Early
        elif deck_size > 15:
            features.append(0.5)  # Mid
        else:
            features.append(0.0)  # Late
        
        # Can call Yaniv (1)
        features.append(1.0 if hand_value <= 7 else 0.0)
        
        # Add 2 padding features to reach 35 total
        features.append(0.0)  # Padding feature 1
        features.append(0.0)  # Padding feature 2
        
        # Ensure we have exactly 35 features
        while len(features) < 35:
            features.append(0.0)
        
        # Trim if too many (shouldn't happen)
        features = features[:35]
        
        return np.array(features)
    
    def select_action(self, game_state, legal_actions):
        """Select action from legal actions using enhanced network"""
        if not legal_actions:
            return None
        
        # Get action probabilities
        action_probs = self.forward(game_state)
        
        # Map to legal actions
        legal_indices = []
        legal_probs = []
        
        for i, action in enumerate(legal_actions):
            if i < len(action_probs):
                legal_indices.append(i)
                legal_probs.append(action_probs[i])
        
        if not legal_probs:
            # Fallback to random
            return np.random.choice(legal_actions)
        
        # Normalize probabilities
        legal_probs = np.array(legal_probs)
        legal_probs = legal_probs / np.sum(legal_probs)
        
        # Sample action
        selected_idx = np.random.choice(len(legal_probs), p=legal_probs)
        return legal_actions[legal_indices[selected_idx]]


def create_simple_game():
    """Create a simplified game engine for training"""
    
    class SimpleYanivGame:
        def __init__(self):
            self.players = []
            self.current_player = 0
            self.game_over = False
            self.winner = None
            self.deck_size = 52
            self.turn_count = 0
        
        def play_game(self, networks):
            """Play a game between networks"""
            if len(networks) != 2:
                networks = networks[:2]  # Take first 2
            
            # Wrap enhanced networks
            wrapped_players = []
            for net in networks:
                if isinstance(net, EnhancedYanivNN):
                    wrapped_players.append(EnhancedNetworkWrapper(net))
                else:
                    wrapped_players.append(net)
            
            # Initialize hands
            hands = [
                [np.random.randint(1, 14) for _ in range(5)],  # Player 1 hand
                [np.random.randint(1, 14) for _ in range(5)]   # Player 2 hand
            ]
            
            self.turn_count = 0
            max_turns = 50
            
            while self.turn_count < max_turns:
                current_idx = self.turn_count % 2
                current_player = wrapped_players[current_idx]
                current_hand = hands[current_idx]
                
                # Calculate hand value
                hand_value = sum(min(card, 10) for card in current_hand)
                
                # Check if can call Yaniv
                if hand_value <= 7:
                    # 30% chance to call Yaniv if possible
                    if np.random.random() < 0.3:
                        self.winner = current_idx
                        return current_idx
                
                # Simulate game state
                game_state = {
                    'hand': [{'value': card} for card in current_hand],
                    'hand_value': hand_value,
                    'deck_size': max(10, 52 - self.turn_count * 2),
                    'opponent_cards': [len(hands[1-current_idx])],
                    'last_discard': {'value': np.random.randint(1, 14)}
                }
                
                # Simple legal actions
                legal_actions = [
                    {'action': 'discard', 'cards': [0], 'draw_from': 'deck'},
                    {'action': 'discard', 'cards': [1], 'draw_from': 'deck'}
                ]
                
                # Player makes decision
                if hasattr(current_player, 'select_action'):
                    action = current_player.select_action(game_state, legal_actions)
                else:
                    action = legal_actions[0]  # Default action
                
                # Apply action (simplified)
                if action and 'cards' in action:
                    card_to_remove = action['cards'][0]
                    if card_to_remove < len(current_hand):
                        current_hand.pop(card_to_remove)
                        current_hand.append(np.random.randint(1, 14))  # Draw new card
                
                self.turn_count += 1
            
            # Game ends, determine winner by hand value
            hand1_value = sum(min(card, 10) for card in hands[0])
            hand2_value = sum(min(card, 10) for card in hands[1])
            
            return 0 if hand1_value <= hand2_value else 1
    
    return SimpleYanivGame()


# Update the genetic algorithm to use the simple game
def play_simple_match(network1, network2):
    """Play a simple match between two networks"""
    game = create_simple_game()
    
    # Play 2 games (alternate starting player)
    wins = 0
    
    for game_num in range(2):
        if game_num == 0:
            players = [network1, network2]
            winner = game.play_game(players)
            if winner == 0:
                wins += 1
        else:
            players = [network2, network1]
            winner = game.play_game(players)
            if winner == 1:
                wins += 1
    
    return wins