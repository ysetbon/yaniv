import random
from typing import List, Dict, Tuple, Optional
from yaniv_neural_network import YanivNeuralNetwork

class YanivGameAI:
    """Yaniv game logic with AI player integration"""
    
    def __init__(self):
        self.deck = []
        self.discard_pile = []
        self.players = []
        self.current_player = 0
        self.game_over = False
        self.winner = None
        
    def initialize_game(self, ai_players: List[YanivNeuralNetwork]):
        """Initialize a new game with AI players"""
        self.players = []
        for i, ai in enumerate(ai_players):
            self.players.append({
                'ai': ai,
                'hand': [],
                'score': 0,
                'id': i
            })
        
        # Create and shuffle deck
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        
        # Deal initial hands
        for player in self.players:
            player['hand'] = [self.deck.pop() for _ in range(5)]
        
        # Initialize discard pile
        self.discard_pile = [self.deck.pop()]
        self.game_over = False
        self.winner = None
        self.current_player = 0
    
    def _create_deck(self) -> List[Dict]:
        """Create a standard deck of cards"""
        deck = []
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        for suit in suits:
            for value in range(1, 14):  # 1-13 (Ace to King)
                deck.append({'suit': suit, 'value': value})
        
        # Add jokers
        deck.append({'suit': 'joker', 'value': 0})
        deck.append({'suit': 'joker', 'value': 0})
        
        return deck
    
    def get_hand_value(self, hand: List[Dict]) -> int:
        """Calculate the total value of a hand"""
        total = 0
        for card in hand:
            if card['suit'] == 'joker':
                total += 0
            elif card['value'] >= 11:  # Face cards
                total += 10
            else:
                total += card['value']
        return total
    
    def get_legal_actions(self, player_idx: int) -> List[Dict]:
        """Get all legal actions for a player"""
        player = self.players[player_idx]
        hand = player['hand']
        actions = []
        
        # Single card discards
        for i, card in enumerate(hand):
            actions.append({
                'type': 'discard_single',
                'cards': [i],
                'draw_from': 'deck'  # Can choose deck or discard pile
            })
            actions.append({
                'type': 'discard_single',
                'cards': [i],
                'draw_from': 'discard'
            })
        
        # Find pairs
        for i in range(len(hand)):
            for j in range(i + 1, len(hand)):
                if hand[i]['value'] == hand[j]['value']:
                    actions.append({
                        'type': 'discard_pair',
                        'cards': [i, j],
                        'draw_from': 'deck'
                    })
                    actions.append({
                        'type': 'discard_pair',
                        'cards': [i, j],
                        'draw_from': 'discard'
                    })
        
        # Find sets (3+ of same value)
        value_groups = {}
        for i, card in enumerate(hand):
            if card['value'] not in value_groups:
                value_groups[card['value']] = []
            value_groups[card['value']].append(i)
        
        for value, indices in value_groups.items():
            if len(indices) >= 3:
                actions.append({
                    'type': 'discard_set',
                    'cards': indices,
                    'draw_from': 'deck'
                })
                actions.append({
                    'type': 'discard_set',
                    'cards': indices,
                    'draw_from': 'discard'
                })
        
        # Find runs (3+ consecutive cards of same suit)
        suit_groups = {}
        for i, card in enumerate(hand):
            if card['suit'] not in suit_groups:
                suit_groups[card['suit']] = []
            suit_groups[card['suit']].append((card['value'], i))
        
        for suit, cards in suit_groups.items():
            if len(cards) >= 3:
                cards.sort(key=lambda x: x[0])
                # Check for consecutive runs
                for start in range(len(cards) - 2):
                    run = [cards[start]]
                    for j in range(start + 1, len(cards)):
                        if cards[j][0] == run[-1][0] + 1:
                            run.append(cards[j])
                        else:
                            break
                    
                    if len(run) >= 3:
                        indices = [card[1] for card in run]
                        actions.append({
                            'type': 'discard_run',
                            'cards': indices,
                            'draw_from': 'deck'
                        })
                        actions.append({
                            'type': 'discard_run',
                            'cards': indices,
                            'draw_from': 'discard'
                        })
        
        # Call Yaniv if hand value <= 7
        if self.get_hand_value(hand) <= 7:
            actions.append({
                'type': 'call_yaniv',
                'cards': []
            })
        
        return actions
    
    def execute_action(self, player_idx: int, action: Dict) -> bool:
        """Execute a player's action"""
        player = self.players[player_idx]
        
        if action['type'] == 'call_yaniv':
            self._handle_yaniv_call(player_idx)
            return True
        
        # Discard cards
        cards_to_discard = sorted(action['cards'], reverse=True)
        discarded = []
        for idx in cards_to_discard:
            discarded.append(player['hand'].pop(idx))
        
        self.discard_pile.extend(discarded)
        
        # Draw new card
        if action['draw_from'] == 'deck' and self.deck:
            player['hand'].append(self.deck.pop())
        elif action['draw_from'] == 'discard' and self.discard_pile:
            player['hand'].append(self.discard_pile.pop())
        
        # Check if deck is empty
        if not self.deck:
            # Reshuffle discard pile
            if len(self.discard_pile) > 1:
                last_card = self.discard_pile.pop()
                self.deck = self.discard_pile
                random.shuffle(self.deck)
                self.discard_pile = [last_card]
        
        return True
    
    def _handle_yaniv_call(self, caller_idx: int):
        """Handle when a player calls Yaniv"""
        caller_value = self.get_hand_value(self.players[caller_idx]['hand'])
        
        # Check if any other player has lower or equal value
        assaf = False
        for i, player in enumerate(self.players):
            if i != caller_idx:
                if self.get_hand_value(player['hand']) <= caller_value:
                    assaf = True
                    # Caller gets 30 penalty points
                    self.players[caller_idx]['score'] += 30
                    break
        
        if not assaf:
            # Caller wins the round
            for i, player in enumerate(self.players):
                player['score'] += self.get_hand_value(player['hand'])
        
        self.game_over = True
        
        # Determine winner (lowest score)
        min_score = float('inf')
        for player in self.players:
            if player['score'] < min_score:
                min_score = player['score']
                self.winner = player['ai']
    
    def get_game_state(self, player_idx: int) -> Dict:
        """Get game state from a player's perspective"""
        player = self.players[player_idx]
        
        # Count opponent cards
        opponent_cards = []
        for i, p in enumerate(self.players):
            if i != player_idx:
                opponent_cards.append(len(p['hand']))
        
        # Pad opponent cards to always have 3 values
        while len(opponent_cards) < 3:
            opponent_cards.append(0)
        
        return {
            'hand': player['hand'],
            'hand_value': self.get_hand_value(player['hand']),
            'deck_size': len(self.deck),
            'opponent_cards': opponent_cards[:3],
            'last_discard': self.discard_pile[-1] if self.discard_pile else None
        }
    
    def play_game(self, ai_players: List[YanivNeuralNetwork]) -> YanivNeuralNetwork:
        """Play a complete game and return the winner"""
        self.initialize_game(ai_players)
        
        max_turns = 200  # Prevent infinite games
        turn_count = 0
        
        while not self.game_over and turn_count < max_turns:
            current_player = self.players[self.current_player]
            
            # Get game state and legal actions
            game_state = self.get_game_state(self.current_player)
            legal_actions = self.get_legal_actions(self.current_player)
            
            if not legal_actions:
                # No legal actions, skip turn
                self.current_player = (self.current_player + 1) % len(self.players)
                continue
            
            # AI selects action
            action = current_player['ai'].get_action(game_state, legal_actions)
            
            # Execute action
            self.execute_action(self.current_player, action)
            
            # Next player
            self.current_player = (self.current_player + 1) % len(self.players)
            turn_count += 1
        
        # If game didn't end naturally, determine winner by hand values
        if not self.winner:
            min_value = float('inf')
            for player in self.players:
                hand_value = self.get_hand_value(player['hand'])
                if hand_value < min_value:
                    min_value = hand_value
                    self.winner = player['ai']
        
        return self.winner