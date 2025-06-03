import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from yaniv_neural_network_optimized import YanivNeuralNetworkOptimized, calculate_hand_value_jit, find_pairs_jit, find_runs_jit
from numba import jit, njit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import queue

class YanivGameAIOptimized:
    """Optimized Yaniv game logic with vectorized operations and parallel processing"""
    
    def __init__(self):
        self.deck = []
        self.discard_pile = []
        self.players = []
        self.current_player = 0
        self.game_over = False
        self.winner = None
        
        # Pre-allocate arrays for optimization
        self.deck_array = np.zeros((54, 2), dtype=np.int8)  # value, suit
        self.hand_arrays = None
        self.hand_sizes = None
        
    def initialize_game(self, ai_players: List[YanivNeuralNetworkOptimized]):
        """Initialize a new game with AI players"""
        self.players = []
        for i, ai in enumerate(ai_players):
            self.players.append({
                'ai': ai,
                'hand': [],
                'score': 0,
                'id': i
            })
        
        # Create and shuffle deck using numpy
        self._create_deck_optimized()
        
        # Pre-allocate hand arrays
        num_players = len(ai_players)
        self.hand_arrays = np.zeros((num_players, 10, 2), dtype=np.int8)  # Max 10 cards per hand
        self.hand_sizes = np.zeros(num_players, dtype=np.int8)
        
        # Deal initial hands
        for i, player in enumerate(self.players):
            player['hand'] = [self._draw_card() for _ in range(5)]
            self.hand_sizes[i] = 5
            self._update_hand_array(i, player['hand'])
        
        # Initialize discard pile
        self.discard_pile = [self._draw_card()]
        self.game_over = False
        self.winner = None
        self.current_player = 0
    
    def _create_deck_optimized(self):
        """Create a standard deck using numpy arrays"""
        deck_list = []
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        
        # Regular cards
        for suit_idx, suit in enumerate(suits):
            for value in range(1, 14):
                deck_list.append({'suit': suit, 'value': value, 'suit_idx': suit_idx})
        
        # Jokers
        deck_list.append({'suit': 'joker', 'value': 0, 'suit_idx': -1})
        deck_list.append({'suit': 'joker', 'value': 0, 'suit_idx': -1})
        
        # Shuffle
        random.shuffle(deck_list)
        self.deck = deck_list
        
        # Update array representation
        for i, card in enumerate(self.deck):
            self.deck_array[i, 0] = card['value']
            self.deck_array[i, 1] = card['suit_idx']
    
    def _draw_card(self) -> Dict:
        """Draw a card from the deck"""
        if self.deck:
            return self.deck.pop()
        return None
    
    def _update_hand_array(self, player_idx: int, hand: List[Dict]):
        """Update the numpy array representation of a hand"""
        for i, card in enumerate(hand):
            if i < 10:  # Max hand size
                self.hand_arrays[player_idx, i, 0] = card['value']
                self.hand_arrays[player_idx, i, 1] = card.get('suit_idx', -1)
    
    def get_hand_value_optimized(self, player_idx: int) -> int:
        """Calculate hand value using JIT-compiled function"""
        hand_size = self.hand_sizes[player_idx]
        values = self.hand_arrays[player_idx, :hand_size, 0]
        suits = self.hand_arrays[player_idx, :hand_size, 1]
        return calculate_hand_value_jit(values, suits)
    
    def get_hand_value(self, hand: List[Dict]) -> int:
        """Calculate the total value of a hand (for compatibility)"""
        total = 0
        for card in hand:
            if card['suit'] == 'joker':
                total += 0
            elif card['value'] >= 11:  # Face cards
                total += 10
            else:
                total += card['value']
        return total
    
    def get_legal_actions_optimized(self, player_idx: int) -> List[Dict]:
        """Get all legal actions using optimized operations"""
        player = self.players[player_idx]
        hand = player['hand']
        hand_size = self.hand_sizes[player_idx]
        actions = []
        
        # Get numpy arrays for this hand
        values = self.hand_arrays[player_idx, :hand_size, 0]
        suits = self.hand_arrays[player_idx, :hand_size, 1]
        
        # Single card discards
        for i in range(hand_size):
            actions.append({
                'type': 'discard_single',
                'cards': [i],
                'draw_from': 'deck'
            })
            actions.append({
                'type': 'discard_single',
                'cards': [i],
                'draw_from': 'discard'
            })
        
        # Find pairs using JIT-compiled function
        pairs = find_pairs_jit(values)
        for pair in pairs:
            actions.append({
                'type': 'discard_pair',
                'cards': list(pair),
                'draw_from': 'deck'
            })
            actions.append({
                'type': 'discard_pair',
                'cards': list(pair),
                'draw_from': 'discard'
            })
        
        # Find sets (3+ of same value) - vectorized
        unique_values, counts = np.unique(values, return_counts=True)
        for value, count in zip(unique_values, counts):
            if count >= 3:
                indices = np.where(values == value)[0].tolist()
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
        
        # Find runs using JIT-compiled function
        runs = find_runs_jit(values, suits)
        for run in runs:
            actions.append({
                'type': 'discard_run',
                'cards': run,
                'draw_from': 'deck'
            })
            actions.append({
                'type': 'discard_run',
                'cards': run,
                'draw_from': 'discard'
            })
        
        # Call Yaniv if hand value <= 7
        if self.get_hand_value_optimized(player_idx) <= 7:
            actions.append({
                'type': 'call_yaniv',
                'cards': []
            })
        
        return actions
    
    def get_legal_actions(self, player_idx: int) -> List[Dict]:
        """Get all legal actions for a player (uses optimized version)"""
        return self.get_legal_actions_optimized(player_idx)
    
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
            player['hand'].append(self._draw_card())
        elif action['draw_from'] == 'discard' and self.discard_pile:
            player['hand'].append(self.discard_pile.pop())
        
        # Update hand array
        self.hand_sizes[player_idx] = len(player['hand'])
        self._update_hand_array(player_idx, player['hand'])
        
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
        caller_value = self.get_hand_value_optimized(caller_idx)
        
        # Check if any other player has lower or equal value
        assaf = False
        for i, player in enumerate(self.players):
            if i != caller_idx:
                if self.get_hand_value_optimized(i) <= caller_value:
                    assaf = True
                    # Caller gets 30 penalty points
                    self.players[caller_idx]['score'] += 30
                    break
        
        if not assaf:
            # Caller wins the round
            for i, player in enumerate(self.players):
                player['score'] += self.get_hand_value_optimized(i)
        
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
        
        # Count opponent cards using numpy
        opponent_cards = []
        for i in range(len(self.players)):
            if i != player_idx:
                opponent_cards.append(self.hand_sizes[i])
        
        # Pad opponent cards to always have 3 values
        while len(opponent_cards) < 3:
            opponent_cards.append(0)
        
        return {
            'hand': player['hand'],
            'hand_value': self.get_hand_value_optimized(player_idx),
            'deck_size': len(self.deck),
            'opponent_cards': opponent_cards[:3],
            'last_discard': self.discard_pile[-1] if self.discard_pile else None
        }
    
    def play_game(self, ai_players: List[YanivNeuralNetworkOptimized]) -> YanivNeuralNetworkOptimized:
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
            for i, player in enumerate(self.players):
                hand_value = self.get_hand_value_optimized(i)
                if hand_value < min_value:
                    min_value = hand_value
                    self.winner = player['ai']
        
        return self.winner


# Parallel game execution functions
def play_single_game(players: Tuple[YanivNeuralNetworkOptimized, YanivNeuralNetworkOptimized]) -> YanivNeuralNetworkOptimized:
    """Play a single game between two players (for parallel execution)"""
    game = YanivGameAIOptimized()
    return game.play_game(list(players))


def play_games_batch(matchups: List[Tuple[YanivNeuralNetworkOptimized, YanivNeuralNetworkOptimized]], 
                    games_per_matchup: int) -> List[Tuple[int, int]]:
    """Play multiple games for a batch of matchups"""
    results = []
    
    for player1, player2 in matchups:
        p1_wins = 0
        p2_wins = 0
        
        for game_num in range(games_per_matchup):
            # Alternate who goes first
            if game_num % 2 == 0:
                winner = play_single_game((player1, player2))
            else:
                winner = play_single_game((player2, player1))
            
            if winner.network_id == player1.network_id:
                p1_wins += 1
            else:
                p2_wins += 1
        
        results.append((p1_wins, p2_wins))
    
    return results


class ParallelGameExecutor:
    """Execute games in parallel using multiprocessing"""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or mp.cpu_count()
    
    def play_tournament_parallel(self, population: List[YanivNeuralNetworkOptimized], 
                               games_per_matchup: int = 10) -> Dict[int, Tuple[int, int]]:
        """Play tournament in parallel and return results"""
        # Create all matchups
        matchups = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                matchups.append((population[i], population[j]))
        
        # Split matchups into chunks for parallel processing
        chunk_size = max(1, len(matchups) // (self.num_workers * 4))
        matchup_chunks = [matchups[i:i + chunk_size] for i in range(0, len(matchups), chunk_size)]
        
        # Process chunks in parallel
        results_dict = {i: (0, 0) for i in range(len(population))}  # network_id -> (wins, games)
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for chunk in matchup_chunks:
                future = executor.submit(play_games_batch, chunk, games_per_matchup)
                futures.append((chunk, future))
            
            # Collect results
            for chunk, future in futures:
                chunk_results = future.result()
                
                for (p1, p2), (p1_wins, p2_wins) in zip(chunk, chunk_results):
                    # Update wins and games for both players
                    wins1, games1 = results_dict[p1.network_id]
                    results_dict[p1.network_id] = (wins1 + p1_wins, games1 + games_per_matchup)
                    
                    wins2, games2 = results_dict[p2.network_id]
                    results_dict[p2.network_id] = (wins2 + p2_wins, games2 + games_per_matchup)
        
        return results_dict
    
    def play_tournament_parallel_with_progress(self, population: List[YanivNeuralNetworkOptimized], 
                                             games_per_matchup: int = 10) -> Dict[int, Tuple[int, int]]:
        """Play tournament in parallel with progress bar"""
        from tqdm import tqdm
        
        # Create all matchups
        matchups = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                matchups.append((population[i], population[j]))
        
        total_matchups = len(matchups)
        total_games = total_matchups * games_per_matchup
        
        # Split matchups into chunks for parallel processing
        chunk_size = max(1, len(matchups) // (self.num_workers * 4))
        matchup_chunks = [matchups[i:i + chunk_size] for i in range(0, len(matchups), chunk_size)]
        
        # Process chunks in parallel with progress bar
        results_dict = {i: (0, 0) for i in range(len(population))}  # network_id -> (wins, games)
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for chunk in matchup_chunks:
                future = executor.submit(play_games_batch, chunk, games_per_matchup)
                futures.append((chunk, future))
            
            # Collect results with progress bar
            completed_games = 0
            with tqdm(total=total_games, desc="Games", unit="game") as pbar:
                for chunk, future in futures:
                    chunk_results = future.result()
                    
                    for (p1, p2), (p1_wins, p2_wins) in zip(chunk, chunk_results):
                        # Update wins and games for both players
                        wins1, games1 = results_dict[p1.network_id]
                        results_dict[p1.network_id] = (wins1 + p1_wins, games1 + games_per_matchup)
                        
                        wins2, games2 = results_dict[p2.network_id]
                        results_dict[p2.network_id] = (wins2 + p2_wins, games2 + games_per_matchup)
                    
                    # Update progress
                    games_in_chunk = len(chunk) * games_per_matchup
                    pbar.update(games_in_chunk)
                    completed_games += games_in_chunk
                    
                    # Show additional stats in progress bar
                    pbar.set_postfix({
                        'matchups': f'{completed_games // games_per_matchup}/{total_matchups}',
                        'workers': self.num_workers
                    })
        
        return results_dict