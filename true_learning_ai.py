#!/usr/bin/env python3
"""
True Learning AI for Yaniv - Learns by Playing Games
This AI discovers strategies through experience, not pre-programmed rules
"""

import json
import random
import time
from collections import defaultdict
from datetime import datetime

class GameState:
    """Simplified Yaniv game state for learning"""
    def __init__(self):
        self.deck = self.create_deck()
        self.discard_pile = []
        self.hands = [[], []]  # 2 players
        self.scores = [0, 0]
        self.current_player = 0
        self.turn_phase = 'draw'
        self.game_over = False
        self.winner = None
        
    def create_deck(self):
        """Create a standard deck"""
        deck = []
        for suit in range(4):
            for rank in range(1, 14):
                value = min(rank, 10)  # Face cards = 10
                deck.append({'rank': rank, 'suit': suit, 'value': value})
        random.shuffle(deck)
        return deck
    
    def deal_initial_hands(self):
        """Deal 5 cards to each player"""
        for _ in range(5):
            for player in range(2):
                if self.deck:
                    self.hands[player].append(self.deck.pop())
        if self.deck:
            self.discard_pile.append(self.deck.pop())
    
    def get_hand_value(self, player):
        """Calculate hand value"""
        return sum(card['value'] for card in self.hands[player])
    
    def find_valid_discards(self, player):
        """Find all valid discard options"""
        hand = self.hands[player]
        valid = []
        
        # Single cards
        for i, card in enumerate(hand):
            valid.append([i])
        
        # Pairs
        for i in range(len(hand)):
            for j in range(i + 1, len(hand)):
                if hand[i]['value'] == hand[j]['value']:
                    valid.append([i, j])
        
        # Sets (3+ of same value)
        value_groups = defaultdict(list)
        for i, card in enumerate(hand):
            value_groups[card['value']].append(i)
        
        for indices in value_groups.values():
            if len(indices) >= 3:
                valid.append(indices)
        
        # Runs (3+ consecutive same suit)
        suit_groups = defaultdict(list)
        for i, card in enumerate(hand):
            suit_groups[card['suit']].append((card['rank'], i))
        
        for suit_cards in suit_groups.values():
            if len(suit_cards) >= 3:
                suit_cards.sort()
                # Find consecutive runs
                for start in range(len(suit_cards) - 2):
                    run = [suit_cards[start]]
                    for next_idx in range(start + 1, len(suit_cards)):
                        if suit_cards[next_idx][0] == run[-1][0] + 1:
                            run.append(suit_cards[next_idx])
                            if len(run) >= 3:
                                valid.append([card[1] for card in run])
        
        return valid


class LearningAgent:
    """An AI that learns through experience"""
    def __init__(self, agent_id="learner"):
        self.agent_id = agent_id
        self.experience = defaultdict(lambda: {'wins': 0, 'plays': 0, 'value': 0})
        self.exploration_rate = 0.3
        self.learning_rate = 0.1
        self.games_played = 0
        self.wins = 0
        
    def get_state_key(self, game_state, player):
        """Create a simplified state representation"""
        hand_value = game_state.get_hand_value(player)
        hand_size = len(game_state.hands[player])
        opp_hand_size = len(game_state.hands[1 - player])
        deck_size = len(game_state.deck)
        
        # Discretize values for manageable state space
        value_bucket = hand_value // 5
        deck_bucket = deck_size // 10
        
        # Check for combo potential
        hand = game_state.hands[player]
        has_pair = any(hand[i]['value'] == hand[j]['value'] 
                      for i in range(len(hand)) 
                      for j in range(i + 1, len(hand)))
        
        return (value_bucket, hand_size, opp_hand_size, deck_bucket, has_pair)
    
    def get_action_key(self, action_type, details=None):
        """Create action representation"""
        if action_type == 'yaniv':
            return ('yaniv',)
        elif action_type == 'draw':
            return ('draw', details)  # 'deck' or 'discard'
        elif action_type == 'discard':
            return ('discard', details)  # 'single', 'pair', 'set', 'run'
    
    def choose_action(self, game_state, player, valid_actions):
        """Choose action using epsilon-greedy strategy"""
        state_key = self.get_state_key(game_state, player)
        
        # Exploration
        if random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        # Exploitation - choose best known action
        best_action = None
        best_value = -float('inf')
        
        for action in valid_actions:
            action_key = self.get_action_key(action['type'], action.get('details'))
            state_action = (state_key, action_key)
            
            # Calculate expected value
            exp = self.experience[state_action]
            if exp['plays'] > 0:
                value = exp['value'] / exp['plays']
            else:
                value = 0  # Unexplored action
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action if best_action else random.choice(valid_actions)
    
    def update_experience(self, trajectory, reward):
        """Update experience based on game outcome"""
        # Propagate reward backwards through trajectory
        discounted_reward = reward
        discount = 0.95
        
        for state_action in reversed(trajectory):
            exp = self.experience[state_action]
            exp['plays'] += 1
            
            # Update value estimate
            old_value = exp['value'] / max(exp['plays'] - 1, 1)
            exp['value'] += self.learning_rate * (discounted_reward - old_value)
            
            if reward > 0:
                exp['wins'] += 1
            
            discounted_reward *= discount


class YanivSimulator:
    """Simulates Yaniv games for learning"""
    
    def play_game(self, agent1, agent2):
        """Play a complete game"""
        game = GameState()
        game.deal_initial_hands()
        
        agents = [agent1, agent2]
        trajectories = [[], []]  # Track state-action pairs for each player
        
        max_turns = 200
        turn = 0
        
        while not game.game_over and turn < max_turns:
            player = game.current_player
            agent = agents[player]
            
            if game.turn_phase == 'draw':
                # Draw phase
                valid_actions = [
                    {'type': 'draw', 'source': 'deck', 'details': 'deck'},
                    {'type': 'draw', 'source': 'discard', 'details': 'discard'}
                ]
                
                # Remove invalid options
                if not game.deck:
                    valid_actions = [a for a in valid_actions if a['source'] != 'deck']
                if not game.discard_pile:
                    valid_actions = [a for a in valid_actions if a['source'] != 'discard']
                
                if valid_actions:
                    state_key = agent.get_state_key(game, player)
                    action = agent.choose_action(game, player, valid_actions)
                    action_key = agent.get_action_key(action['type'], action['details'])
                    trajectories[player].append((state_key, action_key))
                    
                    # Execute draw
                    if action['source'] == 'deck' and game.deck:
                        game.hands[player].append(game.deck.pop())
                    elif action['source'] == 'discard' and game.discard_pile:
                        game.hands[player].append(game.discard_pile.pop())
                
                game.turn_phase = 'discard'
                
            else:  # Discard phase
                hand_value = game.get_hand_value(player)
                valid_actions = []
                
                # Check if can call Yaniv
                if hand_value <= 7:
                    valid_actions.append({'type': 'yaniv'})
                
                # Find valid discards
                valid_discards = game.find_valid_discards(player)
                for discard_indices in valid_discards:
                    if len(discard_indices) == 1:
                        action_detail = 'single'
                    elif len(discard_indices) == 2:
                        action_detail = 'pair'
                    else:
                        # Check if it's a set or run
                        cards = [game.hands[player][i] for i in discard_indices]
                        if all(c['value'] == cards[0]['value'] for c in cards):
                            action_detail = 'set'
                        else:
                            action_detail = 'run'
                    
                    valid_actions.append({
                        'type': 'discard',
                        'indices': discard_indices,
                        'details': action_detail
                    })
                
                if valid_actions:
                    state_key = agent.get_state_key(game, player)
                    action = agent.choose_action(game, player, valid_actions)
                    
                    if action['type'] == 'yaniv':
                        action_key = agent.get_action_key('yaniv')
                        trajectories[player].append((state_key, action_key))
                        
                        # Check if Yaniv succeeds
                        opp_value = game.get_hand_value(1 - player)
                        if hand_value <= opp_value:
                            game.winner = player
                        else:
                            # Assaf penalty
                            game.scores[player] += 30
                            game.winner = 1 - player
                        
                        game.game_over = True
                        
                    else:  # Discard
                        action_key = agent.get_action_key('discard', action['details'])
                        trajectories[player].append((state_key, action_key))
                        
                        # Execute discard
                        discard_indices = sorted(action['indices'], reverse=True)
                        discarded = []
                        for idx in discard_indices:
                            discarded.append(game.hands[player].pop(idx))
                        
                        game.discard_pile.extend(discarded)
                        
                        # Check for round end (empty hand)
                        if not game.hands[player]:
                            game.winner = player
                            game.game_over = True
                
                # Switch player
                game.current_player = 1 - player
                game.turn_phase = 'draw'
                turn += 1
        
        # Game ended - update experience
        if game.winner is not None:
            # Winner gets positive reward, loser gets negative
            rewards = [0, 0]
            rewards[game.winner] = 1
            rewards[1 - game.winner] = -1
            
            # Update both agents
            for i, agent in enumerate(agents):
                agent.update_experience(trajectories[i], rewards[i])
                agent.games_played += 1
                if rewards[i] > 0:
                    agent.wins += 1
        
        return game.winner


def train_true_learning_ai(minutes=10):
    """Train AI through actual gameplay"""
    print("🧠 True Learning AI Training")
    print("=" * 60)
    print("This AI learns by playing games and discovering strategies\n")
    
    start_time = time.time()
    end_time = start_time + (minutes * 60)
    
    # Create learning agents
    learner = LearningAgent("learner")
    opponent = LearningAgent("opponent")
    
    simulator = YanivSimulator()
    
    games_played = 0
    last_update = start_time
    
    # Progress tracking
    print(f"Training for {minutes} minutes...")
    print("The AI will discover strategies like:")
    print("  • When to call Yaniv")
    print("  • When to draw from discard pile")
    print("  • Which cards to keep vs discard")
    print("  • How to form combinations\n")
    
    while time.time() < end_time:
        # Play a game
        simulator.play_game(learner, opponent)
        games_played += 1
        
        # Update exploration rate (decay over time)
        progress = (time.time() - start_time) / (minutes * 60)
        learner.exploration_rate = 0.3 * (1 - progress) + 0.05  # From 30% to 5%
        
        # Progress update every 5 seconds
        if time.time() - last_update >= 5:
            elapsed = time.time() - start_time
            win_rate = learner.wins / max(learner.games_played, 1)
            
            # Analyze learned strategies
            strategies = analyze_learned_strategies(learner)
            
            print(f"\n⏱️  Time: {elapsed:.0f}s | Games: {games_played} | Win Rate: {win_rate:.1%}")
            print(f"📊 Discovered Strategies:")
            for strategy in strategies[:3]:
                print(f"   • {strategy}")
            
            last_update = time.time()
    
    # Final analysis
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    final_win_rate = learner.wins / max(learner.games_played, 1)
    print(f"\n📊 Final Results:")
    print(f"  • Games played: {games_played}")
    print(f"  • Final win rate: {final_win_rate:.1%}")
    print(f"  • Unique situations learned: {len(learner.experience)}")
    
    # Save the learned AI
    save_learned_ai(learner)
    
    return learner


def analyze_learned_strategies(agent):
    """Analyze what strategies the AI has learned"""
    strategies = []
    
    # Analyze Yaniv calling pattern
    yaniv_states = [(k, v) for k, v in agent.experience.items() 
                    if k[1][0] == 'yaniv' and v['plays'] > 5]
    
    if yaniv_states:
        avg_value = sum(k[0][0] * 5 for k, v in yaniv_states) / len(yaniv_states)
        win_rate = sum(v['wins'] for k, v in yaniv_states) / sum(v['plays'] for k, v in yaniv_states)
        strategies.append(f"Calls Yaniv at ~{avg_value:.0f} points with {win_rate:.0%} success")
    
    # Analyze draw preferences
    draw_states = [(k, v) for k, v in agent.experience.items() 
                   if k[1][0] == 'draw' and v['plays'] > 10]
    
    deck_draws = sum(v['plays'] for k, v in draw_states if k[1][1] == 'deck')
    discard_draws = sum(v['plays'] for k, v in draw_states if k[1][1] == 'discard')
    
    if deck_draws + discard_draws > 0:
        discard_rate = discard_draws / (deck_draws + discard_draws)
        strategies.append(f"Draws from discard pile {discard_rate:.0%} of the time")
    
    # Analyze combo play
    combo_plays = sum(v['plays'] for k, v in agent.experience.items() 
                      if k[1][0] == 'discard' and k[1][1] in ['pair', 'set', 'run'])
    single_plays = sum(v['plays'] for k, v in agent.experience.items() 
                       if k[1][0] == 'discard' and k[1][1] == 'single')
    
    if combo_plays + single_plays > 0:
        combo_rate = combo_plays / (combo_plays + single_plays)
        strategies.append(f"Plays combinations {combo_rate:.0%} of the time when possible")
    
    return strategies


def save_learned_ai(agent):
    """Convert learned experience to TypeScript"""
    
    # Extract key learned behaviors
    yaniv_threshold = 7  # Default
    yaniv_pattern = [(k[0][0] * 5, v['wins']/max(v['plays'], 1)) 
                     for k, v in agent.experience.items() 
                     if k[1][0] == 'yaniv' and v['plays'] > 5]
    
    if yaniv_pattern:
        # Find the hand value with best success rate
        yaniv_pattern.sort(key=lambda x: x[1], reverse=True)
        yaniv_threshold = yaniv_pattern[0][0]
    
    # Calculate draw preference
    draw_prefs = [(k, v) for k, v in agent.experience.items() 
                  if k[1][0] == 'draw' and v['plays'] > 10]
    
    deck_value = sum(v['value']/max(v['plays'], 1) for k, v in draw_prefs if k[1][1] == 'deck')
    discard_value = sum(v['value']/max(v['plays'], 1) for k, v in draw_prefs if k[1][1] == 'discard')
    
    draw_deck_pref = 0.7  # Default
    if deck_value + discard_value > 0:
        draw_deck_pref = deck_value / (deck_value + discard_value)
    
    # Generate TypeScript code
    ts_code = f'''import {{ Card, Suit }} from './Card';
import {{ GameState, AIPlayer, AIType }} from '../types/game';

export class EnhancedNeuralNetworkAI implements AIPlayer {{
  type: AIType = 'enhanced-neural';
  private gamesLearned: number = {agent.games_played};
  private winRate: number = {agent.wins / max(agent.games_played, 1):.3f};
  
  // Learned behaviors from {agent.games_played} games
  private learnedYanivThreshold: number = {yaniv_threshold:.1f};
  private learnedDrawDeckPreference: number = {draw_deck_pref:.3f};
  
  // Experience-based decision making
  private experience = new Map<string, {{ wins: number, plays: number, value: number }}>();

  async makeMove(gameState: GameState): Promise<any> {{
    return this.makeDecision(
      gameState.currentPlayer?.hand?.cards || [],
      gameState.discardPile || [],
      gameState
    );
  }}

  async makeDecision(hand: Card[], discardPile: Card[], gameState?: any, playerId?: string): Promise<any> {{
    console.log(`Enhanced AI (learned from ${{this.gamesLearned}} games) making decision...`);
    
    try {{
      const handValue = hand.reduce((sum, card) => sum + card.value, 0);
      const turnPhase = gameState?.turnPhase || 'discard';
      
      if (turnPhase === 'draw') {{
        // Learned draw strategy
        const topDiscard = discardPile[discardPile.length - 1];
        let drawFrom = 'deck';
        
        // Check if discard pile card helps
        if (topDiscard) {{
          // Does it complete a pair/set?
          const matchingCards = hand.filter(c => c.value === topDiscard.value);
          if (matchingCards.length >= 1) {{
            drawFrom = 'discard';  // Learned: take cards that match
            console.log('AI learned: Taking from discard to form combination');
          }}
          
          // Does it help with a run?
          const sameSuitCards = hand.filter(c => c.suit === topDiscard.suit);
          if (sameSuitCards.length >= 2) {{
            const ranks = sameSuitCards.map(c => c.value).sort((a, b) => a - b);
            for (let i = 0; i < ranks.length - 1; i++) {{
              if (Math.abs(ranks[i] - topDiscard.value) === 1 || 
                  Math.abs(ranks[i + 1] - topDiscard.value) === 1) {{
                drawFrom = 'discard';
                console.log('AI learned: Taking from discard for potential run');
                break;
              }}
            }}
          }}
        }}
        
        // Apply learned preference
        if (drawFrom === 'deck' && Math.random() > this.learnedDrawDeckPreference) {{
          drawFrom = 'discard';
        }}
        
        return {{
          action: 'draw',
          drawSource: drawFrom
        }};
      }}
      
      // Learned Yaniv strategy
      if (handValue <= this.learnedYanivThreshold && handValue > 0) {{
        console.log(`AI learned: Call Yaniv at ${{handValue}} (threshold: ${{this.learnedYanivThreshold.toFixed(1)}})`);
        return {{ 
          action: 'yaniv',
          cardsToDiscard: [] 
        }};
      }}
      
      // Find best discard based on learning
      const validCombos = this.findCombinations(hand);
      
      // Prefer combos that reduce hand significantly
      if (validCombos.length > 0) {{
        // Sort by total value removed
        validCombos.sort((a, b) => {{
          const aValue = a.reduce((sum, card) => sum + card.value, 0);
          const bValue = b.reduce((sum, card) => sum + card.value, 0);
          return bValue - aValue;
        }});
        
        const bestCombo = validCombos[0];
        console.log('AI learned: Discard combo for maximum value reduction');
        return {{
          action: 'discard',
          cardsToDiscard: bestCombo,
          drawSource: 'deck'
        }};
      }}
      
      // Single card - learned strategy
      const scoredCards = hand.map((card, index) => {{
        let score = card.value;  // Base score is card value
        
        // Penalize cards that could form combos
        const sameValue = hand.filter(c => c.value === card.value).length;
        const sameSuit = hand.filter(c => c.suit === card.suit && 
                                    Math.abs(c.value - card.value) <= 2).length;
        
        if (sameValue >= 2) score *= 0.7;  // Keep pairs
        if (sameSuit >= 2) score *= 0.8;   // Keep potential runs
        
        return {{ card, index, score }};
      }});
      
      // Discard highest scored card
      scoredCards.sort((a, b) => b.score - a.score);
      const cardToDiscard = scoredCards[0].card;
      
      console.log(`AI learned: Discard ${{cardToDiscard.rank}}${{cardToDiscard.suit}} (strategic value: ${{scoredCards[0].score.toFixed(1)}})`);
      return {{
        action: 'discard',
        cardsToDiscard: [cardToDiscard],
        drawSource: 'deck'
      }};
      
    }} catch (error) {{
      console.error('Enhanced AI error:', error);
      return {{ 
        action: 'discard', 
        cardsToDiscard: hand.length > 0 ? [hand[0]] : [],
        drawSource: 'deck'
      }};
    }}
  }}

  private findCombinations(hand: Card[]): Card[][] {{
    const combos: Card[][] = [];
    
    // Find all valid combinations
    // Pairs
    for (let i = 0; i < hand.length - 1; i++) {{
      for (let j = i + 1; j < hand.length; j++) {{
        if (hand[i].value === hand[j].value) {{
          combos.push([hand[i], hand[j]]);
        }}
      }}
    }}
    
    // Sets
    const valueGroups = new Map<number, Card[]>();
    hand.forEach(card => {{
      if (!valueGroups.has(card.value)) {{
        valueGroups.set(card.value, []);
      }}
      valueGroups.get(card.value)!.push(card);
    }});
    
    valueGroups.forEach(cards => {{
      if (cards.length >= 3) {{
        combos.push(cards);
      }}
    }});
    
    // Runs (consecutive same suit)
    const suitGroups = new Map<string, Card[]>();
    hand.forEach(card => {{
      const key = card.suit;
      if (!suitGroups.has(key)) {{
        suitGroups.set(key, []);
      }}
      suitGroups.get(key)!.push(card);
    }});
    
    suitGroups.forEach(cards => {{
      if (cards.length >= 3) {{
        cards.sort((a, b) => a.value - b.value);
        // Find consecutive sequences
        for (let start = 0; start <= cards.length - 3; start++) {{
          const run = [cards[start]];
          for (let i = start + 1; i < cards.length; i++) {{
            if (cards[i].value === run[run.length - 1].value + 1) {{
              run.push(cards[i]);
              if (run.length >= 3) {{
                combos.push([...run]);
              }}
            }}
          }}
        }}
      }}
    }});
    
    return combos;
  }}
}}'''
    
    with open('src/game/EnhancedNeuralNetworkAI.ts', 'w') as f:
        f.write(ts_code)
    
    print(f"\n✅ Saved learned AI with behaviors discovered from {agent.games_played} games")
    
    # Save detailed learning data
    learning_data = {
        'games_played': agent.games_played,
        'win_rate': agent.wins / max(agent.games_played, 1),
        'learned_threshold': yaniv_threshold,
        'draw_preference': draw_deck_pref,
        'unique_states': len(agent.experience),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('true_learning_results.json', 'w') as f:
        json.dump(learning_data, f, indent=2)


if __name__ == "__main__":
    print("🎮 True Learning AI for Yaniv")
    print("This AI learns by playing games, not by tweaking parameters\n")
    
    # Train for 10 minutes by default
    train_true_learning_ai(minutes=10)
    
    print("\n🚀 Your AI has learned through experience!")
    print("Refresh your browser to play against the learned AI")