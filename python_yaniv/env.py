"""
Yaniv Game Environment - Matches TypeScript rules exactly

Rules (matching src/game/Game.ts):
- 52-card deck (no jokers)
- Card values: A=1, 2-10=face value, J=11, Q=12, K=13
- Deal 5 cards per player
- Valid discards: single, set (2+ same rank), run (3+ consecutive same suit)
- Yaniv threshold: hand value <= 7
- Assaf: +30 penalty if opponent has <= caller's value
- 50-point reduction rule at multiples of 50
- Target score: 101 (first to reach loses)
- Draw from deck or from opponent's last discard (first card only)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import random
import copy


class Suit(Enum):
    SPADES = "♠"
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"


class Rank(Enum):
    ACE = ("A", 1)
    TWO = ("2", 2)
    THREE = ("3", 3)
    FOUR = ("4", 4)
    FIVE = ("5", 5)
    SIX = ("6", 6)
    SEVEN = ("7", 7)
    EIGHT = ("8", 8)
    NINE = ("9", 9)
    TEN = ("10", 10)
    JACK = ("J", 11)
    QUEEN = ("Q", 12)
    KING = ("K", 13)

    def __init__(self, symbol: str, value: int):
        self.symbol = symbol
        self._value = value

    @property
    def value(self) -> int:
        return self._value


@dataclass(frozen=True)
class Card:
    """Immutable card representation matching TS Card type."""
    suit: Suit
    rank: Rank

    @property
    def value(self) -> int:
        """Card value matching CardUtils.getRankValue in TS."""
        return self.rank.value

    @property
    def id(self) -> str:
        return f"{self.suit.value}-{self.rank.symbol}"

    def __repr__(self) -> str:
        return f"{self.rank.symbol}{self.suit.value}"

    def to_index(self) -> int:
        """Convert card to 0-51 index for encoding."""
        suit_idx = list(Suit).index(self.suit)
        rank_idx = list(Rank).index(self.rank)
        return suit_idx * 13 + rank_idx


class ActionType(Enum):
    DISCARD = "discard"
    DRAW_DECK = "draw_deck"
    DRAW_DISCARD = "draw_discard"
    YANIV = "yaniv"


@dataclass
class Action:
    """
    Action representation for the Yaniv game.

    In TS, a turn has two phases: discard then draw.
    Here we combine them for training convenience.
    """
    action_type: ActionType
    discard_cards: Tuple[Card, ...] = field(default_factory=tuple)
    draw_source: str = "deck"  # "deck" or "discard"

    def __repr__(self) -> str:
        if self.action_type == ActionType.YANIV:
            return "YANIV"
        cards_str = ",".join(str(c) for c in self.discard_cards)
        return f"Discard({cards_str})->Draw({self.draw_source})"


@dataclass
class Player:
    """Player state."""
    id: str
    name: str
    hand: List[Card] = field(default_factory=list)
    score: int = 0
    has_called_yaniv_last_round: bool = False


class Deck:
    """52-card deck matching TS Deck class."""

    def __init__(self, seed: Optional[int] = None):
        self.cards: List[Card] = []
        self.rng = random.Random(seed)
        self._initialize()
        self.shuffle()

    def _initialize(self) -> None:
        """Create standard 52-card deck (no jokers)."""
        self.cards = []
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(suit, rank))

    def shuffle(self) -> None:
        """Fisher-Yates shuffle matching TS implementation."""
        for i in range(len(self.cards) - 1, 0, -1):
            j = self.rng.randint(0, i)
            self.cards[i], self.cards[j] = self.cards[j], self.cards[i]

    def draw(self, count: int = 1) -> List[Card]:
        """Draw cards from top of deck."""
        drawn = []
        for _ in range(count):
            if self.cards:
                drawn.append(self.cards.pop())
        return drawn

    def add_cards(self, cards: List[Card]) -> None:
        """Add cards to deck."""
        self.cards.extend(cards)

    def size(self) -> int:
        return len(self.cards)

    def is_empty(self) -> bool:
        return len(self.cards) == 0


def get_hand_value(cards: List[Card]) -> int:
    """Calculate hand value matching CardUtils.getHandValue in TS."""
    return sum(card.value for card in cards)


def is_set(cards: List[Card]) -> bool:
    """
    Check if cards form a valid set (2+ cards of same rank).
    Matches CardUtils.isSet in TS.
    """
    if len(cards) < 2:
        return False
    rank = cards[0].rank
    return all(card.rank == rank for card in cards)


def is_run(cards: List[Card]) -> bool:
    """
    Check if cards form a valid run (3+ consecutive same suit).
    Matches CardUtils.isRun in TS.
    """
    if len(cards) < 3:
        return False

    suit = cards[0].suit
    if not all(card.suit == suit for card in cards):
        return False

    sorted_cards = sorted(cards, key=lambda c: c.value)
    for i in range(1, len(sorted_cards)):
        if sorted_cards[i].value != sorted_cards[i - 1].value + 1:
            return False

    return True


def is_valid_discard(cards: List[Card]) -> bool:
    """
    Validate discard combination.
    Matches CardUtils.isValidDiscard in TS.
    """
    if len(cards) == 0:
        return False
    if len(cards) == 1:
        return True
    return is_set(cards) or is_run(cards)


def get_all_discard_combinations(hand: List[Card]) -> List[Tuple[Card, ...]]:
    """
    Generate all valid discard combinations from a hand.
    Returns list of card tuples that can be discarded.
    """
    combinations = []

    # Single cards
    for card in hand:
        combinations.append((card,))

    # Sets (pairs, three-of-a-kind, four-of-a-kind)
    by_rank: Dict[Rank, List[Card]] = {}
    for card in hand:
        by_rank.setdefault(card.rank, []).append(card)

    for rank, cards in by_rank.items():
        if len(cards) >= 2:
            # Generate all subsets of size >= 2
            from itertools import combinations as itertools_combinations
            for size in range(2, len(cards) + 1):
                for combo in itertools_combinations(cards, size):
                    combinations.append(tuple(combo))

    # Runs (3+ consecutive same suit)
    by_suit: Dict[Suit, List[Card]] = {}
    for card in hand:
        by_suit.setdefault(card.suit, []).append(card)

    for suit, cards in by_suit.items():
        if len(cards) >= 3:
            # Sort by value
            sorted_cards = sorted(cards, key=lambda c: c.value)
            # Find all consecutive sequences of length >= 3
            for start_idx in range(len(sorted_cards)):
                run = [sorted_cards[start_idx]]
                for next_idx in range(start_idx + 1, len(sorted_cards)):
                    if sorted_cards[next_idx].value == run[-1].value + 1:
                        run.append(sorted_cards[next_idx])
                        if len(run) >= 3:
                            combinations.append(tuple(run))
                    else:
                        break

    return combinations


class YanivEnv:
    """
    Yaniv game environment matching TypeScript YanivGame rules.

    Supports 2+ players, but primarily designed for 2-player training.
    """

    YANIV_THRESHOLD = 7
    ASSAF_PENALTY = 30
    TARGET_SCORE = 101
    CARDS_PER_PLAYER = 5

    def __init__(
        self,
        num_players: int = 2,
        seed: Optional[int] = None,
        max_turns: int = 500
    ):
        self.num_players = num_players
        self.seed = seed
        self.max_turns = max_turns
        self.rng = random.Random(seed)

        self.players: List[Player] = []
        self.deck: Optional[Deck] = None
        self.discard_pile: List[Card] = []
        self.current_player_idx: int = 0
        self.round_number: int = 1
        self.turn_count: int = 0
        self.game_over: bool = False
        self.winner_id: Optional[str] = None

        # Track last discard by each player (for draw from discard)
        self.last_discards: Dict[str, List[Card]] = {}

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment to start a new game."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        # Initialize players
        self.players = [
            Player(id=f"player-{i}", name=f"Player {i}")
            for i in range(self.num_players)
        ]

        self.round_number = 1
        self.turn_count = 0
        self.game_over = False
        self.winner_id = None
        self.last_discards = {}

        self._start_round()

        return self._get_observation(self.current_player_idx)

    def _start_round(self) -> None:
        """Start a new round (deal cards, setup discard pile)."""
        self.deck = Deck(seed=self.rng.randint(0, 2**31))

        # Deal 5 cards to each player
        for player in self.players:
            player.hand = self.deck.draw(self.CARDS_PER_PLAYER)

        # Initial discard
        self.discard_pile = self.deck.draw(1)

        # Reset turn tracking
        self.current_player_idx = 0
        self.last_discards = {}

        # Treat initial card as if second player discarded it (like TS)
        if len(self.players) > 1 and self.discard_pile:
            self.last_discards[self.players[1].id] = list(self.discard_pile)

    def _get_observation(self, player_idx: int) -> Dict[str, Any]:
        """Get observation for a player."""
        player = self.players[player_idx]

        # Get opponent info
        opponents = []
        for i, p in enumerate(self.players):
            if i != player_idx:
                opponents.append({
                    "id": p.id,
                    "hand_count": len(p.hand),
                    "score": p.score
                })

        # Get drawable discard card
        drawable_discard = None
        opponent_id = self.players[(player_idx + 1) % len(self.players)].id
        if opponent_id in self.last_discards and self.last_discards[opponent_id]:
            drawable_discard = self.last_discards[opponent_id][0]

        return {
            "player_id": player.id,
            "hand": list(player.hand),
            "hand_value": get_hand_value(player.hand),
            "score": player.score,
            "discard_pile": list(self.discard_pile),
            "discard_pile_top": self.discard_pile[-1] if self.discard_pile else None,
            "drawable_discard": drawable_discard,
            "deck_size": self.deck.size() if self.deck else 0,
            "opponents": opponents,
            "round_number": self.round_number,
            "can_call_yaniv": self._can_call_yaniv(player_idx),
            "has_called_yaniv_last_round": player.has_called_yaniv_last_round,
            "is_current_player": self.current_player_idx == player_idx
        }

    def _can_call_yaniv(self, player_idx: int) -> bool:
        """Check if player can call Yaniv."""
        player = self.players[player_idx]
        hand_value = get_hand_value(player.hand)
        return hand_value <= self.YANIV_THRESHOLD and not player.has_called_yaniv_last_round

    def get_legal_actions(self, player_idx: Optional[int] = None) -> List[Action]:
        """
        Get all legal actions for a player.

        Actions combine discard + draw into single action for training.
        """
        if player_idx is None:
            player_idx = self.current_player_idx

        if self.game_over:
            return []

        player = self.players[player_idx]
        actions = []

        # Yaniv action (if eligible)
        if self._can_call_yaniv(player_idx):
            actions.append(Action(action_type=ActionType.YANIV))

        # Get all valid discard combinations
        discard_combos = get_all_discard_combinations(player.hand)

        # Check if can draw from discard
        opponent_id = self.players[(player_idx + 1) % len(self.players)].id
        can_draw_from_discard = (
            opponent_id in self.last_discards and
            len(self.last_discards[opponent_id]) > 0
        )

        # For each discard combo, create actions with both draw sources
        for combo in discard_combos:
            # Draw from deck
            actions.append(Action(
                action_type=ActionType.DISCARD,
                discard_cards=combo,
                draw_source="deck"
            ))

            # Draw from discard (if available)
            if can_draw_from_discard:
                # Cannot draw what you just discarded (optional rule)
                discard_card = self.last_discards[opponent_id][0]
                actions.append(Action(
                    action_type=ActionType.DISCARD,
                    discard_cards=combo,
                    draw_source="discard"
                ))

        return actions

    def step(
        self,
        action: Action,
        player_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action and return (observation, reward, done, info).
        """
        if player_idx is None:
            player_idx = self.current_player_idx

        if self.game_over:
            return self._get_observation(player_idx), 0.0, True, {"error": "game_over"}

        if player_idx != self.current_player_idx:
            return self._get_observation(player_idx), 0.0, False, {"error": "not_your_turn"}

        player = self.players[player_idx]
        reward = 0.0
        info: Dict[str, Any] = {}

        if action.action_type == ActionType.YANIV:
            # Handle Yaniv call
            result = self._call_yaniv(player_idx)
            info["yaniv_result"] = result
            reward = self._calculate_yaniv_reward(player_idx, result)

        else:
            # Handle discard + draw
            # Validate discard
            if not self._validate_discard(player_idx, action.discard_cards):
                return self._get_observation(player_idx), -1.0, False, {"error": "invalid_discard"}

            # Execute discard
            for card in action.discard_cards:
                player.hand.remove(card)
                self.discard_pile.append(card)

            # Track this discard
            self.last_discards[player.id] = list(action.discard_cards)

            # Execute draw
            if action.draw_source == "discard":
                drawn = self._draw_from_discard(player_idx)
            else:
                drawn = self._draw_from_deck(player_idx)

            if drawn:
                player.hand.extend(drawn)
                info["drawn_cards"] = drawn

            # Small reward shaping for reducing hand value
            new_hand_value = get_hand_value(player.hand)
            reward = -0.001 * new_hand_value  # Slight preference for lower hand

            # Move to next turn
            self._next_turn()

        self.turn_count += 1
        done = self.game_over or self.turn_count >= self.max_turns

        if self.turn_count >= self.max_turns and not self.game_over:
            # Timeout - determine winner by lowest hand value
            info["timeout"] = True
            self._handle_timeout()
            done = True

        return self._get_observation(player_idx), reward, done, info

    def _validate_discard(self, player_idx: int, cards: Tuple[Card, ...]) -> bool:
        """Validate that player has cards and they form valid discard."""
        player = self.players[player_idx]

        # Check player has all cards
        hand_copy = list(player.hand)
        for card in cards:
            if card not in hand_copy:
                return False
            hand_copy.remove(card)

        return is_valid_discard(list(cards))

    def _draw_from_deck(self, player_idx: int) -> List[Card]:
        """Draw from deck, reshuffling discard if needed."""
        if self.deck.is_empty() and len(self.discard_pile) > 1:
            # Reshuffle discard pile (keep top card)
            top_card = self.discard_pile.pop()
            self.deck.add_cards(self.discard_pile)
            self.deck.shuffle()
            self.discard_pile = [top_card]

        return self.deck.draw(1)

    def _draw_from_discard(self, player_idx: int) -> List[Card]:
        """Draw from opponent's last discard (first card only)."""
        opponent_id = self.players[(player_idx + 1) % len(self.players)].id

        if opponent_id not in self.last_discards or not self.last_discards[opponent_id]:
            return []

        # Take first card from opponent's last discard
        card_to_take = self.last_discards[opponent_id][0]

        # Find and remove from discard pile
        if card_to_take in self.discard_pile:
            self.discard_pile.remove(card_to_take)

        # Clear opponent's last discard
        del self.last_discards[opponent_id]

        return [card_to_take]

    def _call_yaniv(self, player_idx: int) -> Dict[str, Any]:
        """Handle Yaniv call and scoring."""
        caller = self.players[player_idx]
        caller_value = get_hand_value(caller.hand)

        lowest_value = caller_value
        assaf = False

        # Check for Assaf
        for i, player in enumerate(self.players):
            if i != player_idx:
                player_value = get_hand_value(player.hand)
                if player_value <= caller_value:
                    assaf = True
                    lowest_value = min(lowest_value, player_value)

        scores: Dict[str, int] = {}

        # Calculate scores
        for player in self.players:
            hand_value = get_hand_value(player.hand)

            if player.id == caller.id and assaf:
                # Assaf penalty
                scores[player.id] = self.ASSAF_PENALTY
                player.has_called_yaniv_last_round = True
            elif hand_value == lowest_value:
                # Winner gets 0
                scores[player.id] = 0
                player.has_called_yaniv_last_round = False
            else:
                # Others get their hand value
                scores[player.id] = hand_value
                player.has_called_yaniv_last_round = False

            player.score += scores[player.id]

            # 50-point reduction rule
            if player.score % 50 == 0 and player.score > 0:
                player.score -= 50

        # Check for game end
        self._check_game_end()

        if not self.game_over:
            self._start_new_round()

        return {
            "assaf": assaf,
            "scores": scores,
            "caller_value": caller_value,
            "game_over": self.game_over
        }

    def _calculate_yaniv_reward(self, player_idx: int, result: Dict[str, Any]) -> float:
        """Calculate reward for Yaniv call."""
        if result["assaf"]:
            return -1.0  # Penalty for failed Yaniv

        # Check if this player won the round
        scores = result["scores"]
        player_id = self.players[player_idx].id
        if scores[player_id] == 0:
            return 1.0  # Reward for successful Yaniv
        return 0.0

    def _start_new_round(self) -> None:
        """Start a new round after Yaniv."""
        self.round_number += 1
        self._start_round()

    def _next_turn(self) -> None:
        """Move to next player's turn."""
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def _check_game_end(self) -> None:
        """Check if any player has reached target score."""
        for player in self.players:
            if player.score >= self.TARGET_SCORE:
                self.game_over = True
                # Winner is the player with lowest score
                winner = min(self.players, key=lambda p: p.score)
                self.winner_id = winner.id
                return

    def _handle_timeout(self) -> None:
        """Handle game timeout - winner is lowest hand value."""
        self.game_over = True
        winner = min(self.players, key=lambda p: get_hand_value(p.hand))
        self.winner_id = winner.id

    def get_winner(self) -> Optional[str]:
        """Get winner ID if game is over."""
        return self.winner_id

    def get_player_score(self, player_idx: int) -> int:
        """Get player's current score."""
        return self.players[player_idx].score

    def render(self) -> str:
        """Render game state as string."""
        lines = [f"=== Round {self.round_number}, Turn {self.turn_count} ==="]

        for i, player in enumerate(self.players):
            marker = " *" if i == self.current_player_idx else ""
            hand_str = ", ".join(str(c) for c in player.hand)
            hand_value = get_hand_value(player.hand)
            lines.append(
                f"{player.name}{marker}: [{hand_str}] "
                f"(value={hand_value}, score={player.score})"
            )

        if self.discard_pile:
            lines.append(f"Discard top: {self.discard_pile[-1]}")
        lines.append(f"Deck: {self.deck.size()} cards")

        return "\n".join(lines)


def play_game(
    policies: List["BaseBot"],
    env: Optional[YanivEnv] = None,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Play a complete game with given policies.

    Args:
        policies: List of policy objects (one per player)
        env: Optional environment to use
        seed: Random seed
        verbose: Print game progress

    Returns:
        Dict with game results
    """
    if env is None:
        env = YanivEnv(num_players=len(policies), seed=seed)

    obs = env.reset(seed=seed)
    done = False

    while not done:
        player_idx = env.current_player_idx
        policy = policies[player_idx]

        legal_actions = env.get_legal_actions(player_idx)
        if not legal_actions:
            # No legal actions - should not happen
            break

        action = policy.choose_action(obs, legal_actions)
        obs, reward, done, info = env.step(action)

        if verbose:
            print(env.render())
            print(f"Action: {action}")
            print()

    return {
        "winner_id": env.get_winner(),
        "winner_idx": env.players.index(
            next(p for p in env.players if p.id == env.winner_id)
        ) if env.winner_id else None,
        "scores": {p.id: p.score for p in env.players},
        "rounds": env.round_number,
        "turns": env.turn_count
    }
