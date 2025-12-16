"""
Baseline bots for Yaniv - required for stable GA evaluation.

Provides:
- RandomBot: Uniform random action selection
- GreedyBot: Minimizes hand value after each action
- RuleBasedBot: Heuristic-based decision making
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import random

from .env import (
    Action, ActionType, Card, get_hand_value, is_set, is_run,
    get_all_discard_combinations
)


class BaseBot(ABC):
    """Abstract base class for all bots."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    @abstractmethod
    def choose_action(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Action]
    ) -> Action:
        """Choose an action given observation and legal actions."""
        pass

    def set_seed(self, seed: int) -> None:
        """Set random seed."""
        self.rng = random.Random(seed)


class RandomBot(BaseBot):
    """
    Random action selection.
    Useful as baseline and for diversity in training.
    """

    def choose_action(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Action]
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")
        return self.rng.choice(legal_actions)


class GreedyBot(BaseBot):
    """
    Greedy bot that minimizes hand value after each move.

    Strategy:
    - Always call Yaniv if possible (hand value <= 7)
    - Otherwise, discard the combination that maximizes value removed
    - Prefer drawing from deck (blind draw avoids giving info to opponent)
    """

    def __init__(self, yaniv_threshold: int = 5, seed: Optional[int] = None):
        super().__init__(seed)
        self.yaniv_threshold = yaniv_threshold

    def choose_action(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Action]
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")

        hand = observation["hand"]
        hand_value = get_hand_value(hand)

        # Check for Yaniv action
        yaniv_actions = [a for a in legal_actions if a.action_type == ActionType.YANIV]
        if yaniv_actions and hand_value <= self.yaniv_threshold:
            return yaniv_actions[0]

        # Filter to discard actions only
        discard_actions = [
            a for a in legal_actions if a.action_type == ActionType.DISCARD
        ]

        if not discard_actions:
            # Fallback to any legal action
            return self.rng.choice(legal_actions)

        # Score each action by resulting hand value (lower is better)
        def score_action(action: Action) -> float:
            # Calculate hand value after discard
            remaining_hand = [c for c in hand if c not in action.discard_cards]
            discard_value = sum(c.value for c in action.discard_cards)

            # Estimate draw value (average card value ~7)
            expected_draw_value = 7.0
            if action.draw_source == "discard":
                drawable = observation.get("drawable_discard")
                if drawable:
                    expected_draw_value = drawable.value

            resulting_value = get_hand_value(remaining_hand) + expected_draw_value
            return resulting_value

        # Choose action with lowest resulting hand value
        best_action = min(discard_actions, key=score_action)
        return best_action


class RuleBasedBot(BaseBot):
    """
    Rule-based bot with heuristics for Yaniv.

    Strategies:
    - Call Yaniv when hand value <= 5 (conservative)
    - Call Yaniv at 6-7 only if opponent likely has higher
    - Prefer discarding high-value singles
    - Keep cards that might form sets/runs
    - Draw from discard only if card is useful
    """

    def __init__(
        self,
        aggressive_yaniv: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.aggressive_yaniv = aggressive_yaniv

    def choose_action(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Action]
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")

        hand = observation["hand"]
        hand_value = get_hand_value(hand)
        opponents = observation.get("opponents", [])

        # Yaniv decision
        yaniv_actions = [a for a in legal_actions if a.action_type == ActionType.YANIV]
        if yaniv_actions:
            if self._should_call_yaniv(hand_value, opponents):
                return yaniv_actions[0]

        # Filter to discard actions
        discard_actions = [
            a for a in legal_actions if a.action_type == ActionType.DISCARD
        ]

        if not discard_actions:
            return self.rng.choice(legal_actions)

        # Rank actions by heuristic score
        scored_actions = [
            (a, self._score_discard_action(a, observation))
            for a in discard_actions
        ]
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        # Add some randomness to avoid being predictable
        top_k = min(3, len(scored_actions))
        top_actions = scored_actions[:top_k]
        weights = [s + 1.0 for _, s in top_actions]  # +1 to avoid zero weights

        chosen = self.rng.choices(
            [a for a, _ in top_actions],
            weights=weights,
            k=1
        )[0]

        return chosen

    def _should_call_yaniv(
        self,
        hand_value: int,
        opponents: List[Dict[str, Any]]
    ) -> bool:
        """Decide whether to call Yaniv."""
        if hand_value <= 4:
            return True  # Very safe

        if hand_value == 5:
            # Call unless opponent has very few cards
            for opp in opponents:
                if opp["hand_count"] <= 2:
                    return False  # Opponent might have very low value
            return True

        if hand_value <= 7 and self.aggressive_yaniv:
            # Risky call - check opponent hand counts
            for opp in opponents:
                if opp["hand_count"] <= 3:
                    return False  # Too risky
            return True

        return False

    def _score_discard_action(
        self,
        action: Action,
        observation: Dict[str, Any]
    ) -> float:
        """Score a discard action (higher is better)."""
        hand = observation["hand"]
        score = 0.0

        # Base score: value of cards discarded
        discard_value = sum(c.value for c in action.discard_cards)
        score += discard_value * 2.0

        # Bonus for discarding sets/runs (multi-card discards)
        if len(action.discard_cards) >= 2:
            score += len(action.discard_cards) * 3.0

        # Bonus for discarding high cards (K, Q, J)
        high_cards = sum(1 for c in action.discard_cards if c.value >= 10)
        score += high_cards * 5.0

        # Calculate remaining hand
        remaining = [c for c in hand if c not in action.discard_cards]

        # Penalty if breaking potential sets/runs
        remaining_potential = self._count_potential_combos(remaining)
        hand_potential = self._count_potential_combos(hand)
        if remaining_potential < hand_potential:
            score -= (hand_potential - remaining_potential) * 2.0

        # Draw source preference
        drawable = observation.get("drawable_discard")
        if action.draw_source == "discard" and drawable:
            # Only prefer discard if card is useful
            if self._is_card_useful(drawable, remaining):
                score += 5.0
            else:
                score -= 2.0  # Drawing useless card gives info to opponent
        elif action.draw_source == "deck":
            score += 1.0  # Slight preference for blind draw

        return score

    def _count_potential_combos(self, cards: List[Card]) -> int:
        """Count potential sets/runs in a hand."""
        potential = 0

        # Count pairs (potential sets)
        from collections import Counter
        rank_counts = Counter(c.rank for c in cards)
        for count in rank_counts.values():
            if count >= 2:
                potential += count - 1

        # Count consecutive same-suit (potential runs)
        from collections import defaultdict
        by_suit = defaultdict(list)
        for c in cards:
            by_suit[c.suit].append(c.value)

        for values in by_suit.values():
            if len(values) >= 2:
                sorted_vals = sorted(values)
                consecutive = 1
                for i in range(1, len(sorted_vals)):
                    if sorted_vals[i] == sorted_vals[i-1] + 1:
                        consecutive += 1
                    else:
                        consecutive = 1
                    if consecutive >= 2:
                        potential += 1

        return potential

    def _is_card_useful(self, card: Card, hand: List[Card]) -> bool:
        """Check if a card would be useful in hand."""
        # Check if it forms a pair/set
        for c in hand:
            if c.rank == card.rank:
                return True

        # Check if it extends a potential run
        for c in hand:
            if c.suit == card.suit:
                if abs(c.value - card.value) <= 2:
                    return True

        # Low value cards are generally useful
        if card.value <= 3:
            return True

        return False


class HallOfFameBot(BaseBot):
    """
    Bot that uses a neural network policy from training.
    This is a wrapper for trained policies to use in evaluation.
    """

    def __init__(self, policy: Any, seed: Optional[int] = None):
        super().__init__(seed)
        self.policy = policy

    def choose_action(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Action]
    ) -> Action:
        return self.policy.choose_action(observation, legal_actions)
