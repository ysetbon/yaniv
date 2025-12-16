"""
State and Action Encoding for Action-Scoring Neural Network

The action-scoring approach:
1. Build features for (state, action) pairs
2. NN outputs single scalar score
3. Evaluate all legal actions, pick the best

This handles variable action counts cleanly for GA training.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .env import (
    Card, Action, ActionType, Suit, Rank,
    get_hand_value, is_set, is_run
)


class StateEncoder:
    """
    Encodes game state into feature vector.

    Features (similar to TS NeuralNetworkAI encoding):
    - Hand: 52-dim multi-hot (1 if card present)
    - Top discard: 52-dim one-hot
    - Top 3 discards: 156-dim (52 * 3)
    - Scalars: hand_value, hand_count, discard_pile_size, round_number, can_call_yaniv
    - Opponents (up to 3): hand_count, score per opponent
    """

    NUM_CARDS = 52  # 4 suits * 13 ranks
    MAX_OPPONENTS = 3
    NUM_TOP_DISCARDS = 3

    def __init__(self):
        # Calculate dimensions
        self.hand_dim = self.NUM_CARDS  # 52
        self.top_discard_dim = self.NUM_CARDS * self.NUM_TOP_DISCARDS  # 156
        self.scalar_dim = 5  # hand_value, hand_count, discard_size, round, can_yaniv
        self.opponent_dim = self.MAX_OPPONENTS * 2  # (hand_count, score) per opponent

        self.state_dim = (
            self.hand_dim +
            self.top_discard_dim +
            self.scalar_dim +
            self.opponent_dim
        )  # 52 + 156 + 5 + 6 = 219

    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        """Encode observation into state vector."""
        features = []

        # Hand encoding (52-dim multi-hot)
        hand_encoding = np.zeros(self.NUM_CARDS, dtype=np.float32)
        for card in observation["hand"]:
            idx = self._card_to_index(card)
            hand_encoding[idx] = 1.0
        features.append(hand_encoding)

        # Top discard cards (52 * 3 = 156 dim)
        discard_encoding = np.zeros(
            self.NUM_CARDS * self.NUM_TOP_DISCARDS, dtype=np.float32
        )
        discard_pile = observation.get("discard_pile", [])
        for i, card in enumerate(reversed(discard_pile[-self.NUM_TOP_DISCARDS:])):
            if i >= self.NUM_TOP_DISCARDS:
                break
            idx = i * self.NUM_CARDS + self._card_to_index(card)
            discard_encoding[idx] = 1.0
        features.append(discard_encoding)

        # Scalar features (normalized)
        scalars = np.zeros(self.scalar_dim, dtype=np.float32)
        scalars[0] = observation.get("hand_value", 0) / 100.0  # hand_value
        scalars[1] = len(observation.get("hand", [])) / 10.0  # hand_count
        scalars[2] = len(observation.get("discard_pile", [])) / 52.0  # discard_size
        scalars[3] = observation.get("round_number", 1) / 20.0  # round_number
        scalars[4] = float(observation.get("can_call_yaniv", False))  # can_yaniv
        features.append(scalars)

        # Opponent features
        opponent_encoding = np.zeros(self.opponent_dim, dtype=np.float32)
        opponents = observation.get("opponents", [])
        for i, opp in enumerate(opponents[:self.MAX_OPPONENTS]):
            base_idx = i * 2
            opponent_encoding[base_idx] = opp.get("hand_count", 5) / 10.0
            opponent_encoding[base_idx + 1] = opp.get("score", 0) / 200.0
        features.append(opponent_encoding)

        return np.concatenate(features)

    def _card_to_index(self, card: Card) -> int:
        """Convert card to 0-51 index."""
        suit_idx = list(Suit).index(card.suit)
        rank_idx = list(Rank).index(card.rank)
        return suit_idx * 13 + rank_idx

    @property
    def dim(self) -> int:
        """Return state vector dimension."""
        return self.state_dim


class ActionEncoder:
    """
    Encodes actions into feature vectors.

    For each action:
    - action_type: 4-dim one-hot (discard, draw_deck, draw_discard, yaniv)
    - draw_source: 2-dim one-hot (deck, discard)
    - discard_mask: 52-dim multi-hot for cards being discarded
    - Scalars: discard_count, discard_value, resulting_hand_value
    """

    NUM_CARDS = 52
    NUM_ACTION_TYPES = 4
    NUM_DRAW_SOURCES = 2

    def __init__(self):
        self.action_type_dim = self.NUM_ACTION_TYPES  # 4
        self.draw_source_dim = self.NUM_DRAW_SOURCES  # 2
        self.discard_mask_dim = self.NUM_CARDS  # 52
        self.scalar_dim = 4  # discard_count, discard_value, result_value, is_yaniv

        self.action_dim = (
            self.action_type_dim +
            self.draw_source_dim +
            self.discard_mask_dim +
            self.scalar_dim
        )  # 4 + 2 + 52 + 4 = 62

    def encode(
        self,
        action: Action,
        observation: Dict[str, Any]
    ) -> np.ndarray:
        """Encode action into feature vector."""
        features = []

        # Action type one-hot
        action_type_enc = np.zeros(self.NUM_ACTION_TYPES, dtype=np.float32)
        action_type_idx = {
            ActionType.DISCARD: 0,
            ActionType.DRAW_DECK: 1,
            ActionType.DRAW_DISCARD: 2,
            ActionType.YANIV: 3
        }.get(action.action_type, 0)
        action_type_enc[action_type_idx] = 1.0
        features.append(action_type_enc)

        # Draw source one-hot
        draw_source_enc = np.zeros(self.NUM_DRAW_SOURCES, dtype=np.float32)
        if action.draw_source == "deck":
            draw_source_enc[0] = 1.0
        else:
            draw_source_enc[1] = 1.0
        features.append(draw_source_enc)

        # Discard mask
        discard_mask = np.zeros(self.NUM_CARDS, dtype=np.float32)
        for card in action.discard_cards:
            idx = self._card_to_index(card)
            discard_mask[idx] = 1.0
        features.append(discard_mask)

        # Scalar features
        hand = observation.get("hand", [])
        discard_count = len(action.discard_cards)
        discard_value = sum(c.value for c in action.discard_cards)

        # Calculate resulting hand value after action
        if action.action_type == ActionType.YANIV:
            result_value = get_hand_value(hand)
        else:
            remaining = [c for c in hand if c not in action.discard_cards]
            # Estimate draw value
            if action.draw_source == "discard":
                drawable = observation.get("drawable_discard")
                draw_value = drawable.value if drawable else 7.0
            else:
                draw_value = 7.0  # Average card value
            result_value = get_hand_value(remaining) + draw_value

        scalars = np.array([
            discard_count / 5.0,
            discard_value / 50.0,
            result_value / 100.0,
            float(action.action_type == ActionType.YANIV)
        ], dtype=np.float32)
        features.append(scalars)

        return np.concatenate(features)

    def encode_batch(
        self,
        actions: List[Action],
        observation: Dict[str, Any]
    ) -> np.ndarray:
        """Encode multiple actions into feature matrix [num_actions, action_dim]."""
        return np.stack([self.encode(a, observation) for a in actions])

    def _card_to_index(self, card: Card) -> int:
        """Convert card to 0-51 index."""
        suit_idx = list(Suit).index(card.suit)
        rank_idx = list(Rank).index(card.rank)
        return suit_idx * 13 + rank_idx

    @property
    def dim(self) -> int:
        """Return action vector dimension."""
        return self.action_dim


class StateActionEncoder:
    """
    Combined encoder for (state, action) pairs.
    Used by the action-scoring network.
    """

    def __init__(self):
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
        self.input_dim = self.state_encoder.dim + self.action_encoder.dim

    def encode(
        self,
        observation: Dict[str, Any],
        action: Action
    ) -> np.ndarray:
        """Encode (state, action) pair into single vector."""
        state_vec = self.state_encoder.encode(observation)
        action_vec = self.action_encoder.encode(action, observation)
        return np.concatenate([state_vec, action_vec])

    def encode_state_action_batch(
        self,
        observation: Dict[str, Any],
        actions: List[Action]
    ) -> np.ndarray:
        """
        Encode state with multiple actions for batch evaluation.
        Returns [num_actions, input_dim] matrix.
        """
        state_vec = self.state_encoder.encode(observation)
        # Repeat state for each action
        state_batch = np.repeat(
            state_vec[np.newaxis, :],
            len(actions),
            axis=0
        )
        action_batch = self.action_encoder.encode_batch(actions, observation)
        return np.concatenate([state_batch, action_batch], axis=1)

    @property
    def dim(self) -> int:
        """Return combined input dimension."""
        return self.input_dim


# Utility functions for card indexing
def card_index_to_card(idx: int) -> Card:
    """Convert 0-51 index back to Card."""
    suit_idx = idx // 13
    rank_idx = idx % 13
    return Card(suit=list(Suit)[suit_idx], rank=list(Rank)[rank_idx])


def create_card_one_hot(card: Card) -> np.ndarray:
    """Create 52-dim one-hot vector for a single card."""
    vec = np.zeros(52, dtype=np.float32)
    suit_idx = list(Suit).index(card.suit)
    rank_idx = list(Rank).index(card.rank)
    vec[suit_idx * 13 + rank_idx] = 1.0
    return vec


def create_hand_multi_hot(cards: List[Card]) -> np.ndarray:
    """Create 52-dim multi-hot vector for a hand."""
    vec = np.zeros(52, dtype=np.float32)
    for card in cards:
        suit_idx = list(Suit).index(card.suit)
        rank_idx = list(Rank).index(card.rank)
        vec[suit_idx * 13 + rank_idx] = 1.0
    return vec
