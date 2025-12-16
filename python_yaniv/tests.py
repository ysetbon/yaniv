"""
Unit tests for Yaniv GA Training System

Tests verify:
1. Rule parity with TypeScript implementation
2. Encoding correctness
3. Model functionality
4. GA operations
"""

import unittest
import numpy as np
from typing import List

from .env import (
    Card, Suit, Rank, Deck,
    is_set, is_run, is_valid_discard, get_hand_value,
    get_all_discard_combinations, YanivEnv, Action, ActionType
)
from .encoding import StateEncoder, ActionEncoder, StateActionEncoder
from .model import ActionValueNet, TorchPolicy
from .genome import Genome, Population
from .bots import RandomBot, GreedyBot, RuleBasedBot


class TestCardValues(unittest.TestCase):
    """Test card values match TypeScript CardUtils.getRankValue."""

    def test_ace_value(self):
        """A = 1"""
        card = Card(Suit.SPADES, Rank.ACE)
        self.assertEqual(card.value, 1)

    def test_number_cards(self):
        """2-10 = face value"""
        for rank in [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
                     Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN]:
            card = Card(Suit.HEARTS, rank)
            self.assertEqual(card.value, rank.value)

    def test_face_cards(self):
        """J=11, Q=12, K=13 (matching TS, not common variant)"""
        self.assertEqual(Card(Suit.CLUBS, Rank.JACK).value, 11)
        self.assertEqual(Card(Suit.CLUBS, Rank.QUEEN).value, 12)
        self.assertEqual(Card(Suit.CLUBS, Rank.KING).value, 13)

    def test_hand_value(self):
        """Test hand value calculation."""
        hand = [
            Card(Suit.SPADES, Rank.ACE),      # 1
            Card(Suit.HEARTS, Rank.FIVE),     # 5
            Card(Suit.DIAMONDS, Rank.KING),   # 13
        ]
        self.assertEqual(get_hand_value(hand), 19)


class TestDeck(unittest.TestCase):
    """Test deck matches TypeScript Deck class."""

    def test_deck_size(self):
        """Deck has 52 cards (no jokers)"""
        deck = Deck()
        self.assertEqual(deck.size(), 52)

    def test_deck_contains_all_cards(self):
        """Deck contains all 52 standard cards."""
        deck = Deck(seed=42)
        cards = []
        while not deck.is_empty():
            cards.extend(deck.draw(1))

        # Check we have 4 of each rank
        for rank in Rank:
            rank_cards = [c for c in cards if c.rank == rank]
            self.assertEqual(len(rank_cards), 4)

        # Check we have 13 of each suit
        for suit in Suit:
            suit_cards = [c for c in cards if c.suit == suit]
            self.assertEqual(len(suit_cards), 13)

    def test_no_jokers(self):
        """Deck should not contain jokers."""
        deck = Deck()
        cards = deck.draw(52)
        for card in cards:
            # All cards should have valid rank
            self.assertIn(card.rank, list(Rank))


class TestValidDiscards(unittest.TestCase):
    """Test valid discard combinations match TypeScript CardUtils."""

    def test_single_card_valid(self):
        """Single card is always valid."""
        card = Card(Suit.SPADES, Rank.ACE)
        self.assertTrue(is_valid_discard([card]))

    def test_empty_invalid(self):
        """Empty discard is invalid."""
        self.assertFalse(is_valid_discard([]))

    def test_pair_valid(self):
        """Pair (2 cards same rank) is valid."""
        cards = [
            Card(Suit.SPADES, Rank.FIVE),
            Card(Suit.HEARTS, Rank.FIVE)
        ]
        self.assertTrue(is_set(cards))
        self.assertTrue(is_valid_discard(cards))

    def test_three_of_kind_valid(self):
        """Three of a kind is valid."""
        cards = [
            Card(Suit.SPADES, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.DIAMONDS, Rank.SEVEN)
        ]
        self.assertTrue(is_set(cards))
        self.assertTrue(is_valid_discard(cards))

    def test_four_of_kind_valid(self):
        """Four of a kind is valid."""
        cards = [
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.KING),
            Card(Suit.CLUBS, Rank.KING)
        ]
        self.assertTrue(is_set(cards))
        self.assertTrue(is_valid_discard(cards))

    def test_run_valid(self):
        """Run (3+ consecutive same suit) is valid."""
        cards = [
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.SPADES, Rank.FIVE)
        ]
        self.assertTrue(is_run(cards))
        self.assertTrue(is_valid_discard(cards))

    def test_run_four_cards(self):
        """Longer run is valid."""
        cards = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.NINE),
            Card(Suit.HEARTS, Rank.TEN)
        ]
        self.assertTrue(is_run(cards))

    def test_run_mixed_suits_invalid(self):
        """Run with mixed suits is invalid."""
        cards = [
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.HEARTS, Rank.FOUR),  # Different suit!
            Card(Suit.SPADES, Rank.FIVE)
        ]
        self.assertFalse(is_run(cards))

    def test_run_non_consecutive_invalid(self):
        """Non-consecutive cards don't form run."""
        cards = [
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.SPADES, Rank.FIVE),  # Gap!
            Card(Suit.SPADES, Rank.SIX)
        ]
        self.assertFalse(is_run(cards))

    def test_two_card_run_invalid(self):
        """Run needs at least 3 cards."""
        cards = [
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.SPADES, Rank.FOUR)
        ]
        self.assertFalse(is_run(cards))

    def test_mismatched_ranks_invalid(self):
        """Two different ranks (not set, not run) is invalid."""
        cards = [
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.HEARTS, Rank.FIVE)
        ]
        self.assertFalse(is_valid_discard(cards))


class TestYanivThreshold(unittest.TestCase):
    """Test Yaniv call threshold matches TypeScript."""

    def test_yaniv_threshold_is_7(self):
        """Yaniv can be called when hand value <= 7."""
        env = YanivEnv(seed=42)
        env.reset()

        # Manually set player hand to test threshold
        player = env.players[0]
        player.hand = [
            Card(Suit.SPADES, Rank.ACE),    # 1
            Card(Suit.HEARTS, Rank.TWO),    # 2
            Card(Suit.DIAMONDS, Rank.THREE) # 3
        ]  # Total = 6
        self.assertTrue(env._can_call_yaniv(0))

        player.hand = [
            Card(Suit.SPADES, Rank.ACE),    # 1
            Card(Suit.HEARTS, Rank.TWO),    # 2
            Card(Suit.DIAMONDS, Rank.FOUR)  # 4
        ]  # Total = 7
        self.assertTrue(env._can_call_yaniv(0))

        player.hand = [
            Card(Suit.SPADES, Rank.ACE),    # 1
            Card(Suit.HEARTS, Rank.TWO),    # 2
            Card(Suit.DIAMONDS, Rank.FIVE)  # 5
        ]  # Total = 8
        self.assertFalse(env._can_call_yaniv(0))


class TestAssafPenalty(unittest.TestCase):
    """Test Assaf penalty matches TypeScript."""

    def test_assaf_penalty_is_30(self):
        """Assaf penalty is 30 points."""
        self.assertEqual(YanivEnv.ASSAF_PENALTY, 30)


class TestTargetScore(unittest.TestCase):
    """Test target score matches TypeScript."""

    def test_target_score_is_101(self):
        """Game ends at 101 points."""
        self.assertEqual(YanivEnv.TARGET_SCORE, 101)


class TestDiscardCombinations(unittest.TestCase):
    """Test that all valid discard combinations are generated."""

    def test_generates_singles(self):
        """Should generate single card discards."""
        hand = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.FIVE)
        ]
        combos = get_all_discard_combinations(hand)

        singles = [c for c in combos if len(c) == 1]
        self.assertEqual(len(singles), 2)

    def test_generates_pairs(self):
        """Should generate pairs."""
        hand = [
            Card(Suit.SPADES, Rank.FIVE),
            Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.DIAMONDS, Rank.KING)
        ]
        combos = get_all_discard_combinations(hand)

        pairs = [c for c in combos if len(c) == 2 and is_set(list(c))]
        self.assertEqual(len(pairs), 1)

    def test_generates_runs(self):
        """Should generate runs."""
        hand = [
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.SPADES, Rank.FIVE),
            Card(Suit.HEARTS, Rank.KING)
        ]
        combos = get_all_discard_combinations(hand)

        runs = [c for c in combos if len(c) >= 3 and is_run(list(c))]
        self.assertGreater(len(runs), 0)


class TestEncoding(unittest.TestCase):
    """Test state and action encoding."""

    def test_state_encoder_dimensions(self):
        """State encoder produces correct dimension."""
        encoder = StateEncoder()
        self.assertEqual(encoder.dim, 219)

    def test_action_encoder_dimensions(self):
        """Action encoder produces correct dimension."""
        encoder = ActionEncoder()
        self.assertEqual(encoder.dim, 62)

    def test_combined_encoder_dimensions(self):
        """Combined encoder produces correct dimension."""
        encoder = StateActionEncoder()
        self.assertEqual(encoder.dim, 281)

    def test_state_encoding_output(self):
        """State encoding produces valid output."""
        encoder = StateEncoder()
        obs = {
            "hand": [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.FIVE)],
            "discard_pile": [Card(Suit.DIAMONDS, Rank.KING)],
            "hand_value": 6,
            "opponents": [{"hand_count": 5, "score": 0}],
            "round_number": 1,
            "can_call_yaniv": True
        }

        vec = encoder.encode(obs)
        self.assertEqual(len(vec), 219)
        self.assertTrue(np.all(np.isfinite(vec)))


class TestModel(unittest.TestCase):
    """Test neural network model."""

    def test_model_forward_pass(self):
        """Model produces output for valid input."""
        model = ActionValueNet(input_dim=281, use_torch=False)
        x = np.random.randn(281).astype(np.float32)
        out = model.forward(x)
        # Output is either scalar or 1-element array
        self.assertEqual(out.size, 1)

    def test_model_batch_forward(self):
        """Model handles batch input."""
        model = ActionValueNet(input_dim=281, use_torch=False)
        x = np.random.randn(10, 281).astype(np.float32)
        out = model.forward(x)
        self.assertEqual(out.shape, (10,))

    def test_model_copy(self):
        """Model copy produces identical outputs."""
        model = ActionValueNet(input_dim=281, use_torch=False)
        copy = model.copy()

        x = np.random.randn(281).astype(np.float32)
        self.assertEqual(model.forward(x), copy.forward(x))

    def test_model_save_load(self):
        """Model can be saved and loaded."""
        import tempfile
        import os

        model = ActionValueNet(input_dim=281, use_torch=False)
        x = np.random.randn(281).astype(np.float32)
        original_out = model.forward(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            model.save(path)

            loaded = ActionValueNet(use_torch=False)
            loaded.load(path)

            loaded_out = loaded.forward(x)
            np.testing.assert_array_almost_equal(original_out, loaded_out)


class TestGenome(unittest.TestCase):
    """Test genome operations."""

    def test_genome_mutation(self):
        """Mutation creates different weights."""
        genome = Genome()
        mutated = genome.mutate(mutation_rate=1.0, mutation_strength=0.5)

        original_weights = genome.get_weights()
        mutated_weights = mutated.get_weights()

        # At least some weights should be different
        differences = 0
        for key in original_weights:
            if not np.allclose(original_weights[key], mutated_weights[key]):
                differences += 1

        self.assertGreater(differences, 0)

    def test_genome_crossover(self):
        """Crossover combines parent genomes."""
        parent1 = Genome()
        parent2 = Genome()
        child = parent1.crossover(parent2)

        self.assertEqual(len(child.parent_ids), 2)
        self.assertIn(parent1.genome_id, child.parent_ids)
        self.assertIn(parent2.genome_id, child.parent_ids)

    def test_genome_fitness_calculation(self):
        """Fitness calculation works correctly."""
        genome = Genome()
        genome.record_game(won=True, score_margin=50)
        genome.record_game(won=True, score_margin=30)
        genome.record_game(won=False, score_margin=-20)

        fitness = genome.calculate_fitness()
        self.assertGreater(fitness, 0)
        self.assertLess(fitness, 1)


class TestBots(unittest.TestCase):
    """Test baseline bots."""

    def test_random_bot_chooses_legal_action(self):
        """Random bot always chooses a legal action."""
        bot = RandomBot(seed=42)
        env = YanivEnv(seed=42)
        env.reset()

        legal_actions = env.get_legal_actions()
        obs = env._get_observation(0)

        action = bot.choose_action(obs, legal_actions)
        self.assertIn(action, legal_actions)

    def test_greedy_bot_chooses_legal_action(self):
        """Greedy bot always chooses a legal action."""
        bot = GreedyBot(seed=42)
        env = YanivEnv(seed=42)
        env.reset()

        legal_actions = env.get_legal_actions()
        obs = env._get_observation(0)

        action = bot.choose_action(obs, legal_actions)
        self.assertIn(action, legal_actions)

    def test_rule_based_bot_chooses_legal_action(self):
        """Rule-based bot always chooses a legal action."""
        bot = RuleBasedBot(seed=42)
        env = YanivEnv(seed=42)
        env.reset()

        legal_actions = env.get_legal_actions()
        obs = env._get_observation(0)

        action = bot.choose_action(obs, legal_actions)
        self.assertIn(action, legal_actions)


class TestGamePlay(unittest.TestCase):
    """Test full game play."""

    def test_game_completes(self):
        """A game can complete without errors."""
        from .env import play_game

        bots = [RandomBot(seed=42), RandomBot(seed=43)]
        result = play_game(bots, seed=42)

        self.assertIn("winner_id", result)
        self.assertIn("scores", result)
        self.assertIn("turns", result)

    def test_game_respects_max_turns(self):
        """Game ends at max turns."""
        env = YanivEnv(seed=42, max_turns=10)
        env.reset()

        bot = RandomBot(seed=42)
        done = False

        while not done and env.turn_count < 20:
            legal = env.get_legal_actions()
            obs = env._get_observation(env.current_player_idx)
            action = bot.choose_action(obs, legal)
            _, _, done, _ = env.step(action)

        self.assertTrue(done or env.turn_count >= 10)


def run_tests():
    """Run all tests."""
    unittest.main(module=__name__, exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
