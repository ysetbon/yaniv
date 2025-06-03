#!/usr/bin/env python3
"""
Test the trained Yaniv AI
"""

from yaniv_neural_network import YanivNeuralNetwork
from yaniv_game_ai import YanivGameAI
import os

def test_single_game():
    """Test a single game between two AI players"""
    print("Testing single game between two random AI players...")
    
    # Create two random AI players
    player1 = YanivNeuralNetwork(network_id=1)
    player2 = YanivNeuralNetwork(network_id=2)
    
    # Play game
    game = YanivGameAI()
    winner = game.play_game([player1, player2])
    
    print(f"Game finished! Winner: Network {winner.network_id}")
    print(f"Final scores:")
    for i, player in enumerate(game.players):
        print(f"  Player {i}: {player['score']} points")

def test_trained_ai():
    """Test the best trained AI"""
    if not os.path.exists('saved_networks/best_overall.json'):
        print("No trained network found. Please run train_ai.py first.")
        return
    
    print("\nTesting trained AI...")
    
    # Load trained AI
    trained_ai = YanivNeuralNetwork(network_id=0)
    trained_ai.load('saved_networks/best_overall.json')
    
    # Create random opponent
    random_ai = YanivNeuralNetwork(network_id=1)
    
    # Play multiple games
    trained_wins = 0
    games_to_play = 100
    
    print(f"Playing {games_to_play} games: Trained AI vs Random AI")
    
    for i in range(games_to_play):
        game = YanivGameAI()
        
        # Alternate who goes first
        if i % 2 == 0:
            winner = game.play_game([trained_ai, random_ai])
        else:
            winner = game.play_game([random_ai, trained_ai])
        
        if winner == trained_ai:
            trained_wins += 1
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{games_to_play} games")
    
    win_rate = trained_wins / games_to_play
    print(f"\nResults:")
    print(f"Trained AI win rate: {win_rate:.2%}")
    print(f"Random AI win rate: {(1 - win_rate):.2%}")

def main():
    print("Yaniv AI Test Suite")
    print("=" * 50)
    
    # Test single game
    test_single_game()
    
    print("\n" + "=" * 50)
    
    # Test trained AI
    test_trained_ai()

if __name__ == "__main__":
    main()