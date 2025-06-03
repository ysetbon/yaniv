"""Test script to compare enhanced AI with previous version"""

import numpy as np
import json
import time
from yaniv_neural_network_enhanced import EnhancedYanivNN, AdvancedFeatureExtractor, EnsembleAI
from yaniv_neural_network_optimized import YanivNNOptimized
from yaniv_game_ai_optimized import YanivGameOptimized
import matplotlib.pyplot as plt


def load_original_network(filename: str) -> YanivNNOptimized:
    """Load original network for comparison"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    network = YanivNNOptimized(
        input_size=data.get('input_size', 11),
        hidden_size=data.get('hidden_size', 64),
        output_size=data.get('output_size', 16)
    )
    
    # Load weights
    network.W1 = np.array(data['W1'])
    network.b1 = np.array(data['b1'])
    network.W2 = np.array(data['W2'])
    network.b2 = np.array(data['b2'])
    
    return network


def convert_game_state_for_enhanced(game_state):
    """Convert game state to work with enhanced feature extractor"""
    # Convert card format
    hand_cards = []
    for card in game_state.get('hand', []):
        hand_cards.append({
            'value': card.get('value', card),
            'suit': card.get('suit', 0)
        })
    
    enhanced_state = {
        'hand': hand_cards,
        'deck_size': game_state.get('deck_size', 52),
        'discard_top': game_state.get('discard_top', 0),
        'opponents': game_state.get('opponents', [])
    }
    
    return enhanced_state


def test_network_performance(network, network_type: str, num_games: int = 100):
    """Test a network's performance"""
    wins = 0
    total_turns = 0
    yaniv_calls = 0
    successful_yanivs = 0
    
    print(f"\nTesting {network_type}...")
    
    for game_num in range(num_games):
        game = YanivGameOptimized()
        
        # Create a simple opponent
        opponent = YanivNNOptimized()
        
        # Wrap networks based on type
        if network_type == "Enhanced":
            player1 = lambda state: enhanced_network_decision(network, state)
        elif network_type == "Ensemble":
            player1 = lambda state: ensemble_decision(network, state)
        else:
            player1 = network
        
        players = [player1, opponent]
        
        # Play game
        winner, game_stats = game.play_game_with_stats(players)
        
        if winner == 0:
            wins += 1
        
        total_turns += game_stats.get('turns', 0)
        yaniv_calls += game_stats.get('yaniv_calls', 0)
        successful_yanivs += game_stats.get('successful_yanivs', 0)
    
    win_rate = wins / num_games
    avg_turns = total_turns / num_games
    yaniv_rate = yaniv_calls / num_games
    yaniv_success_rate = successful_yanivs / yaniv_calls if yaniv_calls > 0 else 0
    
    print(f"Results for {network_type}:")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Average Turns: {avg_turns:.1f}")
    print(f"  Yaniv Call Rate: {yaniv_rate:.2%}")
    print(f"  Yaniv Success Rate: {yaniv_success_rate:.2%}")
    
    return {
        'win_rate': win_rate,
        'avg_turns': avg_turns,
        'yaniv_rate': yaniv_rate,
        'yaniv_success_rate': yaniv_success_rate
    }


def enhanced_network_decision(network: EnhancedYanivNN, game_state):
    """Make decision using enhanced network"""
    enhanced_state = convert_game_state_for_enhanced(game_state)
    features = AdvancedFeatureExtractor.extract_features(enhanced_state)
    
    # Get action probabilities
    action_probs = network.forward(features)
    
    # Sample action
    action = np.random.choice(len(action_probs), p=action_probs)
    
    return action


def ensemble_decision(ensemble: EnsembleAI, game_state):
    """Make decision using ensemble"""
    enhanced_state = convert_game_state_for_enhanced(game_state)
    action_probs = ensemble.predict(enhanced_state)
    
    # Sample action
    action = np.random.choice(len(action_probs), p=action_probs)
    
    return action


def compare_networks():
    """Compare different AI versions"""
    results = {}
    
    # Test original network
    try:
        original_network = load_original_network("saved_networks/best_overall_optimized.json")
        results['Original'] = test_network_performance(original_network, "Original", num_games=100)
    except Exception as e:
        print(f"Could not load original network: {e}")
    
    # Test enhanced network
    try:
        enhanced_network = EnhancedYanivNN.load("enhanced_yaniv_best.json")
        results['Enhanced'] = test_network_performance(enhanced_network, "Enhanced", num_games=100)
    except Exception as e:
        print(f"Could not load enhanced network: {e}")
        # Create and test a new one
        enhanced_network = EnhancedYanivNN()
        results['Enhanced'] = test_network_performance(enhanced_network, "Enhanced", num_games=100)
    
    # Test ensemble
    try:
        ensemble = EnsembleAI.load("enhanced_yaniv_ensemble.json")
        results['Ensemble'] = test_network_performance(ensemble, "Ensemble", num_games=100)
    except Exception as e:
        print(f"Could not load ensemble: {e}")
    
    # Visualize results
    if len(results) > 1:
        visualize_comparison(results)
    
    return results


def visualize_comparison(results):
    """Create comparison plots"""
    networks = list(results.keys())
    metrics = ['win_rate', 'avg_turns', 'yaniv_rate', 'yaniv_success_rate']
    metric_names = ['Win Rate', 'Avg Turns', 'Yaniv Call Rate', 'Yaniv Success Rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[net][metric] for net in networks]
        
        bars = axes[i].bar(networks, values, color=['blue', 'green', 'red'][:len(networks)])
        axes[i].set_title(name)
        axes[i].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}' if metric != 'win_rate' else f'{value:.2%}',
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_ai_comparison.png')
    plt.show()


def test_feature_importance():
    """Analyze feature importance for enhanced network"""
    try:
        network = EnhancedYanivNN.load("enhanced_yaniv_best.json")
    except:
        network = EnhancedYanivNN()
    
    # Create sample game states with variations
    base_state = {
        'hand': [{'value': 5, 'suit': 0}, {'value': 7, 'suit': 1}, 
                 {'value': 9, 'suit': 2}, {'value': 3, 'suit': 3},
                 {'value': 11, 'suit': 0}],
        'deck_size': 30,
        'discard_top': 6,
        'opponents': [{'cards': 4}, {'cards': 5}, {'cards': 3}]
    }
    
    # Get base features and prediction
    base_features = AdvancedFeatureExtractor.extract_features(base_state)
    base_prediction = network.forward(base_features)
    
    feature_importance = []
    feature_names = [
        'Card 1', 'Card 2', 'Card 3', 'Card 4', 'Card 5',
        'Hand Value', 'Suit 0', 'Suit 1', 'Suit 2', 'Suit 3',
        'Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5',
        'Rank 6', 'Rank 7', 'Rank 8', 'Rank 9', 'Rank 10',
        'Rank J', 'Rank Q', 'Rank K',
        'Potential Sets', 'Potential Runs', 'Potential Pairs',
        'Deck Size', 'Discard Top',
        'Opponent 1 Cards', 'Opponent 2 Cards', 'Opponent 3 Cards',
        'Game Phase', 'Can Call Yaniv'
    ]
    
    # Perturb each feature and measure impact
    for i in range(len(base_features)):
        perturbed_features = base_features.copy()
        perturbed_features[i] = 0  # Zero out feature
        
        perturbed_prediction = network.forward(perturbed_features)
        
        # Measure change in prediction
        importance = np.sum(np.abs(base_prediction - perturbed_prediction))
        feature_importance.append(importance)
    
    # Sort and display top features
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    print("\nTop 10 Most Important Features:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        if idx < len(feature_names):
            print(f"  {feature_names[idx]}: {feature_importance[idx]:.3f}")


if __name__ == "__main__":
    print("Testing Enhanced Yaniv AI Implementation")
    print("=" * 50)
    
    # Run comparison
    results = compare_networks()
    
    # Test feature importance
    test_feature_importance()
    
    print("\nTest completed!")