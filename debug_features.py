#!/usr/bin/env python3
"""
Debug the feature extraction to fix dimension mismatch
"""

def debug_feature_extraction():
    """Debug what features are being extracted"""
    
    # Simulate the feature extraction
    features = []
    
    print("Enhanced AI Feature Extraction Debug")
    print("=" * 40)
    
    # Basic hand features (5)
    print("1. Hand cards (5 features):")
    hand = [5, 7, 9, 3, 11]  # Example hand
    for i in range(5):
        if i < len(hand):
            features.append(hand[i] / 13.0)
            print(f"   Card {i+1}: {hand[i]} -> {hand[i] / 13.0:.3f}")
        else:
            features.append(0.0)
            print(f"   Card {i+1}: 0 -> 0.000")
    
    # Hand value (1)
    print("2. Hand value (1 feature):")
    hand_value = sum(hand)
    features.append(hand_value / 50.0)
    print(f"   Hand value: {hand_value} -> {hand_value / 50.0:.3f}")
    
    # Suit distribution (4)
    print("3. Suit distribution (4 features):")
    suit_features = [0.25, 0.25, 0.25, 0.25]
    features.extend(suit_features)
    for i, sf in enumerate(suit_features):
        print(f"   Suit {i}: {sf}")
    
    # Rank distribution (13)
    print("4. Rank distribution (13 features):")
    rank_features = [0.08] * 13
    features.extend(rank_features)
    for i, rf in enumerate(rank_features):
        print(f"   Rank {i+1}: {rf}")
    
    # Potential combinations (3)
    print("5. Potential combinations (3 features):")
    combo_features = [0.1, 0.1, 0.1]
    features.extend(combo_features)
    combo_names = ["Sets", "Runs", "Pairs"]
    for name, cf in zip(combo_names, combo_features):
        print(f"   {name}: {cf}")
    
    # Game state features (2)
    print("6. Game state (2 features):")
    deck_size = 30
    last_discard = 6
    features.append(deck_size / 52.0)
    features.append(last_discard / 13.0)
    print(f"   Deck size: {deck_size} -> {deck_size / 52.0:.3f}")
    print(f"   Last discard: {last_discard} -> {last_discard / 13.0:.3f}")
    
    # Opponent information (3)
    print("7. Opponent info (3 features):")
    opponent_cards = [4, 5, 3]
    for i in range(3):
        if i < len(opponent_cards):
            features.append(opponent_cards[i] / 10.0)
            print(f"   Opponent {i+1}: {opponent_cards[i]} -> {opponent_cards[i] / 10.0:.3f}")
        else:
            features.append(0.0)
            print(f"   Opponent {i+1}: 0 -> 0.000")
    
    # Game phase (1)
    print("8. Game phase (1 feature):")
    game_phase = 0.5  # Mid game
    features.append(game_phase)
    print(f"   Phase: {game_phase}")
    
    # Can call Yaniv (1)
    print("9. Can call Yaniv (1 feature):")
    can_yaniv = 0.0  # Cannot call
    features.append(can_yaniv)
    print(f"   Can Yaniv: {can_yaniv}")
    
    print(f"\nTOTAL FEATURES: {len(features)}")
    print(f"EXPECTED: 35")
    print(f"DIFFERENCE: {35 - len(features)}")
    
    if len(features) != 35:
        print(f"\n❌ MISMATCH FOUND!")
        print(f"Need to add {35 - len(features)} more features")
    else:
        print(f"\n✅ Perfect match!")
    
    # Feature breakdown
    feature_counts = [5, 1, 4, 13, 3, 2, 3, 1, 1]
    feature_names = ["Hand cards", "Hand value", "Suit dist", "Rank dist", 
                    "Combinations", "Game state", "Opponents", "Game phase", "Can Yaniv"]
    
    print(f"\nFeature breakdown:")
    total_check = 0
    for name, count in zip(feature_names, feature_counts):
        print(f"  {name}: {count}")
        total_check += count
    print(f"  TOTAL: {total_check}")
    
    return features

def fix_feature_extraction():
    """Create the corrected feature extraction"""
    
    features = debug_feature_extraction()
    
    if len(features) != 35:
        missing = 35 - len(features)
        print(f"\nAdding {missing} padding features...")
        
        # Add padding features
        for i in range(missing):
            features.append(0.0)
            print(f"  Padding {i+1}: 0.0")
        
        print(f"Final feature count: {len(features)}")
    
    return features

if __name__ == "__main__":
    fix_feature_extraction()