#!/usr/bin/env python3
"""
Test that the feature extraction fix works
"""

from yaniv_game_enhanced_wrapper import EnhancedNetworkWrapper
from yaniv_neural_network_enhanced import EnhancedYanivNN
import numpy as np

def test_feature_dimensions():
    """Test that we now get exactly 35 features"""
    
    print("Testing Feature Extraction Fix")
    print("=" * 40)
    
    # Create enhanced network
    network = EnhancedYanivNN()
    wrapper = EnhancedNetworkWrapper(network)
    
    # Create test game state
    test_game_state = {
        'hand': [{'value': 5}, {'value': 7}, {'value': 9}],
        'hand_value': 21,
        'deck_size': 30,
        'opponent_cards': [4, 5],
        'last_discard': {'value': 6}
    }
    
    # Extract features
    features = wrapper._convert_to_enhanced_features(test_game_state)
    
    print(f"Feature array shape: {features.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Expected: 35")
    
    if len(features) == 35:
        print("✅ SUCCESS! Feature extraction now works correctly")
        
        # Test forward pass
        try:
            output = wrapper.forward(test_game_state)
            print(f"✅ Forward pass successful")
            print(f"Output shape: {output.shape}")
            print(f"Output sum: {np.sum(output):.3f} (should be ~1.0 for softmax)")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
            
    else:
        print(f"❌ FAILED! Still have {len(features)} features instead of 35")
        return False
    
    print(f"\n🚀 Enhanced AI training should now work!")
    return True

if __name__ == "__main__":
    test_feature_dimensions()