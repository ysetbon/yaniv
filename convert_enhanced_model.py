"""
Convert Enhanced Neural Network model from Python to TensorFlow.js format
"""

import json
import numpy as np
import os
from yaniv_neural_network_enhanced import EnhancedYanivNN

def convert_enhanced_model_to_tfjs(model_path, output_dir):
    """Convert enhanced Python model to TensorFlow.js format"""
    
    # Load the model
    if model_path.endswith('.json'):
        network = EnhancedYanivNN.load(model_path)
    else:
        # Create a new enhanced network with default weights
        network = EnhancedYanivNN()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TensorFlow.js model structure
    model_config = {
        "modelTopology": {
            "class_name": "Sequential",
            "config": {
                "name": "enhanced_yaniv_model",
                "layers": []
            },
            "keras_version": "2.11.0",
            "backend": "tensorflow"
        },
        "format": "layers-model",
        "generatedBy": "Python Enhanced Model Converter",
        "convertedBy": "TensorFlow.js Converter"
    }
    
    # Add layers to config
    layers = []
    
    # Input layer
    layers.append({
        "class_name": "InputLayer",
        "config": {
            "batch_input_shape": [None, 35],
            "dtype": "float32",
            "sparse": False,
            "name": "input_layer"
        }
    })
    
    # Hidden layers
    layer_names = ["dense_1", "dense_2", "dense_3"]
    activations = ["relu", "relu", "relu"]
    
    for i, (layer_data, name, activation) in enumerate(zip(network.layers[:-1], layer_names, activations)):
        layers.append({
            "class_name": "Dense",
            "config": {
                "name": name,
                "trainable": True,
                "dtype": "float32",
                "units": layer_data['weights'].shape[1],
                "activation": activation,
                "use_bias": True,
                "kernel_initializer": {"class_name": "GlorotUniform"},
                "bias_initializer": {"class_name": "Zeros"}
            }
        })
    
    # Output layer
    output_layer = network.layers[-1]
    layers.append({
        "class_name": "Dense",
        "config": {
            "name": "output",
            "trainable": True,
            "dtype": "float32",
            "units": output_layer['weights'].shape[1],
            "activation": "softmax",
            "use_bias": True,
            "kernel_initializer": {"class_name": "GlorotUniform"},
            "bias_initializer": {"class_name": "Zeros"}
        }
    })
    
    model_config["modelTopology"]["config"]["layers"] = layers
    
    # Save model.json
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(model_config, f)
    
    # Prepare weights
    weights_data = []
    
    # Add weights for each layer (transpose for TensorFlow.js)
    for i, layer in enumerate(network.layers):
        # Kernel weights (transposed)
        kernel = layer['weights'].T.astype(np.float32)
        weights_data.append(kernel.flatten())
        
        # Bias weights
        bias = layer['bias'].flatten().astype(np.float32)
        weights_data.append(bias)
    
    # Note: Skip connections are not directly supported in simple Sequential model
    # They would need a Functional API model in TensorFlow.js
    # For now, we'll create a standard feedforward network
    
    # Concatenate all weights
    all_weights = np.concatenate(weights_data)
    
    # Save weights.bin
    with open(os.path.join(output_dir, 'weights.bin'), 'wb') as f:
        f.write(all_weights.tobytes())
    
    # Create weight manifest
    weight_specs = []
    offset = 0
    
    # Add weight specs for each layer
    for i, (layer, name) in enumerate(zip(network.layers, layer_names + ["output"])):
        # Kernel
        kernel_shape = [layer['weights'].shape[0], layer['weights'].shape[1]]
        kernel_size = np.prod(kernel_shape) * 4  # float32 = 4 bytes
        weight_specs.append({
            "name": f"{name}/kernel",
            "shape": kernel_shape,
            "dtype": "float32"
        })
        
        # Bias
        bias_shape = [layer['bias'].shape[1]]
        bias_size = np.prod(bias_shape) * 4
        weight_specs.append({
            "name": f"{name}/bias",
            "shape": bias_shape,
            "dtype": "float32"
        })
    
    # Update model.json with weight manifest
    with open(os.path.join(output_dir, 'model.json'), 'r') as f:
        model_json = json.load(f)
    
    model_json["weightsManifest"] = [{
        "paths": ["weights.bin"],
        "weights": weight_specs
    }]
    
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"Model converted successfully!")
    print(f"Files created in {output_dir}:")
    print("  - model.json")
    print("  - weights.bin")
    print(f"Model architecture: 35 inputs -> {network.hidden_sizes} -> 16 outputs")


def create_sample_enhanced_model():
    """Create a sample enhanced model for testing"""
    network = EnhancedYanivNN()
    
    # Save it
    network.save("sample_enhanced_model.json")
    print("Created sample enhanced model: sample_enhanced_model.json")
    
    return network


if __name__ == "__main__":
    # Create output directory for web
    output_dir = "public/models/yaniv-enhanced"
    
    # Check if we have a trained enhanced model
    if os.path.exists("enhanced_yaniv_best.json"):
        print("Converting trained enhanced model...")
        convert_enhanced_model_to_tfjs("enhanced_yaniv_best.json", output_dir)
    elif os.path.exists("quick_enhanced_best.json"):
        print("Converting quick-trained enhanced model...")
        convert_enhanced_model_to_tfjs("quick_enhanced_best.json", output_dir)
    else:
        print("No trained enhanced model found. Creating sample model...")
        network = create_sample_enhanced_model()
        convert_enhanced_model_to_tfjs("sample_enhanced_model.json", output_dir)
    
    print("\nTo use the enhanced AI in the game:")
    print("1. The enhanced AI is now the default option in the dropdown")
    print("2. The model files are in public/models/yaniv-enhanced/")
    print("3. Run 'npm run dev' to start the game")