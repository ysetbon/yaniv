#!/usr/bin/env python3
"""
Convert Python-trained Yaniv AI model to TensorFlow.js format
"""

import json
import numpy as np
import os

def convert_python_to_tfjs(input_file, output_dir):
    """Convert Python neural network JSON to TensorFlow.js compatible format"""
    
    # Load the Python model
    with open(input_file, 'r') as f:
        model_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert weights to TensorFlow.js format
    # The Python model has 2 layers: input->hidden and hidden->output
    w1 = np.array(model_data['w1'], dtype=np.float32)  # Shape: (11, 64)
    b1 = np.array(model_data['b1'], dtype=np.float32)  # Shape: (64,)
    w2 = np.array(model_data['w2'], dtype=np.float32)  # Shape: (64, 16)
    b2 = np.array(model_data['b2'], dtype=np.float32)  # Shape: (16,)
    
    # TensorFlow.js expects weights in a specific format
    weights_manifest = {
        "format": "layers-model",
        "generatedBy": "Yaniv AI Converter",
        "convertedBy": "Python to TFJS",
        "modelTopology": {
            "keras_version": "2.11.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": "yaniv_ai_model",
                    "layers": [
                        {
                            "class_name": "InputLayer",
                            "config": {
                                "batch_input_shape": [None, 11],
                                "dtype": "float32",
                                "sparse": False,
                                "ragged": False,
                                "name": "input_layer"
                            }
                        },
                        {
                            "class_name": "Dense",
                            "config": {
                                "name": "dense_1",
                                "trainable": True,
                                "dtype": "float32",
                                "units": 64,
                                "activation": "relu",
                                "use_bias": True,
                                "kernel_initializer": {"class_name": "GlorotUniform"},
                                "bias_initializer": {"class_name": "Zeros"}
                            }
                        },
                        {
                            "class_name": "Dense",
                            "config": {
                                "name": "dense_2",
                                "trainable": True,
                                "dtype": "float32",
                                "units": 16,
                                "activation": "softmax",
                                "use_bias": True,
                                "kernel_initializer": {"class_name": "GlorotUniform"},
                                "bias_initializer": {"class_name": "Zeros"}
                            }
                        }
                    ]
                }
            }
        },
        "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": [
                {
                    "name": "dense_1/kernel",
                    "shape": [11, 64],
                    "dtype": "float32"
                },
                {
                    "name": "dense_1/bias",
                    "shape": [64],
                    "dtype": "float32"
                },
                {
                    "name": "dense_2/kernel",
                    "shape": [64, 16],
                    "dtype": "float32"
                },
                {
                    "name": "dense_2/bias",
                    "shape": [16],
                    "dtype": "float32"
                }
            ]
        }]
    }
    
    # Save model.json
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(weights_manifest, f, indent=2)
    
    # Save weights as binary file
    # Concatenate all weights in the order specified in the manifest
    all_weights = np.concatenate([
        w1.transpose().flatten(),  # TensorFlow uses different weight ordering
        b1.flatten(),
        w2.transpose().flatten(),
        b2.flatten()
    ])
    
    # Save as binary file
    all_weights.astype(np.float32).tofile(os.path.join(output_dir, 'weights.bin'))
    
    # Also save metadata
    metadata = {
        "fitness": model_data.get("fitness", 0),
        "wins": model_data.get("wins", 0),
        "games_played": model_data.get("games_played", 0),
        "network_id": model_data.get("network_id", 0),
        "input_size": 11,
        "hidden_size": 64,
        "output_size": 16
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model converted successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Fitness: {metadata['fitness']:.3f}")
    print(f"Win rate: {metadata['wins']}/{metadata['games_played']} games")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Python Yaniv AI model to TensorFlow.js format")
    parser.add_argument("--input", default="saved_networks/best_overall_optimized.json",
                        help="Path to input Python model JSON file")
    parser.add_argument("--output", default="public/models/yaniv-trained-optimized",
                        help="Output directory for TensorFlow.js model")
    
    args = parser.parse_args()
    
    convert_python_to_tfjs(args.input, args.output)