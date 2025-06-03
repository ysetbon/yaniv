// Create random weights for the enhanced neural network
// This creates a basic model that can be improved through training

const fs = require('fs');
const path = require('path');

function createRandomWeights() {
    // Layer sizes: 35 -> 128 -> 64 -> 32 -> 16
    const layers = [
        { input: 35, output: 128 },
        { input: 128, output: 64 },
        { input: 64, output: 32 },
        { input: 32, output: 16 }
    ];
    
    let weights = [];
    
    for (const layer of layers) {
        // Kernel weights (He initialization)
        const kernelSize = layer.input * layer.output;
        const std = Math.sqrt(2.0 / layer.input);
        
        for (let i = 0; i < kernelSize; i++) {
            weights.push((Math.random() - 0.5) * 2 * std);
        }
        
        // Bias weights (zeros)
        for (let i = 0; i < layer.output; i++) {
            weights.push(0.0);
        }
    }
    
    // Convert to Float32Array and then to Buffer
    const float32Array = new Float32Array(weights);
    const buffer = Buffer.from(float32Array.buffer);
    
    // Save to weights.bin
    const outputPath = path.join(__dirname, 'public', 'models', 'yaniv-enhanced', 'weights.bin');
    fs.writeFileSync(outputPath, buffer);
    
    console.log(`Created ${weights.length} weights (${buffer.length} bytes)`);
    console.log('Enhanced AI model files created successfully!');
    console.log('- Model: public/models/yaniv-enhanced/model.json');
    console.log('- Weights: public/models/yaniv-enhanced/weights.bin');
    
    return weights.length;
}

// Run if called directly
if (require.main === module) {
    createRandomWeights();
}

module.exports = { createRandomWeights };