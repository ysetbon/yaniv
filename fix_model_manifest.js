// Fix the model.json to include weights manifest
const fs = require('fs');
const path = require('path');

function fixModelManifest() {
    console.log('Fixing Enhanced AI model manifest...');
    
    const modelPath = path.join(__dirname, 'public', 'models', 'yaniv-enhanced', 'model.json');
    
    try {
        // Read current model
        const modelData = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
        
        // Add weights manifest
        modelData.weightsManifest = [{
            "paths": ["weights.bin"],
            "weights": [
                {
                    "name": "dense_1/kernel",
                    "shape": [35, 128],
                    "dtype": "float32"
                },
                {
                    "name": "dense_1/bias",
                    "shape": [128],
                    "dtype": "float32"
                },
                {
                    "name": "dense_2/kernel",
                    "shape": [128, 64],
                    "dtype": "float32"
                },
                {
                    "name": "dense_2/bias",
                    "shape": [64],
                    "dtype": "float32"
                },
                {
                    "name": "dense_3/kernel",
                    "shape": [64, 32],
                    "dtype": "float32"
                },
                {
                    "name": "dense_3/bias",
                    "shape": [32],
                    "dtype": "float32"
                },
                {
                    "name": "output/kernel",
                    "shape": [32, 16],
                    "dtype": "float32"
                },
                {
                    "name": "output/bias",
                    "shape": [16],
                    "dtype": "float32"
                }
            ]
        }];
        
        // Save fixed model
        fs.writeFileSync(modelPath, JSON.stringify(modelData, null, 2));
        
        console.log('✅ Model manifest fixed!');
        console.log('✅ Enhanced AI should now work properly');
        console.log('\nRefresh your browser to see the improved AI in action!');
        
    } catch (error) {
        console.error('Error fixing model:', error);
    }
}

// Run the fix
fixModelManifest();