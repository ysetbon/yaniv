#!/usr/bin/env node
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const tf = __importStar(require("@tensorflow/tfjs"));
require("@tensorflow/tfjs-backend-cpu");
const TrainingSystemV2_1 = require("./game/TrainingSystemV2");
const NeuralNetworkAI_node_1 = require("./game/NeuralNetworkAI-node");
const readline = __importStar(require("readline"));
const fs = __importStar(require("fs"));
// Set up TensorFlow.js backend
async function setupTF() {
    await tf.ready();
    console.log('TensorFlow.js backend:', tf.getBackend());
}
// Initialize TensorFlow
setupTF();
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});
function question(prompt) {
    return new Promise(resolve => {
        rl.question(prompt, resolve);
    });
}
async function main() {
    // Ensure TensorFlow is ready
    await tf.ready();
    console.log('=== Yaniv AI Training System ===\n');
    // Get training parameters
    const episodes = parseInt(await question('Number of training episodes (default 1000): ') || '1000');
    const trainEvery = parseInt(await question('Train every N episodes (default 10): ') || '10');
    const saveEvery = parseInt(await question('Save model every N episodes (default 50): ') || '50');
    console.log(`\nStarting training with:`);
    console.log(`- Episodes: ${episodes}`);
    console.log(`- Train every: ${trainEvery} episodes`);
    console.log(`- Save every: ${saveEvery} episodes\n`);
    rl.close();
    const trainingSystem = new TrainingSystemV2_1.TrainingSystemV2();
    let lastSaveTime = Date.now();
    // Override the training loop to add progress reporting
    for (let i = 0; i < episodes; i++) {
        // Run self-play episode
        await trainingSystem.runSelfPlayEpisode();
        // Train periodically
        if ((i + 1) % trainEvery === 0) {
            const avgLoss = await trainingSystem.train();
            const progress = ((i + 1) / episodes * 100).toFixed(1);
            console.log(`[${progress}%] Episode ${i + 1}/${episodes} - Loss: ${avgLoss.toFixed(4)}`);
        }
        // Save model periodically
        if ((i + 1) % saveEvery === 0) {
            const modelDir = `./models/yaniv-ai-model-${i + 1}`;
            // Create directory
            if (!fs.existsSync(modelDir)) {
                fs.mkdirSync(modelDir, { recursive: true });
            }
            // Save using Node.js handler
            const model = trainingSystem.getNeuralAI()['model'];
            if (model) {
                await (0, NeuralNetworkAI_node_1.saveModelNode)(model, modelDir);
                const timeSinceLastSave = ((Date.now() - lastSaveTime) / 1000).toFixed(1);
                console.log(`✓ Model checkpoint saved to ${modelDir} (${timeSinceLastSave}s)`);
                lastSaveTime = Date.now();
            }
        }
        // Show progress bar every 10 episodes
        if ((i + 1) % 10 === 0 && (i + 1) % trainEvery !== 0) {
            const progress = ((i + 1) / episodes * 100).toFixed(1);
            const barLength = 50;
            const filledLength = Math.floor(barLength * (i + 1) / episodes);
            const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);
            process.stdout.write(`\r[${bar}] ${progress}%`);
        }
    }
    // Final save
    const finalModelDir = './models/yaniv-ai-model-final';
    if (!fs.existsSync(finalModelDir)) {
        fs.mkdirSync(finalModelDir, { recursive: true });
    }
    const model = trainingSystem.getNeuralAI()['model'];
    if (model) {
        await (0, NeuralNetworkAI_node_1.saveModelNode)(model, finalModelDir);
    }
    console.log(`\n\n✅ Training complete!`);
    console.log(`Final model saved to: ${finalModelDir}`);
    console.log(`\nTo use the trained model, copy the model files to your web app.`);
    console.log(`Model files: ${finalModelDir}/model.json and ${finalModelDir}/weights.bin`);
    process.exit(0);
}
// Handle errors
process.on('unhandledRejection', (error) => {
    console.error('\n❌ Training error:', error);
    process.exit(1);
});
// Run the training
main().catch(console.error);
