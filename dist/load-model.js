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
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
async function convertModelForWeb(modelPath, outputDir) {
    console.log(`Loading model from: ${modelPath}`);
    // Load the model
    const model = await tf.loadLayersModel(modelPath);
    // Create output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    // Save in web format
    const webPath = `file://${path.resolve(outputDir)}`;
    await model.save(webPath);
    console.log(`Model converted and saved to: ${outputDir}`);
    console.log('\nTo use in your web app:');
    console.log('1. Copy the model files to your public directory');
    console.log('2. Load with: await tf.loadLayersModel("/path/to/model.json")');
}
// Usage
const args = process.argv.slice(2);
if (args.length < 2) {
    console.log('Usage: npm run load-model <model-path> <output-dir>');
    console.log('Example: npm run load-model file://./yaniv-ai-model-final ./public/models/yaniv-ai');
    process.exit(1);
}
convertModelForWeb(args[0], args[1]).catch(console.error);
