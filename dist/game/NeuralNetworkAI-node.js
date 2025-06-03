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
exports.saveModelNode = saveModelNode;
const tf = __importStar(require("@tensorflow/tfjs"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
// Node.js-specific save handler
async function saveModelNode(model, savePath) {
    // Remove 'file://' prefix if present
    const cleanPath = savePath.replace('file://', '');
    // Create directory if it doesn't exist
    const dir = path.dirname(cleanPath);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    // Save model using TensorFlow.js io handlers
    const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
        // Save model.json
        const modelJSON = {
            modelTopology: artifacts.modelTopology,
            weightsManifest: [{
                    paths: ['weights.bin'],
                    weights: artifacts.weightSpecs
                }],
            format: artifacts.format,
            generatedBy: artifacts.generatedBy,
            convertedBy: artifacts.convertedBy
        };
        fs.writeFileSync(path.join(cleanPath, 'model.json'), JSON.stringify(modelJSON, null, 2));
        // Save weights
        if (artifacts.weightData) {
            let weightsBuffer;
            if (artifacts.weightData instanceof ArrayBuffer) {
                weightsBuffer = Buffer.from(new Uint8Array(artifacts.weightData));
            }
            else if (Array.isArray(artifacts.weightData)) {
                // Handle array of ArrayBuffers
                const buffers = artifacts.weightData.map(ab => Buffer.from(new Uint8Array(ab)));
                weightsBuffer = Buffer.concat(buffers);
            }
            else {
                // Handle other formats
                weightsBuffer = Buffer.from(artifacts.weightData);
            }
            fs.writeFileSync(path.join(cleanPath, 'weights.bin'), weightsBuffer);
        }
        return {
            modelArtifactsInfo: {
                dateSaved: new Date(),
                modelTopologyType: 'JSON',
            }
        };
    }));
    return saveResult;
}
