import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';

// Node.js-specific save handler
export async function saveModelNode(model: tf.LayersModel, savePath: string) {
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
    
    fs.writeFileSync(
      path.join(cleanPath, 'model.json'),
      JSON.stringify(modelJSON, null, 2)
    );
    
    // Save weights
    if (artifacts.weightData) {
      let weightsBuffer: Buffer;
      
      if (artifacts.weightData instanceof ArrayBuffer) {
        weightsBuffer = Buffer.from(new Uint8Array(artifacts.weightData));
      } else if (Array.isArray(artifacts.weightData)) {
        // Handle array of ArrayBuffers
        const buffers = artifacts.weightData.map(ab => Buffer.from(new Uint8Array(ab)));
        weightsBuffer = Buffer.concat(buffers);
      } else {
        // Handle other formats
        weightsBuffer = Buffer.from(artifacts.weightData as any);
      }
      
      fs.writeFileSync(
        path.join(cleanPath, 'weights.bin'),
        weightsBuffer
      );
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