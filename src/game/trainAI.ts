import { TrainingSystemV2 } from './TrainingSystemV2';

export async function trainAI(episodes: number = 1000, trainEvery: number = 10) {
  console.log('Starting Yaniv AI training...');
  
  const trainingSystem = new TrainingSystemV2();
  
  // Run training loop
  await trainingSystem.runTrainingLoop(episodes, trainEvery);
  
  console.log('Training completed!');
  console.log('Model saved to local storage as "yaniv-ai-model"');
  
  return trainingSystem.getNeuralAI();
}

// Function to run training in the browser console
if (typeof window !== 'undefined') {
  (window as any).trainYanivAI = trainAI;
  console.log('Training function available: call window.trainYanivAI() to start training');
}