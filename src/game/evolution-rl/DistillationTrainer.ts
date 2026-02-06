/**
 * Phase 2: Distillation
 *
 * Takes the elite policies from Phase 1 and:
 * 1. Runs them through many games, recording (state, action) pairs
 * 2. Weights training data by policy fitness score
 * 3. Trains a neural network via supervised learning (imitation learning)
 *
 * The neural network architecture:
 * - Input: 113-dim state vector
 * - Hidden: 128 → 64 (ReLU)
 * - Output heads:
 *   - card_logits: 52 (one per card, for discard scoring)
 *   - yaniv_logit: 1 (call Yaniv score)
 *   - draw_logit: 1 (draw from discard preference)
 *   - value: 1 (state value estimate for PPO)
 */

import * as tf from '@tensorflow/tfjs';
import { GeneticPolicy } from './GeneticPolicy';
import { GameSimulator, TrajectoryStep } from './GameSimulator';
import { STATE_SIZE, cardToIndex, enumerateValidDiscards } from './StateEncoder';

// NN output head sizes
export const CARD_LOGITS_SIZE = 52;
export const YANIV_LOGIT_SIZE = 1;
export const DRAW_LOGIT_SIZE = 1;
export const VALUE_SIZE = 1;

export interface DistillationConfig {
  gamesPerElite: number;      // Games each elite plays for data (default: 500)
  batchSize: number;           // Training batch size (default: 64)
  epochs: number;              // Training epochs (default: 30)
  learningRate: number;        // Adam learning rate (default: 0.001)
  hiddenSize1: number;         // First hidden layer size (default: 128)
  hiddenSize2: number;         // Second hidden layer size (default: 64)
}

export const DEFAULT_DISTILLATION_CONFIG: DistillationConfig = {
  gamesPerElite: 500,
  batchSize: 64,
  epochs: 30,
  learningRate: 0.001,
  hiddenSize1: 128,
  hiddenSize2: 64,
};

export interface DistillationStats {
  totalSamples: number;
  discardSamples: number;
  drawSamples: number;
  epoch: number;
  discardLoss: number;
  drawLoss: number;
}

export type DistillationProgressCallback = (stats: DistillationStats) => void;

/** A training sample for the neural network */
interface TrainingSample {
  state: Float32Array;
  // For discard phase:
  isDiscardPhase: boolean;
  discardCardsMask: boolean[];    // 52-dim: which cards were discarded
  isYaniv: boolean;
  validComboMasks: boolean[][];   // List of 52-dim masks for each valid combo
  canCallYaniv: boolean;
  // For draw phase:
  drewFromDiscard: boolean;
  // Weight (from elite fitness)
  weight: number;
}

export class DistillationTrainer {
  private config: DistillationConfig;
  model: tf.LayersModel | null = null;

  constructor(config?: Partial<DistillationConfig>) {
    this.config = { ...DEFAULT_DISTILLATION_CONFIG, ...config };
  }

  /**
   * Build the neural network model.
   */
  buildModel(): tf.LayersModel {
    const input = tf.input({ shape: [STATE_SIZE] });

    // Shared hidden layers
    let x = tf.layers.dense({
      units: this.config.hiddenSize1,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'hidden1',
    }).apply(input) as tf.SymbolicTensor;

    x = tf.layers.dense({
      units: this.config.hiddenSize2,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'hidden2',
    }).apply(x) as tf.SymbolicTensor;

    // Policy head: card logits (52 outputs, no activation - raw logits)
    const cardLogits = tf.layers.dense({
      units: CARD_LOGITS_SIZE,
      name: 'card_logits',
    }).apply(x) as tf.SymbolicTensor;

    // Policy head: yaniv logit (1 output)
    const yanivLogit = tf.layers.dense({
      units: YANIV_LOGIT_SIZE,
      name: 'yaniv_logit',
    }).apply(x) as tf.SymbolicTensor;

    // Policy head: draw logit (1 output)
    const drawLogit = tf.layers.dense({
      units: DRAW_LOGIT_SIZE,
      name: 'draw_logit',
    }).apply(x) as tf.SymbolicTensor;

    // Value head (1 output, tanh for [-1, 1] range)
    const value = tf.layers.dense({
      units: VALUE_SIZE,
      activation: 'tanh',
      name: 'value_head',
    }).apply(x) as tf.SymbolicTensor;

    this.model = tf.model({
      inputs: input,
      outputs: [cardLogits, yanivLogit, drawLogit, value],
    });

    return this.model;
  }

  /**
   * Collect training data by running elite policies through games.
   * Weight samples by each elite's fitness score.
   */
  collectTrainingData(
    elites: GeneticPolicy[],
  ): TrainingSample[] {
    const samples: TrainingSample[] = [];
    const totalFitness = elites.reduce((s, e) => s + e.fitness, 0);

    for (const elite of elites) {
      const weight = totalFitness > 0 ? elite.fitness / totalFitness * elites.length : 1;

      for (let g = 0; g < this.config.gamesPerElite; g++) {
        // Elite plays against another random elite (diversity)
        const opponent = elites[Math.floor(Math.random() * elites.length)];
        const result = GameSimulator.playGame(elite, opponent, true);

        // Collect from player 0's trajectory (the elite we're distilling)
        for (const step of result.trajectory0) {
          const sample = this.trajectoryStepToSample(step, weight);
          if (sample) samples.push(sample);
        }
      }
    }

    return samples;
  }

  private trajectoryStepToSample(step: TrajectoryStep, weight: number): TrainingSample | null {
    const { state, snapshot, action } = step;

    if (action.phase === 'discard') {
      const validCombos = enumerateValidDiscards(snapshot.hand);
      const validComboMasks = validCombos.map(combo => {
        const mask = new Array(52).fill(false);
        for (const card of combo) mask[cardToIndex(card)] = true;
        return mask;
      });

      return {
        state,
        isDiscardPhase: true,
        discardCardsMask: action.cardsMask || new Array(52).fill(false),
        isYaniv: action.type === 'yaniv',
        validComboMasks,
        canCallYaniv: snapshot.canCallYaniv,
        drewFromDiscard: false,
        weight,
      };
    } else {
      return {
        state,
        isDiscardPhase: false,
        discardCardsMask: new Array(52).fill(false),
        isYaniv: false,
        validComboMasks: [],
        canCallYaniv: false,
        drewFromDiscard: action.type === 'draw_discard',
        weight,
      };
    }
  }

  /**
   * Train the neural network on collected samples via supervised learning.
   */
  async train(
    samples: TrainingSample[],
    onProgress?: DistillationProgressCallback,
  ): Promise<void> {
    if (!this.model) this.buildModel();

    const discardSamples = samples.filter(s => s.isDiscardPhase);
    const drawSamples = samples.filter(s => !s.isDiscardPhase);

    const optimizer = tf.train.adam(this.config.learningRate);

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      // Shuffle samples
      this.shuffleArray(discardSamples);
      this.shuffleArray(drawSamples);

      let epochDiscardLoss = 0;
      let epochDrawLoss = 0;
      let discardBatches = 0;
      let drawBatches = 0;

      // Train discard head
      for (let i = 0; i < discardSamples.length; i += this.config.batchSize) {
        const batch = discardSamples.slice(i, i + this.config.batchSize);
        const loss = this.trainDiscardBatch(batch, optimizer);
        epochDiscardLoss += loss;
        discardBatches++;
      }

      // Train draw head
      for (let i = 0; i < drawSamples.length; i += this.config.batchSize) {
        const batch = drawSamples.slice(i, i + this.config.batchSize);
        const loss = this.trainDrawBatch(batch, optimizer);
        epochDrawLoss += loss;
        drawBatches++;
      }

      onProgress?.({
        totalSamples: samples.length,
        discardSamples: discardSamples.length,
        drawSamples: drawSamples.length,
        epoch: epoch + 1,
        discardLoss: discardBatches > 0 ? epochDiscardLoss / discardBatches : 0,
        drawLoss: drawBatches > 0 ? epochDrawLoss / drawBatches : 0,
      });
    }

    optimizer.dispose();
  }

  /**
   * Train on a batch of discard-phase samples.
   * Uses the factored action space: combo score = sum of card logits.
   * Loss: cross-entropy over valid discard combinations + yaniv.
   */
  private trainDiscardBatch(batch: TrainingSample[], optimizer: tf.Optimizer): number {
    let batchLoss = 0;

    const lossFunc = () => {
      return tf.tidy(() => {
        const states = tf.tensor2d(batch.map(s => Array.from(s.state)));
        const [cardLogits, yanivLogit] = this.model!.predict(states) as tf.Tensor[];

        let totalLoss = tf.scalar(0);

        for (let b = 0; b < batch.length; b++) {
          const sample = batch[b];
          const sampleCardLogits = cardLogits.slice([b, 0], [1, 52]).squeeze(); // [52]
          const sampleYanivLogit = yanivLogit.slice([b, 0], [1, 1]).squeeze(); // [1]

          // Compute score for each valid combo
          const actionScores: tf.Tensor[] = [];
          let targetIdx = -1;

          // Score each valid discard combo
          for (let c = 0; c < sample.validComboMasks.length; c++) {
            const mask = tf.tensor1d(sample.validComboMasks[c].map(v => v ? 1 : 0));
            const score = sampleCardLogits.mul(mask).sum();
            actionScores.push(score);

            // Check if this is the chosen action
            if (!sample.isYaniv) {
              const chosenMask = sample.discardCardsMask;
              const isMatch = sample.validComboMasks[c].every(
                (v, i) => v === chosenMask[i],
              );
              if (isMatch) targetIdx = c;
            }
            mask.dispose();
          }

          // Add Yaniv as an action if can call
          if (sample.canCallYaniv) {
            actionScores.push(sampleYanivLogit.reshape([]));
            if (sample.isYaniv) targetIdx = actionScores.length - 1;
          }

          if (targetIdx === -1 || actionScores.length === 0) continue;

          // Log-softmax over action scores
          const allScores = tf.stack(actionScores);
          const logSoftmax = tf.logSoftmax(allScores);
          const targetLogProb = logSoftmax.gather(targetIdx);

          // Weighted negative log likelihood
          const loss = targetLogProb.neg().mul(sample.weight);
          totalLoss = totalLoss.add(loss);

          // Clean up
          for (const s of actionScores) s.dispose();
          allScores.dispose();
          logSoftmax.dispose();
          targetLogProb.dispose();
        }

        const avgLoss = totalLoss.div(batch.length);
        batchLoss = avgLoss.dataSync()[0];
        return avgLoss as tf.Scalar;
      });
    };

    optimizer.minimize(lossFunc, true);
    return batchLoss;
  }

  /**
   * Train on a batch of draw-phase samples.
   * Simple binary cross-entropy on draw source.
   */
  private trainDrawBatch(batch: TrainingSample[], optimizer: tf.Optimizer): number {
    let batchLoss = 0;

    const lossFunc = () => {
      return tf.tidy(() => {
        const states = tf.tensor2d(batch.map(s => Array.from(s.state)));
        const outputs = this.model!.predict(states) as tf.Tensor[];
        const drawLogits = outputs[2]; // draw_logit head

        const targets = tf.tensor2d(
          batch.map(s => [s.drewFromDiscard ? 1 : 0]),
        );
        const weights = tf.tensor2d(
          batch.map(s => [s.weight]),
        );

        const bce = tf.losses.sigmoidCrossEntropy(targets, drawLogits);
        const weightedLoss = bce.mul(weights.mean());

        batchLoss = weightedLoss.dataSync()[0];
        return weightedLoss as tf.Scalar;
      });
    };

    optimizer.minimize(lossFunc, true);
    return batchLoss;
  }

  private shuffleArray<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  /**
   * Get the trained model's weights as a serializable object.
   */
  async getModelWeights(): Promise<ArrayBuffer[]> {
    if (!this.model) throw new Error('Model not built');
    const weightData = this.model.getWeights();
    const buffers: ArrayBuffer[] = [];
    for (const w of weightData) {
      buffers.push(await w.data().then((d: Float32Array | Int32Array | Uint8Array) => d.buffer));
    }
    return buffers;
  }

  /**
   * Save model to a path (for Node.js) or indexeddb/localstorage (for browser).
   */
  async saveModel(path: string): Promise<void> {
    if (!this.model) throw new Error('Model not built');
    await this.model.save(path);
  }

  /**
   * Load model from a path.
   */
  async loadModel(path: string): Promise<void> {
    this.model = await tf.loadLayersModel(path);
  }
}
