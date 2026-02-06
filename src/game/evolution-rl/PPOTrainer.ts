/**
 * Phase 3: PPO (Proximal Policy Optimization) Fine-tuning
 *
 * Starting from the distilled neural network (warm start), runs self-play
 * with PPO to push beyond the genetic ceiling.
 *
 * Key features:
 * - Self-play: NN plays against a frozen copy of itself
 * - GAE (Generalized Advantage Estimation) for advantage computation
 * - Clipped surrogate objective
 * - Entropy bonus (starts high, anneals) to prevent style collapse
 * - Periodic opponent refresh (uses latest policy weights)
 */

import * as tf from '@tensorflow/tfjs';
import { Card } from '../../types/game';
import {
  GameSnapshot, encodeState,
  enumerateValidDiscards, cardToIndex,
} from './StateEncoder';
import { GameSimulator, SimPolicy, TrajectoryStep } from './GameSimulator';
import { DistillationTrainer } from './DistillationTrainer';

export interface PPOConfig {
  selfPlayGamesPerBatch: number;  // Games per training batch (default: 50)
  totalBatches: number;           // Total training batches (default: 40)
  ppoEpochs: number;             // PPO epochs per batch (default: 4)
  clipEpsilon: number;            // PPO clip range (default: 0.2)
  learningRate: number;           // Adam learning rate (default: 0.0003)
  gamma: number;                  // Discount factor (default: 0.99)
  lambda: number;                 // GAE lambda (default: 0.95)
  entropyCoeffStart: number;      // Initial entropy bonus (default: 0.05)
  entropyCoeffEnd: number;        // Final entropy bonus (default: 0.01)
  valueCoeff: number;             // Value loss coefficient (default: 0.5)
  maxGradNorm: number;            // Gradient clipping (default: 0.5)
  opponentRefreshInterval: number; // Batches between opponent refresh (default: 5)
  miniBatchSize: number;          // Mini-batch size for PPO updates (default: 32)
}

export const DEFAULT_PPO_CONFIG: PPOConfig = {
  selfPlayGamesPerBatch: 50,
  totalBatches: 40,
  ppoEpochs: 4,
  clipEpsilon: 0.2,
  learningRate: 0.0003,
  gamma: 0.99,
  lambda: 0.95,
  entropyCoeffStart: 0.05,
  entropyCoeffEnd: 0.01,
  valueCoeff: 0.5,
  maxGradNorm: 0.5,
  opponentRefreshInterval: 5,
  miniBatchSize: 32,
};

/** A trajectory step with PPO-specific data */
interface PPOStep {
  state: Float32Array;
  snapshot: GameSnapshot;
  // Discard phase data
  isDiscardPhase: boolean;
  chosenComboIdx: number;        // Index in validCombos of chosen action
  validComboMasks: boolean[][];  // All valid combo masks
  canCallYaniv: boolean;
  isYaniv: boolean;
  cardsMask: boolean[];          // 52-dim mask of chosen discard
  // Draw phase data
  drewFromDiscard: boolean;
  // PPO data
  oldLogProb: number;
  value: number;
  reward: number;
  advantage: number;
  returnValue: number;
}

export interface PPOStats {
  batch: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  avgReward: number;
  winRate: number;
}

export type PPOProgressCallback = (stats: PPOStats) => void;

/**
 * An adapter that wraps the NN model to act as a SimPolicy for self-play.
 * This is the bridge between the neural network and the game simulator.
 */
export class NNSimPolicy implements SimPolicy {
  constructor(
    private model: tf.LayersModel,
    private temperature: number = 1.0,
  ) {}

  decideDiscard(snapshot: GameSnapshot): { action: 'yaniv' | 'discard'; cards?: Card[] } {
    const result = tf.tidy(() => {
      const state = tf.tensor2d([Array.from(encodeState(snapshot))]);
      const outputs = this.model.predict(state) as tf.Tensor[];
      const cardLogits = outputs[0].dataSync(); // [52]
      const yanivLogit = outputs[1].dataSync(); // [1]

      const validCombos = enumerateValidDiscards(snapshot.hand);
      if (validCombos.length === 0) return { action: 'discard' as const, cards: [snapshot.hand[0]] };

      // Score each valid combo
      const scores: number[] = [];
      for (const combo of validCombos) {
        let score = 0;
        for (const card of combo) {
          score += cardLogits[cardToIndex(card)];
        }
        scores.push(score / this.temperature);
      }

      // Add Yaniv option
      if (snapshot.canCallYaniv) {
        scores.push(yanivLogit[0] / this.temperature);
      }

      // Sample from softmax
      const maxScore = Math.max(...scores);
      const expScores = scores.map(s => Math.exp(s - maxScore));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map(e => e / sumExp);

      const actionIdx = this.sampleCategorical(probs);

      if (snapshot.canCallYaniv && actionIdx === validCombos.length) {
        return { action: 'yaniv' as const };
      }

      const chosenCombo = validCombos[Math.min(actionIdx, validCombos.length - 1)];
      return { action: 'discard' as const, cards: chosenCombo };
    });

    return result;
  }

  decideDraw(snapshot: GameSnapshot): 'deck' | 'discard' {
    const drawFromDiscard = tf.tidy(() => {
      const state = tf.tensor2d([Array.from(encodeState(snapshot))]);
      const outputs = this.model.predict(state) as tf.Tensor[];
      const drawLogit = outputs[2].dataSync(); // [1]
      const prob = 1 / (1 + Math.exp(-drawLogit[0])); // sigmoid
      return Math.random() < prob;
    });

    return drawFromDiscard ? 'discard' : 'deck';
  }

  private sampleCategorical(probs: number[]): number {
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r < cum) return i;
    }
    return probs.length - 1;
  }
}

export class PPOTrainer {
  private config: PPOConfig;
  private model: tf.LayersModel;
  private optimizer: tf.Optimizer;

  constructor(model: tf.LayersModel, config?: Partial<PPOConfig>) {
    this.config = { ...DEFAULT_PPO_CONFIG, ...config };
    this.model = model;
    this.optimizer = tf.train.adam(this.config.learningRate);
  }

  /**
   * Run PPO fine-tuning with self-play.
   */
  async train(onProgress?: PPOProgressCallback): Promise<void> {
    // Create a frozen copy of the model as opponent
    let opponentModel = await this.cloneModel(this.model);

    for (let batch = 0; batch < this.config.totalBatches; batch++) {
      // Entropy coefficient annealing
      const progress = batch / this.config.totalBatches;
      const entropyCoeff = this.config.entropyCoeffStart +
        (this.config.entropyCoeffEnd - this.config.entropyCoeffStart) * progress;

      // Self-play: collect trajectories
      const trajectoryData = this.collectSelfPlayData(opponentModel);

      // Compute advantages using GAE
      this.computeGAE(trajectoryData);

      // PPO update
      let epochPolicyLoss = 0;
      let epochValueLoss = 0;
      let epochEntropy = 0;

      for (let epoch = 0; epoch < this.config.ppoEpochs; epoch++) {
        const { policyLoss, valueLoss, entropy } = this.ppoUpdate(
          trajectoryData, entropyCoeff,
        );
        epochPolicyLoss += policyLoss;
        epochValueLoss += valueLoss;
        epochEntropy += entropy;
      }

      // Report stats
      const avgReward = trajectoryData.reduce((s, t) => s + t.reward, 0) / trajectoryData.length;
      const wins = trajectoryData.filter(t => t.reward > 0.5).length;

      onProgress?.({
        batch: batch + 1,
        policyLoss: epochPolicyLoss / this.config.ppoEpochs,
        valueLoss: epochValueLoss / this.config.ppoEpochs,
        entropy: epochEntropy / this.config.ppoEpochs,
        avgReward,
        winRate: wins / Math.max(trajectoryData.length, 1),
      });

      // Refresh opponent periodically
      if ((batch + 1) % this.config.opponentRefreshInterval === 0) {
        opponentModel.dispose();
        opponentModel = await this.cloneModel(this.model);
      }
    }

    opponentModel.dispose();
  }

  /**
   * Collect trajectory data from self-play games.
   */
  private collectSelfPlayData(opponentModel: tf.LayersModel): PPOStep[] {
    const allSteps: PPOStep[] = [];
    const policy = new NNSimPolicy(this.model);
    const opponent = new NNSimPolicy(opponentModel);

    for (let g = 0; g < this.config.selfPlayGamesPerBatch; g++) {
      const result = GameSimulator.playGame(policy, opponent, true);

      // Convert trajectory0 (our policy's perspective) to PPO steps
      for (const step of result.trajectory0) {
        const ppoStep = this.trajectoryToPPOStep(step);
        if (ppoStep) allSteps.push(ppoStep);
      }
    }

    return allSteps;
  }

  /**
   * Convert a raw trajectory step to a PPO step with log-prob and value.
   */
  private trajectoryToPPOStep(step: TrajectoryStep): PPOStep | null {
    const { state, snapshot, action, reward } = step;

    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([Array.from(state)]);
      const outputs = this.model.predict(stateTensor) as tf.Tensor[];
      const cardLogits = outputs[0].dataSync(); // [52]
      const yanivLogit = outputs[1].dataSync(); // [1]
      const drawLogit = outputs[2].dataSync(); // [1]
      const value = outputs[3].dataSync(); // [1]

      if (action.phase === 'discard') {
        const validCombos = enumerateValidDiscards(snapshot.hand);
        if (validCombos.length === 0) return null;

        // Compute scores for all valid actions
        const scores: number[] = [];
        for (const combo of validCombos) {
          let score = 0;
          for (const card of combo) score += cardLogits[cardToIndex(card)];
          scores.push(score);
        }
        if (snapshot.canCallYaniv) scores.push(yanivLogit[0]);

        // Find which action was taken
        let chosenIdx = -1;
        if (action.type === 'yaniv' && snapshot.canCallYaniv) {
          chosenIdx = validCombos.length; // Yaniv is the last action
        } else if (action.cardsMask) {
          for (let c = 0; c < validCombos.length; c++) {
            const comboMask = new Array(52).fill(false);
            for (const card of validCombos[c]) comboMask[cardToIndex(card)] = true;
            if (comboMask.every((v, i) => v === action.cardsMask![i])) {
              chosenIdx = c;
              break;
            }
          }
        }
        if (chosenIdx === -1) chosenIdx = 0; // Fallback

        // Compute log probability
        const maxScore = Math.max(...scores);
        const expScores = scores.map(s => Math.exp(s - maxScore));
        const logSumExp = Math.log(expScores.reduce((a, b) => a + b, 0)) + maxScore;
        const logProb = scores[chosenIdx] - logSumExp;

        const validComboMasks = validCombos.map(combo => {
          const mask = new Array(52).fill(false);
          for (const card of combo) mask[cardToIndex(card)] = true;
          return mask;
        });

        return {
          state,
          snapshot,
          isDiscardPhase: true,
          chosenComboIdx: chosenIdx,
          validComboMasks,
          canCallYaniv: snapshot.canCallYaniv,
          isYaniv: action.type === 'yaniv',
          cardsMask: action.cardsMask || new Array(52).fill(false),
          drewFromDiscard: false,
          oldLogProb: logProb,
          value: value[0],
          reward,
          advantage: 0,
          returnValue: 0,
        } as PPOStep;
      } else {
        // Draw phase
        const prob = 1 / (1 + Math.exp(-drawLogit[0]));
        const drewFromDiscard = action.type === 'draw_discard';
        const logProb = drewFromDiscard ? Math.log(prob + 1e-8) : Math.log(1 - prob + 1e-8);

        return {
          state,
          snapshot,
          isDiscardPhase: false,
          chosenComboIdx: -1,
          validComboMasks: [],
          canCallYaniv: false,
          isYaniv: false,
          cardsMask: new Array(52).fill(false),
          drewFromDiscard,
          oldLogProb: logProb,
          value: value[0],
          reward,
          advantage: 0,
          returnValue: 0,
        } as PPOStep;
      }
    });
  }

  /**
   * Compute GAE (Generalized Advantage Estimation) for a sequence of steps.
   */
  private computeGAE(steps: PPOStep[]): void {
    const { gamma, lambda } = this.config;

    // Process in reverse
    let lastAdvantage = 0;

    for (let i = steps.length - 1; i >= 0; i--) {
      const nextValue = i < steps.length - 1 ? steps[i + 1].value : 0;
      const delta = steps[i].reward + gamma * nextValue - steps[i].value;
      steps[i].advantage = delta + gamma * lambda * lastAdvantage;
      steps[i].returnValue = steps[i].advantage + steps[i].value;
      lastAdvantage = steps[i].advantage;
    }

    // Normalize advantages
    const mean = steps.reduce((s, t) => s + t.advantage, 0) / steps.length;
    const std = Math.sqrt(
      steps.reduce((s, t) => s + (t.advantage - mean) ** 2, 0) / steps.length,
    ) + 1e-8;
    for (const step of steps) {
      step.advantage = (step.advantage - mean) / std;
    }
  }

  /**
   * Perform one PPO update over all collected steps.
   */
  private ppoUpdate(
    steps: PPOStep[],
    entropyCoeff: number,
  ): { policyLoss: number; valueLoss: number; entropy: number } {
    // Shuffle steps
    for (let i = steps.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [steps[i], steps[j]] = [steps[j], steps[i]];
    }

    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    let totalEntropy = 0;
    let batchCount = 0;

    // Process in mini-batches
    for (let start = 0; start < steps.length; start += this.config.miniBatchSize) {
      const miniBatch = steps.slice(start, start + this.config.miniBatchSize);
      if (miniBatch.length === 0) continue;

      const { policyLoss, valueLoss, entropy } = this.updateMiniBatch(
        miniBatch, entropyCoeff,
      );

      totalPolicyLoss += policyLoss;
      totalValueLoss += valueLoss;
      totalEntropy += entropy;
      batchCount++;
    }

    return {
      policyLoss: batchCount > 0 ? totalPolicyLoss / batchCount : 0,
      valueLoss: batchCount > 0 ? totalValueLoss / batchCount : 0,
      entropy: batchCount > 0 ? totalEntropy / batchCount : 0,
    };
  }

  /**
   * PPO update for a mini-batch.
   * Processes each step individually due to variable action spaces,
   * but accumulates gradients.
   */
  private updateMiniBatch(
    miniBatch: PPOStep[],
    entropyCoeff: number,
  ): { policyLoss: number; valueLoss: number; entropy: number } {
    let batchPolicyLoss = 0;
    let batchValueLoss = 0;
    let batchEntropy = 0;

    this.optimizer.minimize(() => {
      return tf.tidy(() => {
        let totalLoss = tf.scalar(0);
        let policyLossSum = 0;
        let valueLossSum = 0;
        let entropySum = 0;
        let count = 0;

        // Batch forward pass
        const states = tf.tensor2d(miniBatch.map(s => Array.from(s.state)));
        const outputs = this.model.apply(states, { training: true }) as tf.Tensor[];
        const allCardLogits = outputs[0]; // [N, 52]
        const allYanivLogits = outputs[1]; // [N, 1]
        const allDrawLogits = outputs[2]; // [N, 1]
        const allValues = outputs[3]; // [N, 1]

        for (let b = 0; b < miniBatch.length; b++) {
          const step = miniBatch[b];
          const cardLogits = allCardLogits.slice([b, 0], [1, 52]).squeeze();
          const yanivLogit = allYanivLogits.slice([b, 0], [1, 1]).squeeze();
          const drawLogit = allDrawLogits.slice([b, 0], [1, 1]).squeeze();
          const valueEstimate = allValues.slice([b, 0], [1, 1]).squeeze();

          if (step.isDiscardPhase) {
            // Compute new log prob for the chosen action
            const scores: tf.Tensor[] = [];

            for (const comboMask of step.validComboMasks) {
              const mask = tf.tensor1d(comboMask.map(v => v ? 1 : 0));
              scores.push(cardLogits.mul(mask).sum());
              mask.dispose();
            }
            if (step.canCallYaniv) {
              scores.push(yanivLogit.reshape([]));
            }

            if (scores.length === 0) continue;

            const allScores = tf.stack(scores);
            const logSoftmax = tf.logSoftmax(allScores);
            const probs = tf.softmax(allScores);
            const chosenIdx = Math.min(step.chosenComboIdx, scores.length - 1);
            const newLogProb = logSoftmax.gather(chosenIdx);

            // Entropy of the distribution
            const stepEntropy = probs.mul(logSoftmax).sum().neg();

            // PPO clipped surrogate
            const ratio = tf.exp(newLogProb.sub(step.oldLogProb));
            const advTensor = tf.scalar(step.advantage);
            const surr1 = ratio.mul(advTensor);
            const surr2 = tf.clipByValue(
              ratio, 1 - this.config.clipEpsilon, 1 + this.config.clipEpsilon,
            ).mul(advTensor);
            const policyLoss = tf.minimum(surr1, surr2).neg();

            // Value loss
            const returnTensor = tf.scalar(step.returnValue);
            const valueLoss = valueEstimate.sub(returnTensor).square().mul(this.config.valueCoeff);

            // Combined loss
            const stepLoss = policyLoss.add(valueLoss).sub(stepEntropy.mul(entropyCoeff));
            totalLoss = totalLoss.add(stepLoss);

            policyLossSum += policyLoss.dataSync()[0];
            valueLossSum += valueLoss.dataSync()[0];
            entropySum += stepEntropy.dataSync()[0];
            count++;

            // Cleanup
            for (const s of scores) s.dispose();
            allScores.dispose();
            logSoftmax.dispose();
            probs.dispose();
          } else {
            // Draw phase: binary cross-entropy with PPO
            const prob = tf.sigmoid(drawLogit);
            const drewDiscard = step.drewFromDiscard;
            const newLogProb = drewDiscard
              ? tf.log(prob.add(1e-8))
              : tf.log(tf.scalar(1).sub(prob).add(1e-8));

            const drawEntropy = prob.mul(tf.log(prob.add(1e-8))).neg()
              .add(tf.scalar(1).sub(prob).mul(tf.log(tf.scalar(1).sub(prob).add(1e-8))).neg());

            const ratio = tf.exp(newLogProb.sub(step.oldLogProb));
            const advTensor = tf.scalar(step.advantage);
            const surr1 = ratio.mul(advTensor);
            const surr2 = tf.clipByValue(
              ratio, 1 - this.config.clipEpsilon, 1 + this.config.clipEpsilon,
            ).mul(advTensor);
            const policyLoss = tf.minimum(surr1, surr2).neg();

            const returnTensor = tf.scalar(step.returnValue);
            const valueLoss = valueEstimate.sub(returnTensor).square().mul(this.config.valueCoeff);

            const stepLoss = policyLoss.add(valueLoss).sub(drawEntropy.mul(entropyCoeff));
            totalLoss = totalLoss.add(stepLoss);

            policyLossSum += policyLoss.dataSync()[0];
            valueLossSum += valueLoss.dataSync()[0];
            entropySum += drawEntropy.dataSync()[0];
            count++;
          }
        }

        const avgLoss = count > 0 ? totalLoss.div(count) : totalLoss;
        batchPolicyLoss = count > 0 ? policyLossSum / count : 0;
        batchValueLoss = count > 0 ? valueLossSum / count : 0;
        batchEntropy = count > 0 ? entropySum / count : 0;

        return avgLoss as tf.Scalar;
      });
    }, true);

    return {
      policyLoss: batchPolicyLoss,
      valueLoss: batchValueLoss,
      entropy: batchEntropy,
    };
  }

  /**
   * Clone a tf.LayersModel by saving/loading weights.
   */
  private async cloneModel(model: tf.LayersModel): Promise<tf.LayersModel> {
    // Build a new model with the same architecture
    const trainer = new DistillationTrainer({
      hiddenSize1: 128,
      hiddenSize2: 64,
    });
    const clone = trainer.buildModel();

    // Copy weights
    const weights = model.getWeights();
    clone.setWeights(weights);

    return clone;
  }
}
