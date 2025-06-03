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
exports.TrainingSystemV2 = void 0;
const tf = __importStar(require("@tensorflow/tfjs"));
const Game_1 = require("./Game");
const NeuralNetworkAI_1 = require("./NeuralNetworkAI");
const Card_1 = require("./Card");
class TrainingSystemV2 {
    constructor() {
        this.experiences = [];
        this.maxExperiences = 10000;
        this.batchSize = 32;
        this.gamma = 0.95;
        this.neuralAI = new NeuralNetworkAI_1.NeuralNetworkAI();
    }
    async runSelfPlayEpisode() {
        const players = ['AI1', 'AI2', 'AI3', 'AI4'];
        const game = new Game_1.YanivGame(players);
        const episodeExperiences = [];
        while (game.getState().gamePhase === 'playing') {
            const state = game.getState();
            const currentPlayer = state.players[state.currentPlayerIndex];
            const playerId = currentPlayer.id;
            // Encode current state
            const stateTensor = this.neuralAI.encodeGameState(currentPlayer.hand, state.discardPile, state, playerId);
            let actionIndex = 0;
            let reward = 0;
            try {
                // Check if can call Yaniv
                if (game.canCallYaniv(playerId)) {
                    const handValue = Card_1.CardUtils.getHandValue(currentPlayer.hand);
                    // Use neural network to decide whether to call Yaniv
                    const decision = await this.neuralAI.makeDecision(currentPlayer.hand, state.discardPile, state, playerId);
                    if (decision.action === 'yaniv') {
                        game.callYaniv(playerId);
                        actionIndex = 4;
                        reward = 100 - handValue; // Better reward for lower hand values
                        // Store experience and break as round ended
                        episodeExperiences.push({
                            state: stateTensor,
                            action: actionIndex,
                            reward: reward,
                            nextState: null,
                            done: true
                        });
                        break;
                    }
                }
                // Handle turn phases
                if (state.turnPhase === 'discard') {
                    // Select cards to discard
                    const validDiscards = this.getAllValidDiscards(currentPlayer.hand);
                    if (validDiscards.length > 0) {
                        // For training, randomly select a valid discard
                        const discardIndex = Math.floor(Math.random() * validDiscards.length);
                        const cardsToDiscard = validDiscards[discardIndex];
                        const discardValue = Card_1.CardUtils.getHandValue(cardsToDiscard);
                        const success = game.discard(playerId, cardsToDiscard);
                        if (success) {
                            actionIndex = 5 + discardIndex; // Map to action space
                            reward = discardValue * 0.3; // Reward for discarding high value cards
                        }
                        else {
                            reward = -20;
                        }
                    }
                }
                else if (state.turnPhase === 'draw') {
                    // Draw phase - decide between deck and discard pile
                    const topDiscards = state.discardPile.slice(-3).reverse();
                    // For training, use a mix of random and strategic decisions
                    if (Math.random() < 0.7 && topDiscards.length > 0) {
                        // Check if discard pile has useful cards
                        const drawCount = Math.min(Math.floor(Math.random() * topDiscards.length) + 1, 3);
                        game.drawFromDiscard(playerId);
                        actionIndex = drawCount; // 1-3 for discard draws
                        // Calculate reward based on hand improvement potential
                        const drawnCards = topDiscards.slice(0, drawCount);
                        const drawnValue = Card_1.CardUtils.getHandValue(drawnCards);
                        reward = -drawnValue * 0.1; // Small penalty for taking cards
                    }
                    else {
                        // Draw from deck
                        game.drawFromDeck(playerId);
                        actionIndex = 0;
                        reward = -0.5; // Small penalty for unknown card
                    }
                }
                // Hand value change calculated in reward
            }
            catch (error) {
                // Invalid action
                reward = -20;
            }
            // Get next state
            const newState = game.getState();
            let nextStateTensor = null;
            if (newState.gamePhase === 'playing') {
                const nextPlayer = newState.players[newState.currentPlayerIndex];
                if (nextPlayer.id === playerId) {
                    // Still the same player's turn (draw phase)
                    nextStateTensor = this.neuralAI.encodeGameState(nextPlayer.hand, newState.discardPile, newState, nextPlayer.id);
                }
            }
            // Store experience
            episodeExperiences.push({
                state: stateTensor,
                action: actionIndex,
                reward: reward,
                nextState: nextStateTensor,
                done: newState.gamePhase !== 'playing'
            });
        }
        // Calculate returns and add experiences
        this.calculateReturns(episodeExperiences);
        this.addExperiences(episodeExperiences);
    }
    getAllValidDiscards(hand) {
        const validDiscards = [];
        // Single cards
        hand.forEach(card => validDiscards.push([card]));
        // Sets
        const cardsByRank = new Map();
        hand.forEach(card => {
            const cards = cardsByRank.get(card.rank) || [];
            cards.push(card);
            cardsByRank.set(card.rank, cards);
        });
        cardsByRank.forEach(cards => {
            if (cards.length >= 2) {
                for (let size = 2; size <= cards.length; size++) {
                    validDiscards.push(cards.slice(0, size));
                }
            }
        });
        // Runs
        const cardsBySuit = new Map();
        hand.forEach(card => {
            const cards = cardsBySuit.get(card.suit) || [];
            cards.push(card);
            cardsBySuit.set(card.suit, cards);
        });
        cardsBySuit.forEach(cards => {
            const sorted = cards.sort((a, b) => a.value - b.value);
            for (let start = 0; start < sorted.length - 2; start++) {
                for (let end = start + 2; end < sorted.length; end++) {
                    const run = sorted.slice(start, end + 1);
                    if (Card_1.CardUtils.isRun(run)) {
                        validDiscards.push(run);
                    }
                }
            }
        });
        return validDiscards;
    }
    calculateReturns(experiences) {
        let runningReturn = 0;
        for (let i = experiences.length - 1; i >= 0; i--) {
            if (experiences[i].done) {
                runningReturn = experiences[i].reward;
            }
            else {
                runningReturn = experiences[i].reward + this.gamma * runningReturn;
            }
            experiences[i].reward = runningReturn; // Update with discounted return
        }
    }
    addExperiences(newExperiences) {
        this.experiences.push(...newExperiences);
        if (this.experiences.length > this.maxExperiences) {
            const toRemove = this.experiences.slice(0, this.experiences.length - this.maxExperiences);
            toRemove.forEach(exp => {
                exp.state.dispose();
                if (exp.nextState)
                    exp.nextState.dispose();
            });
            this.experiences = this.experiences.slice(-this.maxExperiences);
        }
    }
    async train(epochs = 10) {
        if (this.experiences.length < this.batchSize) {
            console.log('Not enough experiences to train');
            return 0;
        }
        let totalLoss = 0;
        for (let epoch = 0; epoch < epochs; epoch++) {
            const batch = this.sampleBatch();
            const states = tf.concat(batch.map(exp => exp.state));
            const actionTargets = tf.tensor2d(batch.map(exp => {
                const target = new Array(55).fill(0);
                target[exp.action] = 1;
                return target;
            }));
            const valueTargets = tf.tensor1d(batch.map(exp => exp.reward / 100)); // Normalize
            const history = await this.neuralAI['model'].fit(states, {
                action_output: actionTargets,
                value_output: valueTargets
            }, {
                batchSize: this.batchSize,
                epochs: 1,
                verbose: 0
            });
            totalLoss += history.history.loss[0];
            states.dispose();
            actionTargets.dispose();
            valueTargets.dispose();
        }
        return totalLoss / epochs;
    }
    sampleBatch() {
        const batch = [];
        const indices = new Set();
        while (batch.length < this.batchSize && batch.length < this.experiences.length) {
            const index = Math.floor(Math.random() * this.experiences.length);
            if (!indices.has(index)) {
                indices.add(index);
                batch.push(this.experiences[index]);
            }
        }
        return batch;
    }
    async runTrainingLoop(episodes = 100, trainEvery = 10) {
        console.log(`Starting training with ${episodes} episodes...`);
        for (let i = 0; i < episodes; i++) {
            await this.runSelfPlayEpisode();
            if ((i + 1) % trainEvery === 0 && this.experiences.length >= this.batchSize) {
                const avgLoss = await this.train();
                console.log(`Episode ${i + 1}: Experiences=${this.experiences.length}, Avg Loss=${avgLoss.toFixed(4)}`);
            }
            if ((i + 1) % 50 === 0) {
                await this.neuralAI.saveModel('localstorage://yaniv-ai-model');
                console.log('Model checkpoint saved');
            }
        }
        // Final save
        await this.neuralAI.saveModel('localstorage://yaniv-ai-model');
        console.log('Training complete! Final model saved.');
    }
    getNeuralAI() {
        return this.neuralAI;
    }
}
exports.TrainingSystemV2 = TrainingSystemV2;
