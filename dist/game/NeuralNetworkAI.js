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
exports.NeuralNetworkAI = void 0;
const tf = __importStar(require("@tensorflow/tfjs"));
const Card_1 = require("./Card");
class NeuralNetworkAI {
    constructor() {
        this.model = null;
        this.stateSize = 234; // Size of encoded state
        this.actionSize = 55; // Possible actions (draw from deck, draw 1-3 from discard, discard combinations, yaniv)
        this.buildModel();
    }
    buildModel() {
        const input = tf.input({ shape: [this.stateSize] });
        // Hidden layers with batch normalization and dropout for generalization
        let x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.dropout({ rate: 0.3 }).apply(x);
        x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.dropout({ rate: 0.2 }).apply(x);
        x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x);
        // Output layer - action probabilities
        const actionOutput = tf.layers.dense({
            units: this.actionSize,
            activation: 'softmax',
            name: 'action_output'
        }).apply(x);
        // Value head for state evaluation
        const valueOutput = tf.layers.dense({
            units: 1,
            activation: 'tanh',
            name: 'value_output'
        }).apply(x);
        this.model = tf.model({
            inputs: input,
            outputs: [actionOutput, valueOutput]
        });
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: {
                action_output: 'categoricalCrossentropy',
                value_output: 'meanSquaredError'
            }
        });
    }
    encodeGameState(hand, discardPile, gameState, playerId) {
        const encoded = new Float32Array(this.stateSize);
        let index = 0;
        // Encode hand (52 cards, one-hot encoding)
        for (const card of hand) {
            const cardIndex = this.cardToIndex(card);
            encoded[index + cardIndex] = 1;
        }
        index += 52;
        // Encode top 3 discard pile cards
        const topDiscards = discardPile.slice(-3).reverse();
        for (let i = 0; i < 3; i++) {
            if (i < topDiscards.length) {
                const cardIndex = this.cardToIndex(topDiscards[i]);
                encoded[index + cardIndex] = 1;
            }
            index += 52;
        }
        // Encode game features
        encoded[index++] = Card_1.CardUtils.getHandValue(hand) / 100; // Normalized hand value
        encoded[index++] = hand.length / 10; // Normalized hand size
        encoded[index++] = discardPile.length / 52; // Normalized discard pile size
        // Encode opponent information (assuming 2-4 players)
        if (gameState.players) {
            const opponents = gameState.players.filter(p => p.id !== playerId);
            for (let i = 0; i < 3; i++) {
                if (i < opponents.length) {
                    encoded[index++] = opponents[i].hand.length / 10;
                    encoded[index++] = opponents[i].score / 500; // Normalized score
                }
                else {
                    encoded[index++] = 0;
                    encoded[index++] = 0;
                }
            }
        }
        // Encode round number
        if (gameState.roundNumber) {
            encoded[index++] = gameState.roundNumber / 20; // Normalized
        }
        // Can call yaniv
        const handValue = Card_1.CardUtils.getHandValue(hand);
        encoded[index++] = handValue <= 7 ? 1 : 0;
        // Possible valid discards count
        const validDiscards = this.getAllValidDiscards(hand);
        encoded[index++] = validDiscards.length / 50; // Normalized
        return tf.tensor2d([Array.from(encoded)]);
    }
    cardToIndex(card) {
        const suitIndex = ['♠', '♥', '♦', '♣'].indexOf(card.suit);
        const rankIndex = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'].indexOf(card.rank);
        return suitIndex * 13 + rankIndex;
    }
    getAllValidDiscards(hand) {
        const validDiscards = [];
        // Single cards
        for (const card of hand) {
            validDiscards.push([card]);
        }
        // Sets (same rank)
        const cardsByRank = new Map();
        for (const card of hand) {
            const existing = cardsByRank.get(card.rank) || [];
            existing.push(card);
            cardsByRank.set(card.rank, existing);
        }
        for (const [_, cards] of cardsByRank) {
            if (cards.length >= 2) {
                for (let size = 2; size <= cards.length; size++) {
                    validDiscards.push(cards.slice(0, size));
                }
            }
        }
        // Runs (same suit, consecutive ranks)
        const cardsBySuit = new Map();
        for (const card of hand) {
            const existing = cardsBySuit.get(card.suit) || [];
            existing.push(card);
            cardsBySuit.set(card.suit, existing);
        }
        for (const [_, cards] of cardsBySuit) {
            const sorted = cards.sort((a, b) => a.value - b.value);
            for (let start = 0; start < sorted.length - 2; start++) {
                for (let end = start + 2; end < sorted.length; end++) {
                    const potentialRun = sorted.slice(start, end + 1);
                    if (Card_1.CardUtils.isRun(potentialRun)) {
                        validDiscards.push(potentialRun);
                    }
                }
            }
        }
        return validDiscards;
    }
    async makeDecision(hand, discardPile, gameState, playerId) {
        if (!this.model) {
            throw new Error('Model not initialized');
        }
        const stateTensor = this.encodeGameState(hand, discardPile, gameState, playerId);
        const [actionProbs, stateValue] = this.model.predict(stateTensor);
        // Get action probabilities
        const probs = await actionProbs.data();
        const validActions = this.getValidActionMask(hand, discardPile);
        // Apply mask to invalid actions
        const maskedProbs = Array.from(probs).map((p, i) => validActions[i] ? p : 0);
        // Sample action based on probabilities (exploration)
        const actionIndex = this.sampleAction(maskedProbs);
        // Cleanup tensors
        stateTensor.dispose();
        actionProbs.dispose();
        stateValue.dispose();
        return this.decodeAction(actionIndex, hand);
    }
    getValidActionMask(hand, discardPile) {
        const mask = new Array(this.actionSize).fill(false);
        // Action 0: Draw from deck (always valid)
        mask[0] = true;
        // Actions 1-3: Draw 1-3 cards from discard
        const topDiscards = discardPile.slice(-3).reverse();
        for (let i = 0; i < topDiscards.length; i++) {
            mask[i + 1] = true;
        }
        // Action 4: Call Yaniv
        const handValue = Card_1.CardUtils.getHandValue(hand);
        mask[4] = handValue <= 7;
        // Actions 5+: Valid discards
        const validDiscards = this.getAllValidDiscards(hand);
        for (let i = 0; i < Math.min(validDiscards.length, 50); i++) {
            mask[i + 5] = true;
        }
        return mask;
    }
    sampleAction(probabilities) {
        const sum = probabilities.reduce((a, b) => a + b, 0);
        if (sum === 0)
            return 0; // Default to draw from deck
        const normalized = probabilities.map(p => p / sum);
        const random = Math.random();
        let cumulative = 0;
        for (let i = 0; i < normalized.length; i++) {
            cumulative += normalized[i];
            if (random < cumulative) {
                return i;
            }
        }
        return 0;
    }
    decodeAction(actionIndex, hand) {
        if (actionIndex === 0) {
            return { action: 'draw', drawSource: 'deck' };
        }
        if (actionIndex >= 1 && actionIndex <= 3) {
            return {
                action: 'draw',
                drawSource: 'discard',
                drawCount: actionIndex
            };
        }
        if (actionIndex === 4) {
            return { action: 'yaniv' };
        }
        // Discard action
        const validDiscards = this.getAllValidDiscards(hand);
        const discardIndex = actionIndex - 5;
        if (discardIndex < validDiscards.length) {
            return {
                action: 'discard',
                cardsToDiscard: validDiscards[discardIndex]
            };
        }
        // Fallback
        return { action: 'draw', drawSource: 'deck' };
    }
    async saveModel(path) {
        if (this.model) {
            await this.model.save(path);
        }
    }
    async loadModel(path) {
        this.model = await tf.loadLayersModel(path);
    }
}
exports.NeuralNetworkAI = NeuralNetworkAI;
