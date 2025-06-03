import { Card, GameState } from '../types/game';
import { CardUtils } from './Card';
import { ComputerAI } from './ComputerAI';
import { NeuralNetworkAI } from './NeuralNetworkAI';
import { PythonTrainedAI } from './PythonTrainedAI';

export type AIMode = 'rule-based' | 'neural-network' | 'python-trained' | 'hybrid';

export class EnhancedComputerAI {
  private neuralAI: NeuralNetworkAI | null = null;
  private pythonAI: PythonTrainedAI | null = null;
  private mode: AIMode;
  private initPromise: Promise<void> | null = null;
  // private hybridThreshold = 0.7; // Confidence threshold for hybrid mode
  
  constructor(mode: AIMode = 'hybrid') {
    this.mode = mode;
    this.initPromise = this.initialize();
  }
  
  private async initialize() {
    if (this.mode === 'neural-network' || this.mode === 'hybrid') {
      await this.initializeNeuralNetwork();
    }
    if (this.mode === 'python-trained') {
      await this.initializePythonAI();
    }
  }
  
  async waitForInit(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise;
    }
  }
  
  isReady(): boolean {
    if (this.mode === 'python-trained') {
      return this.pythonAI !== null;
    }
    if (this.mode === 'neural-network' || this.mode === 'hybrid') {
      return this.neuralAI !== null;
    }
    return true; // rule-based is always ready
  }
  
  private async initializeNeuralNetwork() {
    this.neuralAI = new NeuralNetworkAI();
    
    // Try to load a pre-trained model from public directory
    try {
      await this.neuralAI.loadModel('/models/yaniv-trained-10000/model.json');
      console.log('Loaded pre-trained neural network model: yaniv-trained-10000');
    } catch (error) {
      console.log('No pre-trained model found, using untrained network');
    }
  }
  
  private async initializePythonAI() {
    try {
      this.pythonAI = new PythonTrainedAI();
      
      // Try to load your Python-trained model
      await this.pythonAI.loadModel('/saved_networks/best_overall_optimized.json');
      console.log('Loaded Python-trained AI model: best_overall_optimized');
    } catch (error) {
      console.error('Failed to initialize or load Python-trained model:', error);
      this.pythonAI = null;
    }
  }
  
  async makeDecision(
    hand: Card[], 
    discardPile: Card[],
    canCallYaniv: boolean,
    gameState?: Partial<GameState>,
    playerId?: string
  ): Promise<{
    action: 'yaniv' | 'draw' | 'discard';
    drawSource?: 'deck' | 'discard';
    drawCount?: number;
    cardsToDiscard?: Card[];
  }> {
    // Validate turn phase if gameState is provided
    const turnPhase = gameState?.turnPhase;
    if (turnPhase === 'draw') {
      // During draw phase, can only draw
      return {
        action: 'draw',
        drawSource: Math.random() < 0.3 && discardPile.length > 0 ? 'discard' : 'deck'
      };
    }
    
    if (this.mode === 'rule-based') {
      const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
      return ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
    }
    
    if (this.mode === 'neural-network' && this.neuralAI && gameState && playerId) {
      try {
        console.log('[Neural Network AI] Making decision...');
        const decision = await this.neuralAI.makeDecision(hand, discardPile, gameState, playerId);
        console.log('[Neural Network AI] Decision:', decision.action);
        return decision;
      } catch (error) {
        console.error('Neural network decision failed, falling back to rule-based', error);
        const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
        return ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
      }
    }
    
    if (this.mode === 'python-trained') {
      if (!this.pythonAI) {
        console.warn('Python AI not initialized, falling back to rule-based');
        const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
        return ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
      }
      
      if (!gameState || !playerId) {
        console.warn('Missing gameState or playerId for Python AI, falling back to rule-based');
        const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
        return ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
      }
      
      try {
        console.log('[Python Trained AI] Making decision...');
        const playerIndex = gameState.players?.findIndex(p => p.id === playerId) ?? 0;
        const decision = this.pythonAI.makeMove(gameState as GameState, playerIndex);
        console.log('[Python Trained AI] Decision:', decision.action);
        
        // Convert PythonTrainedAI format to EnhancedComputerAI format
        if (decision.action === 'yaniv') {
          return { action: 'yaniv' };
        } else if (decision.action === 'draw') {
          return { 
            action: 'draw', 
            drawSource: decision.source || 'deck' 
          };
        } else if (decision.action === 'discard' && decision.cards) {
          return { 
            action: 'discard', 
            cardsToDiscard: decision.cards 
          };
        }
        
        // Fallback if invalid decision
        throw new Error('Invalid Python AI decision');
      } catch (error) {
        console.error('Python AI decision failed, falling back to rule-based', error);
        const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
        return ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
      }
    }
    
    if (this.mode === 'hybrid' && this.neuralAI && gameState && playerId) {
      try {
        // Get decisions from both systems
        const neuralDecision = await this.neuralAI.makeDecision(hand, discardPile, gameState, playerId);
        const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
        const ruleDecision = ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
        
        // In hybrid mode, use neural network for strategic decisions
        // and rule-based for tactical/safety decisions
        const handValue = CardUtils.getHandValue(hand);
        
        // Always use rule-based for very low hand values (safety)
        if (handValue <= 5 && canCallYaniv) {
          return ruleDecision;
        }
        
        // Use neural network for complex mid-game decisions
        if (handValue > 10 && handValue < 25) {
          return neuralDecision;
        }
        
        // For edge cases, blend the decisions
        // Prefer neural network but validate with rules
        if (neuralDecision.action === 'yaniv' && !canCallYaniv) {
          return ruleDecision;
        }
        
        return neuralDecision;
        
      } catch (error) {
        console.error('Hybrid decision failed, falling back to rule-based', error);
        const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
        return ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
      }
    }
    
    // Default fallback
    const opponentHandSize = gameState?.players?.find(p => p.id !== playerId)?.hand.length;
    return ComputerAI.makeDecision(hand, discardPile, canCallYaniv, turnPhase || 'discard', opponentHandSize);
  }
  
  async selectDiscard(
    hand: Card[], 
    requiredValue: number
  ): Promise<Card[] | null> {
    // For discard selection, we primarily use the rule-based approach
    // as it's more reliable for exact value matching
    return ComputerAI.selectDiscard(hand, requiredValue);
  }
  
  async setMode(mode: AIMode) {
    this.mode = mode;
    this.neuralAI = null;
    this.pythonAI = null;
    this.initPromise = this.initialize();
    await this.initPromise;
  }
  
  getMode(): AIMode {
    return this.mode;
  }
}