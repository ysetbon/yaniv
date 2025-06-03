import numpy as np
from typing import List, Dict, Tuple, Optional
from yaniv_neural_network_optimized import YanivNeuralNetworkOptimized
import random
from numba import jit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import queue
import threading

# Keep all the existing YanivGameAIOptimized class and other functions from yaniv_game_ai_optimized.py
# Just updating the parallel execution part

def play_games_batch_with_progress(args):
    """Play multiple games for a batch of matchups and report progress"""
    matchups, games_per_matchup, progress_queue = args
    results = []
    
    for player1, player2 in matchups:
        p1_wins = 0
        p2_wins = 0
        
        for game_num in range(games_per_matchup):
            # Alternate who goes first
            if game_num % 2 == 0:
                game = YanivGameAIOptimized()
                winner = game.play_game([player1, player2])
            else:
                game = YanivGameAIOptimized()
                winner = game.play_game([player2, player1])
            
            if winner.network_id == player1.network_id:
                p1_wins += 1
            else:
                p2_wins += 1
        
        results.append((p1_wins, p2_wins))
        
        # Report progress after each matchup
        if progress_queue is not None:
            progress_queue.put(games_per_matchup)
    
    return results


class ParallelGameExecutorFixed:
    """Execute games in parallel with real-time progress updates"""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or mp.cpu_count()
    
    def play_tournament_parallel_with_progress(self, population: List[YanivNeuralNetworkOptimized], 
                                             games_per_matchup: int = 10) -> Dict[int, Tuple[int, int]]:
        """Play tournament in parallel with real-time progress bar"""
        # Create all matchups
        matchups = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                matchups.append((population[i], population[j]))
        
        total_matchups = len(matchups)
        total_games = total_matchups * games_per_matchup
        
        # Split matchups into chunks for parallel processing
        chunk_size = max(1, len(matchups) // (self.num_workers * 4))
        matchup_chunks = [matchups[i:i + chunk_size] for i in range(0, len(matchups), chunk_size)]
        
        # Initialize results dictionary
        results_dict = {i: (0, 0) for i in range(len(population))}
        
        # Use multiprocessing Manager for progress communication
        manager = mp.Manager()
        progress_queue = manager.Queue()
        
        # Progress bar update thread
        progress_bar = tqdm(total=total_games, desc="Games", unit="game")
        completed_games = 0
        
        def update_progress():
            nonlocal completed_games
            while True:
                try:
                    games = progress_queue.get(timeout=0.1)
                    if games == -1:  # Sentinel value to stop
                        break
                    completed_games += games
                    progress_bar.update(games)
                    progress_bar.set_postfix({
                        'matchups': f'{completed_games // games_per_matchup}/{total_matchups}',
                        'workers': self.num_workers
                    })
                except queue.Empty:
                    continue
        
        # Start progress update thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()
        
        try:
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for chunk in matchup_chunks:
                    future = executor.submit(play_games_batch_with_progress, 
                                           (chunk, games_per_matchup, progress_queue))
                    futures.append((chunk, future))
                
                # Collect results
                for chunk, future in futures:
                    chunk_results = future.result()
                    
                    for (p1, p2), (p1_wins, p2_wins) in zip(chunk, chunk_results):
                        # Update wins and games for both players
                        wins1, games1 = results_dict[p1.network_id]
                        results_dict[p1.network_id] = (wins1 + p1_wins, games1 + games_per_matchup)
                        
                        wins2, games2 = results_dict[p2.network_id]
                        results_dict[p2.network_id] = (wins2 + p2_wins, games2 + games_per_matchup)
        
        finally:
            # Stop progress thread
            progress_queue.put(-1)
            progress_thread.join()
            progress_bar.close()
        
        return results_dict