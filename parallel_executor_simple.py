import numpy as np
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

from yaniv_neural_network_optimized import YanivNeuralNetworkOptimized
from yaniv_game_ai_optimized import YanivGameAIOptimized


def play_single_matchup(args):
    """Play all games between two players"""
    player1, player2, games_per_matchup = args
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
    
    return player1.network_id, player2.network_id, p1_wins, p2_wins


class SimpleParallelExecutor:
    """Simpler parallel executor that works well on Windows"""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or mp.cpu_count()
    
    def play_tournament_parallel_with_progress(self, population: List[YanivNeuralNetworkOptimized], 
                                             games_per_matchup: int = 10) -> Dict[int, Tuple[int, int]]:
        """Play tournament in parallel with progress bar"""
        # Create all matchup tasks
        tasks = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                tasks.append((population[i], population[j], games_per_matchup))
        
        total_matchups = len(tasks)
        total_games = total_matchups * games_per_matchup
        
        # Initialize results
        results_dict = {i: (0, 0) for i in range(len(population))}
        
        print(f"Running {total_matchups:,} matchups ({total_games:,} total games) on {self.num_workers} workers...")
        
        # Process all matchups with progress bar
        completed_matchups = 0
        completed_games = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(play_single_matchup, task): task for task in tasks}
            
            # Process completed tasks with progress bar
            with tqdm(total=total_games, desc="Games", unit="game", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                for future in as_completed(future_to_task):
                    p1_id, p2_id, p1_wins, p2_wins = future.result()
                    
                    # Update results
                    wins1, games1 = results_dict[p1_id]
                    results_dict[p1_id] = (wins1 + p1_wins, games1 + games_per_matchup)
                    
                    wins2, games2 = results_dict[p2_id]
                    results_dict[p2_id] = (wins2 + p2_wins, games2 + games_per_matchup)
                    
                    # Update progress
                    completed_matchups += 1
                    completed_games += games_per_matchup
                    pbar.update(games_per_matchup)
                    
                    # Update postfix with additional info
                    if completed_matchups % 10 == 0:  # Update every 10 matchups to reduce overhead
                        pbar.set_postfix({
                            'matchups': f'{completed_matchups}/{total_matchups}',
                            'win_rate': f'{results_dict[0][0]/max(1, results_dict[0][1]):.2%}' if results_dict[0][1] > 0 else '0%'
                        })
        
        return results_dict