"""
Yaniv GA Training System

A genetic algorithm training framework for evolving neural network policies
that play the Yaniv card game. The environment matches the TypeScript game rules exactly.
"""

from .env import YanivEnv, Card, Action
from .encoding import StateEncoder, ActionEncoder
from .bots import RandomBot, GreedyBot, RuleBasedBot
from .model import ActionValueNet, TorchPolicy
from .genome import Genome
from .ga import GeneticAlgorithm

__version__ = "1.0.0"
__all__ = [
    "YanivEnv",
    "Card",
    "Action",
    "StateEncoder",
    "ActionEncoder",
    "RandomBot",
    "GreedyBot",
    "RuleBasedBot",
    "ActionValueNet",
    "TorchPolicy",
    "Genome",
    "GeneticAlgorithm",
]
