"""
Battleship environment using OpenAI Gym toolkit with extensions for LLM-guided RL strategy distillation.

This package provides:
1. Basic battleship environment (Battleship-v0)
2. Adversarial battleship environment (AdversarialBattleship-v0)
3. Extensions for LLM-guided strategy distillation
"""

from gymnasium.envs.registration import register
from gym_battleship.environments.battleship import BattleshipEnv
from gym_battleship.environments.adversarial_battleship import AdversarialBattleshipEnv

# Try to import custom components for LLM-guided strategy distillation
try:
    from gym_battleship.lpml import LPMLAnnotator, LPMLParser
    from gym_battleship.distill import StrategyDistiller, KLDistiller
    __has_distill_components = True
except ImportError:
    __has_distill_components = False


register(
    id='Battleship-v0',
    entry_point='gym_battleship:BattleshipEnv',
)
register(
    id='AdversarialBattleship-v0',
    entry_point='gym_battleship:AdversarialBattleshipEnv',
)

# Extension for large grid variant
register(
    id='Battleship-LargeGrid-v0',
    entry_point='gym_battleship:BattleshipEnv',
    kwargs={'board_size': (12, 12), 'ship_sizes': {5: 1, 4: 1, 3: 2, 2: 1}}
)
