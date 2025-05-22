# Import all the agent classes here to make them available
from .lats_agent import LATSAgent
from .mcts_agent import MCTSAgent
from .simple_search_agent import SimpleSearchAgent

__all__ = ['LATSAgent', 'MCTSAgent', 'SimpleSearchAgent', 'WebShopTreeSearchAgent']
