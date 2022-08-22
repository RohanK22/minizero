import chess
from Game import Game

# Represents a particular state/node in the game tree
class Node(object):
  def __init__(self, prior: float, board: chess.Board):
    self.children = {} # action_no -> Node
    self.prior = prior # move prob from network
    self.visit_count = 0
    self.value_sum = 0
    self.to_play = Game.turn(board)
    self.state = board # chess.Board

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float: # Target value from MCTS
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count