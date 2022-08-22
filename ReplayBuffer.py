import config
import utils
import numpy as np
import random
import Game


class ReplayBuffer(object):

  def __init__(self):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

# =============================================================================
#   def sample_batch(self, num_unroll_steps: int):
#     compute_device = utils.get_compute_device()
#     games = [self.sample_game() for _ in range(self.batch_size)]
#     game_pos = [(g, self.sample_position(g)) for g in games]
#     targets = []
#     for (g, i) in game_pos:
#         moves_popped = []
#         board = g.board.copy()
#         n_to_pop = len(list(g.board.move_stack)) - 1 - i
#         for i in range(n_to_pop):
#             moves_popped.append(board.pop())
#         img = Game.encode_board(board).to(compute_device)
#         
#         # Iterate over num_unroll_steps or actions
#         for current_index in range(i, i + num_unroll_steps + 1):
#             if current_index < len(g.root_values):
#                 value = g.root_values[current_index]
#                 targets.append((img, g.child_visits[current_index], value))
#             else:
#                 targets.append((img, np.zeros(config.action_space_size), 0))
#         
#         while len(moves_popped) != 0:
#             board.push(moves_popped.pop())
#     return targets
# =============================================================================

  def sample_game(self) -> Game:
    return random.choice(self.buffer)

  def sample_position(self, game: Game) -> int:
    num_positions = len(game.root_values)
    mid = num_positions // 2
    return random.randint(mid, num_positions)
