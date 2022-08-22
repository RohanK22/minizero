# =============================================================================
# import chess
# from encode_decode import decode_action
# =============================================================================

# Acts as a wrapper around move by storing the move index (encoded move)
class Action(object):
  def __init__(self, move_no: int):
    self.index = move_no

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index

# =============================================================================
#   def get_move(self, hidden_state: chess.Board) -> chess.Move:
#     return decode_action(hidden_state, self.index)
# =============================================================================
