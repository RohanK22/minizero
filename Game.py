import chess
from typing import List
from utils import show_svg
import numpy as np
import config
import torch


# Wrapper around chess.Board class to play/undo moves, and keep track of history and stats
class Game(object):
  action_space_size = config.action_space_size 
  
  def __init__(self):
    self.board = chess.Board()
    self.root_values = []
    self.child_visits = []
    self.states = [] # Encoded board positions from the start
  
  def turn(board: chess.Board) -> int:
    return int(board.turn == chess.WHITE)
  
  def over(self) -> bool:
    return self.board.is_game_over()
  
  def valid_moves(self) -> List[chess.Move]:
    return list(self.board.legal_moves)

  def is_move_legal(board: chess.Board, move: chess.Move):
    return move in board.legal_moves

  def make_move(self, move: chess.Move) -> None:
    if not Game.is_move_legal(self.board, move):
        show_svg(self.board)
        print(move)
        raise Exception('Invalid move')
    self.states.append(encode_board(self.board))
    self.board.push(move)

  def undo_move(self) -> chess.Move:
    self.state.pop()
    return self.board.pop()
 
  def to_play(self) -> int:
    return Game.turn(self.board)
    
  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    child_visit = np.zeros(Game.action_space_size)
    for index in range(Game.action_space_size):
        if index in root.children:
            child_visit[index] = root.children[index].visit_count / sum_visits # visit prob or prior target
# =============================================================================
#     print(child_visit.shape)
#     print(root.value())
# =============================================================================
    self.child_visits.append(child_visit)
    self.root_values.append(np.array(root.value(), ndmin=2)) # how good this node is. aka value target
    
  def store_sl_statistics(self, p, v):
    self.child_visits.append(p)
    self.root_values.append(v)
    
    


# Given a board and move, the move is represented by an integer
# Adapted code from https://github.com/ZiyuanMa/MuZero/blob/master/pseudocode.py

def encode_action(board: chess.Board, move: chess.Move):
    if not Game.is_move_legal(board, move):
        print(board.fen(), move)
        raise Exception('Cannot encode illegal move')

    initial_pos = move.from_square
    final_pos = move.to_square
    piece = board.piece_at(initial_pos).symbol()
    underpromote = chess.piece_name(move.promotion) if move.promotion else None

    encoded = np.zeros([8, 8, 73]).astype(int)
    j, i = chess.square_file(initial_pos), chess.square_rank(initial_pos)
    y, x = chess.square_file(final_pos), chess.square_rank(final_pos)
    #  file = a to h, rank =  1 to 8
    dx, dy = x-i, y-j
    if piece in ["R", "B", "Q", "K", "P", "r", "b", "q", "k", "p"] and underpromote in [None, "queen"]:  # queen-like moves
        if dx != 0 and dy == 0:  # north-south idx 0-13
            if dx < 0:
                idx = 7 + dx
            elif dx > 0:
                idx = 6 + dx
        if dx == 0 and dy != 0:  # east-west idx 14-27
            if dy < 0:
                idx = 21 + dy
            elif dy > 0:
                idx = 20 + dy
        if dx == dy:  # NW-SE idx 28-41
            if dx < 0:
                idx = 35 + dx
            if dx > 0:
                idx = 34 + dx
        if dx == -dy:  # NE-SW idx 42-55
            if dx < 0:
                idx = 49 + dx
            if dx > 0:
                idx = 48 + dx
    if piece in ["n", "N"]:  # Knight moves 56-63
        if (x, y) == (i+2, j-1):
            idx = 56
        elif (x, y) == (i+2, j+1):
            idx = 57
        elif (x, y) == (i+1, j-2):
            idx = 58
        elif (x, y) == (i-1, j-2):
            idx = 59
        elif (x, y) == (i-2, j+1):
            idx = 60
        elif (x, y) == (i-2, j-1):
            idx = 61
        elif (x, y) == (i-1, j+2):
            idx = 62
        elif (x, y) == (i+1, j+2):
            idx = 63
    # underpromotions
    if piece in ["p", "P"] and (x == 0 or x == 7) and underpromote != None:
        if abs(dx) == 1 and dy == 0:
            if underpromote == "rook":
                idx = 64
            if underpromote == "knight":
                idx = 65
            if underpromote == "bishop":
                idx = 66
        if abs(dx) == 1 and dy == -1:
            if underpromote == "rook":
                idx = 67
            if underpromote == "knight":
                idx = 68
            if underpromote == "bishop":
                idx = 69
        if abs(dx) == 1 and dy == 1:
            if underpromote == "rook":
                idx = 70
            if underpromote == "knight":
                idx = 71
            if underpromote == "bishop":
                idx = 72
    encoded[i, j, idx] = 1
    encoded = encoded.reshape(-1)
    encodedArray = encoded
    encoded = np.where(encoded == 1)[0][0]  # index of action
    return (encodedArray, encoded)

# Decodes an encoded move


def decode_action(board: chess.Board, encoded: int) -> chess.Move:
    encoded_a = np.zeros([4672])
    encoded_a[encoded] = 1
    encoded_a = encoded_a.reshape(8, 8, 73)

    a, b, c = np.where(encoded_a == 1)  # i,j,k = i[0],j[0],k[0]
    i_pos, f_pos, prom = [], [], []
    moves = []
    player = Game.turn(board)

    PIECE_TYPE_MAP = {
        "R": 4, "r": 4,
        "N": 2, "n": 2,
        "B": 3, "b": 3,
        "Q": 5, "q": 5,
        "K": 6, "k": 6,
        "P": 1, "p": 1,
        None: None,
    }

    for pos in zip(a, b, c):
        i, j, k = pos
        initial_pos = (i, j)  # i = rank, j = file
        promoted = None
        if 0 <= k <= 13:
            dy = 0
            if k < 7:
                dx = k - 7
            else:
                dx = k - 6
            final_pos = (i + dx, j + dy)
        elif 14 <= k <= 27:
            dx = 0
            if k < 21:
                dy = k - 21
            else:
                dy = k - 20
            final_pos = (i + dx, j + dy)
        elif 28 <= k <= 41:
            if k < 35:
                dy = k - 35
            else:
                dy = k - 34
            dx = dy
            final_pos = (i + dx, j + dy)
        elif 42 <= k <= 55:
            if k < 49:
                dx = k - 49
            else:
                dx = k - 48
            dy = -dx
            final_pos = (i + dx, j + dy)
        elif 56 <= k <= 63:
            if k == 56:
                final_pos = (i+2, j-1)
            elif k == 57:
                final_pos = (i+2, j+1)
            elif k == 58:
                final_pos = (i+1, j-2)
            elif k == 59:
                final_pos = (i-1, j-2)
            elif k == 60:
                final_pos = (i-2, j+1)
            elif k == 61:
                final_pos = (i-2, j-1)
            elif k == 62:
                final_pos = (i-1, j+2)
            elif k == 63:
                final_pos = (i+1, j+2)
        else:
            if k == 64:
                if player == 0:
                    final_pos = (i-1, j)
                    promoted = "R"
                if player == 1:
                    final_pos = (i+1, j)
                    promoted = "r"
            if k == 65:
                if player == 0:
                    final_pos = (i-1, j)
                    promoted = "N"
                if player == 1:
                    final_pos = (i+1, j)
                    promoted = "n"
            if k == 66:
                if player == 0:
                    final_pos = (i-1, j)
                    promoted = "B"
                if player == 1:
                    final_pos = (i+1, j)
                    promoted = "b"
            if k == 67:
                if player == 0:
                    final_pos = (i-1, j-1)
                    promoted = "R"
                if player == 1:
                    final_pos = (i+1, j-1)
                    promoted = "r"
            if k == 68:
                if player == 0:
                    final_pos = (i-1, j-1)
                    promoted = "N"
                if player == 1:
                    final_pos = (i+1, j-1)
                    promoted = "n"
            if k == 69:
                if player == 0:
                    final_pos = (i-1, j-1)
                    promoted = "B"
                if player == 1:
                    final_pos = (i+1, j-1)
                    promoted = "b"
            if k == 70:
                if player == 0:
                    final_pos = (i-1, j+1)
                    promoted = "R"
                if player == 1:
                    final_pos = (i+1, j+1)
                    promoted = "r"
            if k == 71:
                if player == 0:
                    final_pos = (i-1, j+1)
                    promoted = "N"
                if player == 1:
                    final_pos = (i+1, j+1)
                    promoted = "n"
            if k == 72:
                if player == 0:
                    final_pos = (i-1, j+1)
                    promoted = "B"
                if player == 1:
                    final_pos = (i+1, j+1)
                    promoted = "b"
        # i = rank, j = file
        # final_pos[0] = rank, final_pos[1] = file
        piece = board.piece_at(chess.square(j, i)).symbol()
        # auto-queen promotion for pawn
        if piece in ["P", "p"] and final_pos[0] in [0, 7] and promoted == None:
            if player == 0:
                promoted = "Q"
            else:
                promoted = "q"
        i_pos.append(initial_pos)
        f_pos.append(final_pos), prom.append(promoted)
        moves.append(chess.Move(chess.square(j, i), chess.square(
            final_pos[1], final_pos[0]), PIECE_TYPE_MAP[promoted]))
    return moves[0]  # i_pos, f_pos, prom

# =============================================================================
# # Encoding board state
# # Source of function - https://chess.stackexchange.com/questions/29294/quickly-converting-board-to-bitboard-representation-using-python-chess-library
# def bitboards_to_array(board: chess.Board) -> np.ndarray:
#     black, white = board.occupied_co
#     bitboards = np.array([
#         white & board.pawns,
#         white & board.knights,
#         white & board.bishops,
#         white & board.rooks,
#         white & board.queens,
#         white & board.kings,
#         black & board.pawns,
#         black & board.knights,
#         black & board.bishops,
#         black & board.rooks,
#         black & board.queens,
#         black & board.kings,
#     ], dtype=np.uint64)
#     bb = np.asarray(bitboards, dtype=np.uint64)[:, np.newaxis]
#     s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
#     b = (bb >> s).astype(np.uint8)
#     b = np.unpackbits(b, bitorder="little")
#     return b.reshape(-1, 8, 8)
# =============================================================================

# Encode board as (103, 8, 8) tensor where -- 119 = 14 * 8 (hist length) + 7
def encode_board(board: chess.Board) -> torch.Tensor:
# =============================================================================
#     mvs_popped = []
#     i, history = 1, 8
# =============================================================================
    encoded_board = np.zeros([8,8,22]).astype(int)
    encoder_dict = {"R":0, "N":1, "B":2, "Q":3, "K":4, "P":5, "r":6, "n":7, "b":8, "q":9, "k":10, "p":11}
    for rank in range(8):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece != None:
                piece = piece.symbol()
                encoded_board[rank,file,encoder_dict[piece]] = 1
# =============================================================================
#     board_array = bitboards_to_array(board)
# =============================================================================
# =============================================================================
#     while len(board.move_stack) > 0 and i < history:
#         mvs_popped.append(board.pop())
#         board_array = np.concatenate((board_array, bitboards_to_array(board)))
#         i += 1
# 
#     while i < history:
#         board_array = np.concatenate(
#             (board_array, [np.zeros((8, 8)) for ii in range(12)]))
#         i += 1
# 
#     while len(mvs_popped) != 0:
#         m = mvs_popped.pop()
#         board.push(m)
# =============================================================================

    repetitions_w = 0
    repetitions_b = 0
    
    for repeat_count in range(1,4):
        is_rep = board.is_repetition(repeat_count)
        if is_rep:
            if Game.turn(board):
                # white to play -> black just played
                repetitions_b += 1
            else:
                repetitions_w += 1
                
    current_player = Game.turn(board)
    
    encoded_board[:,:, 12] = current_player # player to move
    encoded_board[:,:, 13] = len(board.move_stack) # num moves so far
    
    encoded_board[:,:, 14] = repetitions_w
    encoded_board[:,:, 15] = repetitions_b
    
    encoded_board[:,:, 16] = int(board.has_kingside_castling_rights(chess.WHITE))
    encoded_board[:,:, 17] = int(board.has_queenside_castling_rights(chess.WHITE))
    encoded_board[:,:, 18] = int(board.has_kingside_castling_rights(chess.BLACK))
    encoded_board[:,:, 19] = int(board.has_queenside_castling_rights(chess.BLACK))
    
    encoded_board[:,:, 20] = int(board.halfmove_clock)
    encoded_board[:,:, 21] = int(board.has_legal_en_passant())
    t = torch.from_numpy(encoded_board).to(dtype=torch.float32)
    return t
