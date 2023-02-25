import chess
import chess.pgn
from Game import Game, encode_action, encode_board
from ReplayBuffer import ReplayBuffer
import numpy as np
from NNetwork import train_network, load_network
import os

def get_games_from_file(pgn_file_path: str) -> ReplayBuffer:
    pgn = open(pgn_file_path)
    games_read, moves_captured = 0, 0
    replay_buffer = ReplayBuffer()
    
    game_from_pgn = chess.pgn.read_game(pgn)
    while game_from_pgn != None:
        game = Game()
        for move in game_from_pgn.mainline_moves():
            (p, move_encoded) = encode_action(game.board, move)
            v = np.random.random((1,1)) * 2 - 1
            game.store_sl_statistics(p, v)
            game.make_move(move) # , encoded_board_state=encode_board(game.board)
            moves_captured += 1
        replay_buffer.save_game(game)
        games_read += 1
        game_from_pgn = chess.pgn.read_game(pgn)
    pgn.close()
    print(games_read, ' games read with ', moves_captured, ' moves captured')
    return replay_buffer
        


if __name__ == "__main__":
    for filename in os.listdir('./games'):
        if filename == ".DS_Store":
            continue
        print(filename)
        relay_buffer_sl = get_games_from_file(("./games/" + filename))
        network = load_network()
        train_network(network, relay_buffer_sl, n_epochs=15)