from NNetwork import Network, load_network, NetworkOutput, train_network, save_network
from Game import Game, encode_board
from Node import Node
from MCTS import expand_node, add_exploration_noise, run_mcts, select_action 
import config
from ReplayBuffer import ReplayBuffer
import time
from utils import show_svg
from IPython.display import SVG
import chess
import chess.svg
import torch

def get_compute_device():
    compute_device = None
    # detect gpu/cpu device to use
    if torch.backends.cuda.is_built():
        compute_device = torch.device('cuda:0') # 0th CUDA device
    if torch.backends.mps.is_available():
        compute_device = torch.device('mps') # For Apple silicon
    else:
        compute_device = torch.device("cpu") # Use CPU if no GPU
    return compute_device

compute_device = get_compute_device()

def play_game(network: Network) -> Game:
  game = Game()
  while not game.over() and len(game.board.move_stack) < config.max_moves:
    root = Node(0, game.board.copy(stack=False))
    (root, max_d) = run_mcts(root, network)
    (move, action_no) = select_action(root)
    game.make_move(move)
    show_svg(SVG(chess.svg.board(game.board)))
    print(game.board)
    game.store_search_statistics(root)
    print(root.to_play, 'plays', move,' ---- v, p, count ', root.value(), root.children[action_no].prior, root.children[action_no].visit_count,'max_d', max_d, sep=' ', end='\n')
  return game

def run_selfplay(network: Network, replay_buffer: ReplayBuffer, loop, n = 10):
  i = 0
  while i < n:
    start = time.time()
    game = play_game(network)
    end = time.time()
    replay_buffer.save_game(game)
    print('loop', loop, 'Self play Game', i + 1,' -- ', end - start, 'time elapsed')
    #show_svg(game.board)
    i += 1
    
if __name__ == "__main__":
    # train loop
    train_loops_elapsed = 0
    while True:
        print('loop ', train_loops_elapsed + 1)
        network = load_network()
        replay_buffer_self_play = ReplayBuffer()
        run_selfplay(network, replay_buffer_self_play, train_loops_elapsed + 1)
        print('!!! games played',len(replay_buffer_self_play.buffer))
        train_network(network, replay_buffer_self_play, epoch_start=0, n_epochs=5) # prevent overfitting since dataset is small
        train_loops_elapsed += 1