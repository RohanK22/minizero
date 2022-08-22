from config import MAXIMUM_FLOAT_VALUE, KnownBounds, known_bounds, num_simulations, discount, pb_c_base, pb_c_init, dirichlet_alpha, root_exploration_fraction
from typing import Optional, List
from NNetwork import Network, NetworkOutput
from Node import Node
import chess
from Action import Action
from Game import decode_action, encode_action, encode_board
import math
import numpy as np
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

# forward prop
def recurrent_inference(network: Network, state: chess.Board) -> NetworkOutput:
    img = encode_board(state).to(compute_device)
    (p, v) = network(img)
    return NetworkOutput(p.detach().cpu().numpy()[0], v.detach().cpu().numpy()[0], state)

# MCTS Algorithm components
def get_reward(outcome: chess.Outcome, root_to_play: int):
    if outcome == None:
        return None # draw
    elif outcome.winner == None:
        return 0 # !!!
    else:
        root_wins = int(1 == root_to_play if outcome.winner == chess.WHITE else root_to_play == 0) 
        print("black wins" if outcome.winner == 0 else "white wins")
        print('root to play', root_to_play)
        if root_wins:
            return 1
        else:
            return -1

# A class that holds the min-max values of the tree. - used for normalizing values of nodes in the tree
class MinMaxStats(object):

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

# MCTS Algorithm
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(root: Node, network: Network):
    # Expand root first
    network_output = recurrent_inference(network, root.state)
    expand_node(root, root.to_play, list(root.state.legal_moves), network_output)
    add_exploration_noise(root)
    
    max_d = 0
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(node)
            search_path.append(node)

        parent = search_path[-2]

        # win
        outcome = node.state.outcome()
        reward = None
        if outcome != None:
            reward = get_reward(outcome, node.to_play) # -1 if losing for node to_play, +1 if winning for node to_play
            print('r', reward)
            print(node.state.fen())
            # directly backprop
        else:
            # continue expanding
            network_output = recurrent_inference(network, node.state)
            expand_node(node, node.to_play, list(node.state.legal_moves), network_output)
        
        backpropagate(search_path, reward if outcome else network_output.value, node.to_play)
        max_d = max(max_d, len(search_path))
    return (root, max_d)

# Select the child with the highest UCB score.
def select_child(node: Node):
  _, action, child = max(
      (ucb_score(node, child), action,
       child) for action, child in node.children.items()) # if there is a tie should it pick the action with highest action index? - Q3
  return action, child

def select_action(node: Node):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  cur_max_visit_count, selected_action_no = max(visit_counts)
  selected_move = decode_action(node.state, selected_action_no)
  return (selected_move, selected_action_no)

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node) -> float:
  #pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init # could simplify? - Q2
  pb_c = math.sqrt(parent.visit_count) / (child.visit_count + 1)
  prior_score = pb_c * child.prior
  # -1 if position is bad for child to_play, +1 if position is good for child to_play
  value_score = (-child.value() + 1)/2 # The value of the child is from the perspective of the parent/opposing player
  return prior_score + value_score # takes account of both policy and value heads


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: int, moves: List[chess.Move], network_output: NetworkOutput):
  (p, v, state) = network_output
  policy_sum = 1 # since we softmax the o/p already no need to normalize
  for m in moves:
    (arr, action_no) = encode_action(node.state, m)
    new_board = node.state.copy(stack=False)
    new_board.push(m)
    node.children[action_no] = Node(p[action_no]/ policy_sum, new_board)
#     print('action no ', action, ' policy pred ', p)

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, root_to_play: int):
    for node in search_path:
        node.value_sum += value if node.to_play == root_to_play else -value
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(node: Node):
  actions = list(node.children.keys())
  noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
  frac = root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac