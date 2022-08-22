from typing import collections

# Config
action_space_size=4672
max_moves=512
dirichlet_alpha=0.3
lr_init=0.003#0.1
lr_decay_rate = 0.1
lr_decay_steps=400e3
num_simulations=500 # 50% of Alphazero
batch_size=64#256
td_steps=max_moves # Always use Monte Carlo return.
num_actors=3000
momentum = 0.9
num_unroll_steps=5
training_steps = 30#int(1000e3)
checkpoint_interval = int(1e3)
window_size = int(1e6)
root_exploration_fraction = 0.25
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])
known_bounds=KnownBounds(-1, 1)
pb_c_base = 19652
pb_c_init = 1.25
discount=1.0
MAXIMUM_FLOAT_VALUE = float('inf')
model_save_path='./models/'#'/content/drive/MyDrive/chess-ai/'
stockfish_model_path='/content/drive/MyDrive/chess-ai/stockfish_15_x64'