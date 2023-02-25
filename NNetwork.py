import torch
from torch import nn
from torch.nn import functional as F
from typing import NamedTuple
import chess
from ReplayBuffer import ReplayBuffer
import config
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import datetime
import numpy as np

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

class OutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Value head
        self.valueHeadConv = nn.Conv2d(256, 1, 1, 1)
#        self.valueHeadConv
        # .to(compute_device)
        
        # relu
        # flatten 64
        self.valueHeadLinear1 = nn.Linear(64, 256) # could remove this
        self.valueHeadLinear1
        self.valueHeadLinear2 = nn.Linear(256, 1)
        self.valueHeadLinear2
        # tanh

        # Policy head
        self.policyHeadConv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.policyHeadConv1
        self.policyHeadConv2 = nn.Conv2d(256, 73, kernel_size=1)
        self.policyHeadConv2
        # flatten and softmax
    
    def forward(self, x):
        v = F.relu(self.valueHeadConv(x))
        v = torch.flatten(v)
        v = v.view(-1, 8*8) 
        v = F.relu(self.valueHeadLinear1(v))
        v = self.valueHeadLinear2(v)
        v = torch.tanh(v) # -1 to 1 value
        v = v.view(-1, 1, 1) 

        p = F.relu(self.policyHeadConv1(x))
        p = F.relu(self.policyHeadConv2(p))
        p = torch.flatten(p)
        p = p.view(-1, 73 * 8 * 8) 
        p = F.softmax(p, dim=-1) # move probablities - move no -> probablitiy
        
        return (p, v)

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(22, 256, kernel_size=3, padding=1)
    # relu
    self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    # relu
    self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    # # relu
    self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    # # relu
    self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    # relu
    
    self.outblock = OutBlock()
    self.outblock.to(compute_device)

  def forward(self, x):
#     x.to(mps_device)
    x = x.view(-1, 22, 8, 8)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = self.outblock(x)
    # x = F.relu(self.conv3(x))
#     v = F.relu(self.valueHeadConv(torch.clone(x)))
#     v = torch.flatten(v)

#     v = F.relu(self.valueHeadLinear1(v))
#     v = torch.tanh(self.valueHeadLinear2(v)) # -1 to 1 value

#     p = F.relu(self.policyHeadConv1(torch.clone(x)))
#     p = F.relu(self.policyHeadConv2(p))
#     p = torch.flatten(p)
#     p = F.softmax(p, dim=-1) # move probablities - move no -> probablitiy

    return x #(p, v)



class NetworkOutput(NamedTuple): # (p, v, state)
    policy_logits: torch.Tensor # 1d tensor
    value: float # -1 to 1
    state: chess.Board

def get_reward(outcome: chess.Outcome, root_to_play: int):
    if outcome == None:
        return 0 # draw
    else:
        root_wins = int(1 == root_to_play if outcome.winner == chess.WHITE else root_to_play == 0) 
        if root_wins:
            return 1
        else:
            return -1
        
class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, value, y_value, policy, y_policy, network: Network):
        value_error = (value - y_value) ** 2
        #print('cross entropy loss shape', (-policy* torch.log(1e-6 + y_policy)).shape)
        policy_error = torch.sum((-policy* torch.log(1e-6 + y_policy)), -1)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in network.parameters())
        norm_loss = l2_lambda * l2_norm
        total_error = (value_error.view(-1).float() + policy_error).mean() + norm_loss
        return total_error


# Get Torch Dataset from ReplayBuffer
class GameDataSet(Dataset):
    def __init__(self, replay_buffer: ReplayBuffer): # dataset = np.array of (s, p, v)
        dataset = []
        for game in replay_buffer.buffer:
            for i, state in enumerate(game.states):
                dataset.append((state.detach().cpu().numpy(),
                                game.child_visits[i],
                                game.root_values[i]
# =============================================================================
#                                 torch.tensor(game.child_visits[i], dtype=torch.float32).to(compute_device), 
#                                 torch.tensor(game.root_values[i], dtype=torch.float32).to(compute_device)
# =============================================================================
                            ))
        dataset = np.array(dataset)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y_p[idx], self.y_v[idx]

def load_network():
  # paths = os.listdir(model_save_path)
  # cur_max = 0
  # for path in paths:
  #   if os.path.isfile(os.path.join(model_save_path, path)):# and int(path[-1]) > cur_max:
  #       p = path
  # print(p)
  model = Network()
  fp = config.model_save_path + 'mz-largest'
  if os.path.isfile(fp):
      model.load_state_dict(torch.load(fp, map_location=torch.device('cpu')))
  else:
      print('Model mz not found so creating new one')
      save_network(model)
  model.eval()
  return model.to(get_compute_device())

def save_network(network: Network):
    torch.save(network.state_dict(), config.model_save_path + 'mz-largest')# + 'minizero_' + str(train_loops))


def train_network(network: Network, replay_buffer: ReplayBuffer, epoch_start = 0, n_epochs = 20):
    print('train begin')
    torch.autograd.set_detect_anomaly(True)
    
    network.train()
    optimizer = torch.optim.SGD(network.parameters(), lr=config.lr_init, momentum=config.momentum)
    criterion = AlphaLoss().to(compute_device)
    
    train_set = GameDataSet(replay_buffer)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = []
    
    for epoch in range(epoch_start, n_epochs):
        print('epoch', epoch)
        total_loss = 0
        losses_per_batch = []
        
        for i,data in enumerate(train_loader,0):
            state, policy, value = data
            state = torch.tensor(state, dtype=torch.float32).to(compute_device)
            policy = torch.tensor(policy, dtype=torch.float32).to(compute_device)
            value = torch.tensor(value, dtype=torch.float32).to(compute_device)
            #value = value.view(-1, 1, 1) 
# =============================================================================
#             print(state.shape)
#             print(policy.shape)
#             print(value.shape)
# =============================================================================
            optimizer.zero_grad()
            policy_pred, value_pred = network(state)
# =============================================================================
#             print(policy_pred.shape)
#             print(value_pred.shape)
# =============================================================================
            loss = criterion(value_pred.clone(), value, policy_pred.clone(), policy, network)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches of size = batch_size
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (os.getpid(), epoch + 1, (i + 1) * config.batch_size, len(train_set), total_loss/10))
                print("Policy:",policy[0].argmax().item(),policy_pred[0].argmax().item())
                print("Value:",value[0].item(),value_pred[0,0].item())
                losses_per_batch.append(total_loss/10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch))
        # save model
        print('save model')
        save_network(network)
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1,n_epochs+1,1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    print('Finished Training')
    plt.savefig(os.path.join("./model_data/", "Loss_vs_Epoch_%s.png" % datetime.datetime.today().strftime("%Y-%m-%d")))
# =============================================================================
#         for image, target_p, target_v in batch:
#             target_p = torch.tensor(target_p, dtype=torch.float32).to(compute_device) # torch.from_numpy(target_p).to(mps_device)
#             target_v = torch.tensor(target_v, dtype=torch.float32).to(compute_device)
#             optimizer.zero_grad()
# 
#             (p, v) = network(image)
# #             print('prediction - p: ', torch.max(p).item(), ' v: ', v.item())
# #             print('actual - p: ', torch.max(target_p).item(), ' v: ', target_v.item())
#             loss = criterion(target_v, v.clone(), target_p, p.clone())# F.mse_loss(v, target_v) + torch.sum(- target_p * F.log_softmax(p, -1), -1)
#             #       print('individual loss: ', loss.item(), end=',')
#             loss.backward()
#             optimizer.step()
#             loss += loss.item()
#         print('avg loss: ', loss)
# =============================================================================
