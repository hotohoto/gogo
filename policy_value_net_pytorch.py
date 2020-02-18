# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import go

from constants import PASS_MOVE

torch.manual_seed(0)

N_BODY_CHANNELS = 32  # 256 for AlphaZero
N_BODY_RESIDUAL_BLOCKS = 4  # 19 for AlphaZero
N_ACTION_HEAD_CHANNELS = 16
N_VALUE_HEAD_CHANNELS = 16
VALUE_HIDDEN_SIZE = 64


def get_current_input(game_state: go.GameState):
    """return the board state from the perspective of the current player.
        state shape: 2*width*height
    """

    current_player = game_state.current_player

    if current_player == go.BLACK:
        opponent = go.WHITE
    elif current_player == go.WHITE:
        opponent = go.BLACK

    current_player_map = game_state.board == current_player
    opponent_map = game_state.board == opponent

    return np.array([current_player_map, opponent_map], dtype=float)


class Net(nn.Module):
    """policy-value network module"""

    def __init__(self, board_size):
        super(Net, self).__init__()

        self.board_size = board_size

        # common layers
        self.body_conv = []
        self.body_bn = []
        for i in range(N_BODY_RESIDUAL_BLOCKS * 2 + 1):
            n_input_channels = 2 if i == 0 else N_BODY_CHANNELS
            n_output_channels = N_BODY_CHANNELS
            self.body_conv.append(
                nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, padding=1)
            )
            self.body_bn.append(nn.BatchNorm2d(N_BODY_CHANNELS))

        # action policy layers
        self.act_conv1 = nn.Conv2d(
            N_BODY_CHANNELS, N_ACTION_HEAD_CHANNELS, kernel_size=1
        )
        self.act_bn_conv1 = nn.BatchNorm2d(N_ACTION_HEAD_CHANNELS)
        self.act_fc1 = nn.Linear(
            N_ACTION_HEAD_CHANNELS * board_size * board_size,
            board_size * board_size + 1,
        )

        # state value layers
        self.val_conv1 = nn.Conv2d(
            N_BODY_CHANNELS, N_VALUE_HEAD_CHANNELS, kernel_size=1
        )
        self.val_bn_conv1 = nn.BatchNorm2d(N_VALUE_HEAD_CHANNELS)
        self.val_fc1 = nn.Linear(
            N_VALUE_HEAD_CHANNELS * board_size * board_size, VALUE_HIDDEN_SIZE
        )
        self.val_bn_fc1 = nn.BatchNorm1d(VALUE_HIDDEN_SIZE)
        self.val_fc2 = nn.Linear(VALUE_HIDDEN_SIZE, 1)

    def forward(self, state_input):
        # common layers
        x1 = self.body_conv[0](state_input)
        x2 = self.body_bn[0](x1)
        x0 = F.relu(x2)

        for i in range(N_BODY_RESIDUAL_BLOCKS):
            x1 = self.body_conv[2 * i + 1](x0)
            x2 = self.body_bn[2 * i + 1](x1)
            x3 = F.relu(x2)
            x4 = self.body_conv[2 * i + 2](x3)
            x5 = self.body_bn[2 * i + 2](x4)
            x0 = F.relu(x5 + x0)

        # action policy layers
        x_act = F.relu(self.act_bn_conv1(self.act_conv1(x0)))
        x_act = x_act.view(
            -1, N_ACTION_HEAD_CHANNELS * self.board_size * self.board_size
        )
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.act_bn_conv1(self.val_conv1(x0)))
        x_val = x_val.view(
            -1, N_VALUE_HEAD_CHANNELS * self.board_size * self.board_size
        )
        x_val = F.relu(self.val_bn_fc1(self.val_fc1(x_val)))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    """policy-value network """

    def __init__(self, board_size, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_size = board_size
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_size).cuda()
        else:
            self.policy_value_net = Net(board_size)
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(), weight_decay=self.l2_const
        )

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def set_eval_mode(self):
        self.policy_value_net.eval()

    def set_train_mode(self):
        self.policy_value_net.train()

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, game_state: go.GameState):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_moves = (
            np.array(game_state.get_legal_moves(False)) @ np.array([game_state.size, 1])
        ).tolist()
        legal_moves_idx = legal_moves + [game_state.size ** 2]
        legal_moves = legal_moves + [PASS_MOVE]

        current_input = np.expand_dims(get_current_input(game_state), axis=0)
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_input)).cuda().float()
            )
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_input)).float()
            )
            act_probs = np.exp(log_act_probs.data.numpy()).flatten()
        _act_probs = zip(legal_moves, act_probs[legal_moves_idx])
        value = value.data[0][0]
        return _act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
