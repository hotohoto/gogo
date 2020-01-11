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

COMMON_CHANNEL = 32
ACTION_CHANNEL = 16
VALUE_CHANNEL = 16
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
        self.conv0 = nn.Conv2d(2, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(COMMON_CHANNEL, COMMON_CHANNEL, kernel_size=3, padding=1)

        self.bn_conv0 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv1 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv2 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv3 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv4 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv5 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv6 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv7 = nn.BatchNorm2d(COMMON_CHANNEL)
        self.bn_conv8 = nn.BatchNorm2d(COMMON_CHANNEL)

        # action policy layers
        self.act_conv1 = nn.Conv2d(COMMON_CHANNEL, ACTION_CHANNEL, kernel_size=1)
        self.act_bn_conv1 = nn.BatchNorm2d(ACTION_CHANNEL)
        self.act_fc1 = nn.Linear(
            ACTION_CHANNEL * board_size * board_size, board_size * board_size + 1
        )

        # state value layers
        self.val_conv1 = nn.Conv2d(COMMON_CHANNEL, VALUE_CHANNEL, kernel_size=1)
        self.val_bn_conv1 = nn.BatchNorm2d(VALUE_CHANNEL)
        self.val_fc1 = nn.Linear(
            VALUE_CHANNEL * board_size * board_size, VALUE_HIDDEN_SIZE
        )
        self.val_bn_fc1 = nn.BatchNorm1d(VALUE_HIDDEN_SIZE)
        self.val_fc2 = nn.Linear(VALUE_HIDDEN_SIZE, 1)

    def forward(self, state_input):
        # common layers
        x0 = F.relu(self.bn_conv0(self.conv0(state_input)))
        x1 = F.relu(self.bn_conv1(self.conv1(x0)))
        x2 = F.relu(self.bn_conv2(self.conv2(x1)) + x0)
        x3 = F.relu(self.bn_conv3(self.conv3(x2)))
        x4 = F.relu(self.bn_conv4(self.conv4(x3)) + x2)
        x5 = F.relu(self.bn_conv5(self.conv5(x4)))
        x6 = F.relu(self.bn_conv6(self.conv6(x5)) + x4)
        x7 = F.relu(self.bn_conv7(self.conv7(x6)))
        x8 = F.relu(self.bn_conv8(self.conv8(x7)) + x6)

        # action policy layers
        x_act = F.relu(self.act_bn_conv1(self.act_conv1(x8)))
        x_act = x_act.view(-1, ACTION_CHANNEL * self.board_size * self.board_size)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.act_bn_conv1(self.val_conv1(x8)))
        x_val = x_val.view(-1, VALUE_CHANNEL * self.board_size * self.board_size)
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
