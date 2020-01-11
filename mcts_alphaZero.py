# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np

import go
from config import DEFAULT_ALPHA_ZERO_N_PLAYOUT
from constants import PASS_MOVE


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0  # total score
        self._P = prior_p  # policy prob.

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(
            self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(
        self, policy_value_net, c_puct=5, n_playout=DEFAULT_ALPHA_ZERO_N_PLAYOUT
    ):
        """
        policy_value_net: network that takes in a board inputs and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy_value_net = policy_value_net
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, game_state: go.GameState):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            if action == PASS_MOVE:
                move = PASS_MOVE
            else:
                move = (action // game_state.size, action % game_state.size)
            game_state.do_move(move)

        if not game_state.is_end_of_game and len(game_state.get_legal_moves(False)) > 0:
            action_probs, leaf_value = self._policy_value_net.policy_value_fn(
                game_state
            )
            node.expand(action_probs)
        else:
            winner = game_state.get_winner()
            # for end state，return the "true" leaf_value
            if winner is None:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == game_state.current_player else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, game_state: go.GameState, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for _ in range(self._n_playout):
            state_copy = game_state.copy()
            self._playout(state_copy)

        if self._root.is_leaf():
            return None, None

        # calc the move probabilities based on visit counts at the root node
        act_visits = [
            (act, node._n_visits) for act, node in self._root._children.items()
        ]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move and last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(
        self,
        policy_value_net,
        c_puct=5,
        n_playout=DEFAULT_ALPHA_ZERO_N_PLAYOUT,
        is_selfplay=False,
    ):
        self.mcts = MCTS(policy_value_net, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(None)

    def get_action(self, game_state, temp=1e-3):
        self.mcts._policy_value_net.set_eval_mode()

        # the pi vector returned by MCTS as in the alphaGo Zero paper
        n_all_actions = game_state.size ** 2 + 1

        if not game_state.is_end_of_game:
            acts, probs = self.mcts.get_move_probs(game_state, temp)
            if acts is None:
                dense_probs = np.zeros(n_all_actions)
                dense_probs[-1] = 1
                return PASS_MOVE, dense_probs

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs
                    + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(None)

            dense_probs = np.zeros(n_all_actions)
            for idx, act_idx in enumerate(acts):
                dense_probs[act_idx if act_idx != PASS_MOVE else -1] = probs[idx]

            return move, dense_probs
        else:
            print("WARNING: game is end")

    def __str__(self):
        return "MCTSAlphaZeroPlayer"
