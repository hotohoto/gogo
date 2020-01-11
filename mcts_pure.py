# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)
"""

from operator import itemgetter

import numpy as np

import go
from config import DEFAULT_PURE_MCTS_N_PLAYOUT
from constants import PASS_MOVE


def _rollout_policy_fn(game_state: go.GameState):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    legal_moves = (
        np.array(game_state.get_legal_moves(False)) @ np.array([game_state.size, 1])
    ).tolist()
    legal_moves = legal_moves + [PASS_MOVE]
    action_probs = np.random.rand(len(legal_moves))
    return zip(legal_moves, action_probs)


def _policy_value_fn(game_state: go.GameState):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    legal_moves = (
        np.array(game_state.get_legal_moves(False)) @ np.array([game_state.size, 1])
    ).tolist()
    legal_moves = legal_moves + [PASS_MOVE]
    action_probs = np.ones(len(legal_moves)) / len(legal_moves)
    return zip(legal_moves, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._P = prior_p

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
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(
        self, policy_value_fn, c_puct=5, n_playout=DEFAULT_PURE_MCTS_N_PLAYOUT
    ):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, game_state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        game_state is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break

            # Greedily select next move.
            action, node = node.select(self._c_puct)
            if action == PASS_MOVE:
                move = PASS_MOVE
            else:
                move = (action // game_state.size, action % game_state.size)
            game_state.do_move(move)

        # Check for end of game
        if not game_state.is_end_of_game and len(game_state.get_legal_moves(False)) > 0:
            action_probs, _ = self._policy_value_fn(game_state)
            node.expand(action_probs)

        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(game_state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, game_state: go.GameState, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = game_state.current_player
        for _ in range(limit):
            if game_state.is_end_of_game or len(game_state.get_legal_moves(False)) == 0:
                break

            action_probs = _rollout_policy_fn(game_state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            if max_action == PASS_MOVE:
                move = PASS_MOVE
            else:
                move = (max_action // game_state.size, max_action % game_state.size)
            game_state.do_move(move)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        winner = game_state.get_winner()
        if winner is None:  # tie
            return 0.0
        else:
            return 1.0 if winner == player else -1.0

    def get_move(self, game_state: go.GameState):
        """Runs all playouts sequentially and returns the most visited action.
        game_state: the current game state

        Return: the selected action
        """
        for _ in range(self._n_playout):
            state_copy = game_state.copy()
            self._playout(state_copy)
        if self._root.is_leaf():
            return PASS_MOVE
        return max(
            self._root._children.items(), key=lambda act_node: act_node[1]._n_visits
        )[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPurePlayer(object):
    """AI player based on MCTS"""

    def __init__(self, c_puct=5, n_playout=DEFAULT_PURE_MCTS_N_PLAYOUT):
        self.mcts = MCTS(_policy_value_fn, c_puct, n_playout)

    def reset_player(self):
        self.mcts.update_with_move(None)

    def get_action(self, game_state: go.GameState):
        if not game_state.is_end_of_game:
            move = self.mcts.get_move(game_state)
            self.mcts.update_with_move(None)
            return move, None
        else:
            print("WARNING: game is end")

    def __str__(self):
        return "MCTSPurePlayer"
