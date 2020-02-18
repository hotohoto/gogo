# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

import go
from config import Config
from constants import PASS_MOVE


class Game(object):
    """game server"""

    def __init__(self, game_state: go.GameState):
        self.initial_game_state = game_state
        self.init_game_state()

    def init_game_state(self):
        self.game_state = self.initial_game_state.copy()

    @staticmethod
    def from_config(config: Config):
        game_state = go.GameState(
            size=config.size, komi=config.komi, enforce_superko=config.enforce_superko
        )
        return Game(game_state)

    def print_board(self, game_state, player1_desc, player2_desc, move_probs=None):
        """Draw the board and show game info"""

        size = game_state.size
        score_black, score_white = game_state.get_scores()

        current_move_text = "*".center(8)
        if game_state.history:
            if game_state.history[-1] == PASS_MOVE:
                last_move_text = "PASS"
            else:
                last_move_text = str(game_state.history[-1])
        else:
            last_move_text = ""
        last_move_text = last_move_text.center(8)

        print(
            "Player BLACK with X",
            f"{score_black}".center(4),
            current_move_text
            if game_state.current_player is go.BLACK
            else last_move_text,
            player1_desc,
        )
        print(
            "Player WHITE with O",
            f"{score_white}".center(4),
            current_move_text
            if game_state.current_player is go.WHITE
            else last_move_text,
            player2_desc,
        )

        if move_probs is not None:
            for i in range(game_state.size):
                for j in range(game_state.size):
                    print(
                        "{0:0.5f}".format(move_probs[i * game_state.size + j]), end=" "
                    )
                print()
            print("pass move: {0:0.5f}".format(move_probs[-1]))

        print("\r\n   ", end="")
        for x in range(size):
            print("{0:5}".format(x), end="")
        print("\r\n")
        for i in range(size):
            print(f"{i}".center(5), end="")
            for j in range(size):
                p = game_state.board[i][j]
                if p == go.BLACK:
                    if game_state.history[-1] == (i, j):
                        print("X".center(5), end="")
                    else:
                        print("x".center(5), end="")
                elif p == go.WHITE:
                    if game_state.history[-1] == (i, j):
                        print("O".center(5), end="")
                    else:
                        print("o".center(5), end="")
                else:
                    print("_".center(5), end="")
            print("\r\n")
        print("\r\n")

    def start_play(self, black_player, white_player, display=True):
        """start a game between two players"""

        black_player.reset_player()
        white_player.reset_player()
        self.init_game_state()

        players = {go.BLACK: black_player, go.WHITE: white_player}

        if display:
            self.print_board(self.game_state, str(black_player), str(white_player))

        while True:
            current_player_id = self.game_state.get_current_player()
            player_in_turn = players[current_player_id]
            move, move_probs = player_in_turn.get_action(self.game_state)

            if move == PASS_MOVE:
                move = PASS_MOVE
            else:
                move = (move // self.game_state.size, move % self.game_state.size)

            is_end_of_game = self.game_state.do_move(move)
            if display:
                self.print_board(
                    self.game_state, str(black_player), str(white_player), move_probs
                )
            if is_end_of_game:
                winner_id = self.game_state.get_winner()
                if display:
                    if winner_id is not None:
                        print("Game end. Winner is", players[winner_id])
                    else:
                        print("Game end. Tie")
                return winner_id

    def start_self_play(self, player, display=False):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        player.reset_player()
        self.init_game_state()

        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.game_state)

            if move == PASS_MOVE:
                move = PASS_MOVE
            else:
                move = (move // self.game_state.size, move % self.game_state.size)

            # store the data
            states.append(self.game_state.copy())
            mcts_probs.append(move_probs)
            current_players.append(self.game_state.current_player)

            # perform a move
            is_game_end = self.game_state.do_move(move)
            if display:
                self.print_board(self.game_state, str(player), str(player), move_probs)
            if is_game_end:
                winner_id = self.game_state.get_winner()
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner_id is not None:
                    winners_z[np.array(current_players) == winner_id] = 1.0
                    winners_z[np.array(current_players) != winner_id] = -1.0
                if display:
                    if winner_id is not None:
                        print(
                            "Game end. Winner is player:",
                            "X" if winner_id == go.BLACK else "O",
                        )
                    else:
                        print("Game end. Tie")
                return winner_id, zip(states, mcts_probs, winners_z)
