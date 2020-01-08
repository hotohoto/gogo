# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from config import Config


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get("width", 8))
        self.height = int(kwargs.get("height", 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get("n_in_row", 5))
        self.players = [0, 1]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception(
                "board width and height can not be "
                "less than {}".format(self.n_in_row)
            )
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = None

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        y = move // self.width
        x = move % self.width
        return [y, x]

    def location_to_move(self, location):
        if len(location) != 2:
            return None
        y = location[0]
        x = location[1]
        move = y * self.width + x
        if move not in range(self.width * self.height):
            return None
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][
                self.last_move // self.width, self.last_move % self.height
            ] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0]
            if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def get_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < n * 2 - 1:
            return None

        for m in moved:
            y = m // width
            x = m % width
            player_id = states[m]

            if (
                x in range(width - n + 1)
                and len(set(states.get(i) for i in range(m, m + n))) == 1
            ):
                return player_id

            if (
                y in range(height - n + 1)
                and len(set(states.get(i) for i in range(m, m + n * width, width))) == 1
            ):
                return player_id

            if (
                x in range(width - n + 1)
                and y in range(height - n + 1)
                and len(
                    set(states.get(i) for i in range(m, m + n * (width + 1), width + 1))
                )
                == 1
            ):
                return player_id

            if (
                x in range(n - 1, width)
                and y in range(height - n + 1)
                and len(
                    set(states.get(i) for i in range(m, m + n * (width - 1), width - 1))
                )
                == 1
            ):
                return player_id

        return None

    def game_end(self):
        """Check whether the game is ended or not and who is the winner"""
        winner_id = self.get_winner()
        if winner_id is not None:
            return True, winner_id
        elif len(self.availables) == 0:
            return True, None
        return False, None

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board):
        self.board = board

    @staticmethod
    def from_config(config: Config):
        return Game(
            Board(width=config.width, height=config.height, n_in_row=config.game_n_row)
        )

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1.id, "with X".rjust(3), player1)
        print("Player", player2.id, "with O".rjust(3), player2)
        print()
        for x in range(width):
            print("{0:8}".format(x), end="")
        print("\r\n")
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end="")
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc)
                if p == player1.id:
                    print("X".center(8), end="")
                elif p == player2.id:
                    print("O".center(8), end="")
                else:
                    print("_".center(8), end="")
            print("\r\n\r\n")

    def start_play(self, player1, player2, start_player=0, display=True):
        """start a game between two players"""
        assert start_player in (0, 1)

        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_id(p1)
        player2.set_id(p2)
        players = {p1: player1, p2: player2}
        if display:
            self.graphic(self.board, player1, player2)
        while True:
            current_player_id = self.board.get_current_player()
            player_in_turn = players[current_player_id]
            move, _ = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if display:
                self.graphic(self.board, player1, player2)
            end, winner_id = self.board.game_end()
            if end:
                if display:
                    if winner_id is not None:
                        print("Game end. Winner is", players[winner_id])
                    else:
                        print("Game end. Tie")
                return winner_id

    def start_self_play(self, player, temp, display=False):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, temp=temp)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if display:
                self.graphic(self.board, p1, p2)
            end, winner_id = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner_id is not None:
                    winners_z[np.array(current_players) == winner_id] = 1.0
                    winners_z[np.array(current_players) != winner_id] = -1.0
                # reset MCTS root node
                player.reset_player()
                if display:
                    if winner_id is not None:
                        print("Game end. Winner is player:", winner_id)
                    else:
                        print("Game end. Tie")
                return winner_id, zip(states, mcts_probs, winners_z)
