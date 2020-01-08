# -*- coding: utf-8 -*-
from __future__ import print_function

from config import Config
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.id = None

    def set_id(self, id):
        self.id = id

    def get_action(self, board):
        try:
            location = input("Your move: ")
            location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception:  # pylint: disable=broad-except
            move = None
        if move is None or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move, None

    def __str__(self):
        return "Human {}".format(self.id)


def main(config):
    try:
        board = Board(
            width=config.width, height=config.height, n_in_row=config.game_n_row
        )
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in PyTorch

        best_policy = PolicyValueNet(
            config.width, config.height, model_file=config.model_file
        )
        mcts_player = MCTSPlayer(
            best_policy.policy_value_fn,
            c_puct=config.c_puct,
            n_playout=config.n_playout,
        )

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=0, display=1)
    except KeyboardInterrupt:
        print("\n\rquit")


if __name__ == "__main__":
    main(Config.from_args())
