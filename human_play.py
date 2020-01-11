# -*- coding: utf-8 -*-
from __future__ import print_function

from config import Config
from constants import PASS_MOVE
from game import Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


class Human(object):
    """
    human player
    """

    def get_action(self, game_state):

        str_input = None
        while True:
            str_input = input("Your move: ")
            if str_input == "":
                return PASS_MOVE, None
            try:
                move = str_input.split(",")
                move = (int(move[0]), int(move[1]))
            except Exception:  # pylint: disable=broad-except
                print("invalid input")
                continue

            if game_state.is_legal(move):
                move = int(move[0]) * game_state.size + int(move[1])
                print(move)
                return move, None

    def reset_player(self):
        pass

    def __str__(self):
        return "Human"


def main(config):
    try:
        game = Game.from_config(config)

        # ############### human VS AI ###################
        # load the trained policy_value_net in PyTorch

        best_policy = PolicyValueNet(config.size, model_file=config.model_file)
        mcts_player = MCTSPlayer(
            best_policy, c_puct=config.c_puct, n_playout=config.n_playout,
        )

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, display=1)
    except KeyboardInterrupt:
        print("\n\rquit")


if __name__ == "__main__":
    main(Config.from_args())
