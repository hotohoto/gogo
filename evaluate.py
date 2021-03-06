# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import defaultdict

import go
from config import Config
from game import Game
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPurePlayer
from policy_value_net_pytorch import PolicyValueNet


def get_pure_mcts_player():
    pure_mcts_player = MCTSPurePlayer()
    return pure_mcts_player


def evaluate_policy(game, my_player, opponent_player=None, n_games=3):
    """
    Evaluate the trained policy by playing against another player
    Note: this is only for monitoring the progress of training
    """

    game.init_game_state()

    if opponent_player is None:
        opponent_player = get_pure_mcts_player()

    win_cnt = defaultdict(int)
    for _ in range(n_games):
        winner = game.start_play(my_player, opponent_player, display=True)
        print(winner)
        win_cnt[winner] += 1
    win_ratio = 1.0 * (win_cnt[go.BLACK] + 0.5 * win_cnt[None]) / n_games
    print(f"win: {win_cnt[go.BLACK]}, lose: {win_cnt[go.WHITE]}, draw: {win_cnt[None]}")
    return win_ratio


def main():
    config = Config.from_args()
    _game = Game.from_config(config)
    policy_value_net = PolicyValueNet(config.size, model_file=config.model_file)
    mcts_player = MCTSPlayer(
        policy_value_net, c_puct=config.c_puct, n_playout=config.n_playout,
    )
    evaluate_policy(_game, mcts_player)


if __name__ == "__main__":
    main()
