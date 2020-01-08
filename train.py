# -*- coding: utf-8 -*-

from __future__ import print_function

import random
from collections import deque

import numpy as np

import evaluate
from config import Config
from game import Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet


class TrainPipeline:
    def __init__(self, config: Config):
        # params of the board and the game
        self.config = config
        self.game = Game.from_config(config)
        # training params
        self.temp = 1.0  # the temperature param
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50  # 50
        self.game_batch_num = 1500  # 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.policy_value_net = PolicyValueNet(
            self.config.width, self.config.height, model_file=config.model_file
        )
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=config.c_puct,
            n_playout=config.n_playout,
            is_selfplay=True,
        )

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in range(4):
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(mcts_prob.reshape(self.config.height, self.config.width)),
                    i,
                )
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
                )
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
                )
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        episodes = 0
        for _ in range(n_games):
            _, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            episodes += len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
        return episodes

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

        explained_var_old = 1 - np.var(
            np.array(winner_batch) - old_v.flatten()
        ) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(
            np.array(winner_batch) - new_v.flatten()
        ) / np.var(np.array(winner_batch))
        print(
            (
                "kl:{:.5f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
            ).format(
                kl, loss, entropy, explained_var_old, explained_var_new,
            )
        )
        return loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                episodes = self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episodes:{}".format(i, episodes))
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    current_mcts_player = MCTSPlayer(
                        self.policy_value_net.policy_value_fn,
                        c_puct=self.config.c_puct,
                        n_playout=self.config.n_playout,
                    )
                    win_ratio = evaluate.evaluate_policy(self.game, current_mcts_player)
                    self.policy_value_net.save_model(
                        self.config.get_current_model_name()
                    )
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(
                            self.config.get_best_model_name()
                        )
        except KeyboardInterrupt:
            print("\n\rquit")


def main():
    training_pipeline = TrainPipeline(Config.from_args())
    training_pipeline.run()


if __name__ == "__main__":
    main()
