# -*- coding: utf-8 -*-

from __future__ import print_function

import itertools
import random
from collections import deque

import numpy as np

import evaluate
from config import Config
from game import Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet, get_current_input


class TrainPipeline:
    def __init__(self, config: Config):
        # params of the game

        self.config = config
        self.game = Game.from_config(config)

        # training params

        self.buffer_size = 10000
        self.min_data_to_collect = 128
        self.batch_size = 128  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.save_freq = 20
        self.eval_freq = self.save_freq * 100
        self.num_total_iter = self.eval_freq * 4
        assert self.num_total_iter % self.save_freq == 0
        assert self.num_total_iter % self.eval_freq == 0

        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.policy_value_net = PolicyValueNet(
            self.config.size, model_file=config.model_file
        )
        self.mcts_player = MCTSPlayer(
            self.policy_value_net,
            c_puct=config.c_puct,
            n_playout=config.n_playout,
            is_selfplay=True,
        )

    def collect_selfplay_data(self):
        """collect self-play data for training"""
        n_game = 0

        while not len(self.data_buffer) > self.min_data_to_collect:
            _, play_data = self.game.start_self_play(self.mcts_player, display=True)
            play_data = [(data[0], data[1], data[2]) for data in play_data]
            self.data_buffer.extend(play_data)
            n_game += 1

        return len(self.data_buffer), n_game

    def transform_data(self, board_input_batch, mcts_probs_batch):
        """rotate or flip the original data"""
        original_shape = (board_input_batch.shape, mcts_probs_batch.shape)

        for i in range(board_input_batch.shape[0]):
            n_rotate = np.random.randint(4)
            flip = np.random.randint(2)

            if not n_rotate or not flip:
                continue

            board_input = board_input_batch[i]
            board_size = board_input.shape[0]
            mcts_probs = mcts_probs_batch[i]
            mcts_place_probs = np.reshape(
                mcts_probs[0 : board_size * board_size], (board_size, board_size)
            )
            mcts_pass_move_prob = mcts_probs[-1]

            if n_rotate:
                board_input = np.rot90(board_input, n_rotate, axes=(1, 2))
                mcts_place_probs = np.rot90(mcts_place_probs, n_rotate)
            if flip:
                board_input = board_input.T
                mcts_place_probs = mcts_place_probs.T

            np.put(board_input_batch, i, board_input)
            np.put(
                mcts_probs_batch, i, np.append(mcts_place_probs, mcts_pass_move_prob)
            )

        transformed_shape = (board_input_batch.shape, mcts_probs_batch.shape)
        assert original_shape == transformed_shape
        return board_input_batch, mcts_probs

    def policy_update(self):
        """update the policy-value net"""
        random.shuffle(self.data_buffer)
        n_batchs = len(self.data_buffer) // self.batch_size

        for i in range(n_batchs):
            mini_batch = list(
                itertools.islice(
                    self.data_buffer, i * self.batch_size, (i + 1) * self.batch_size
                )
            )

            board_input_batch = np.array(
                [get_current_input(data[0]) for data in mini_batch]
            )
            mcts_probs_batch = np.array([data[1] for data in mini_batch])
            winner_batch = np.array([data[2] for data in mini_batch])
            board_input_batch, mcts_probs_batch = self.transform_data(
                board_input_batch, mcts_probs_batch
            )

            old_probs, old_v = self.policy_value_net.policy_value(board_input_batch)

            self.policy_value_net.set_train_mode()

            loss, entropy = self.policy_value_net.train_step(
                board_input_batch, mcts_probs_batch, winner_batch
            )
            new_probs, new_v = self.policy_value_net.policy_value(board_input_batch)
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )

            explained_var_old = 1 - np.var(
                np.array(winner_batch) - old_v.flatten()
            ) / np.var(np.array(winner_batch))
            explained_var_new = 1 - np.var(
                np.array(winner_batch) - new_v.flatten()
            ) / np.var(np.array(winner_batch))
            print(
                (
                    "batch:{}, "
                    "kl:{:.5f}, "
                    "loss:{:.5f}, "
                    "entropy:{:.5f}, "
                    "explained_var_old:{:.3f}, "
                    "explained_var_new:{:.3f}"
                ).format(
                    i, kl, loss, entropy, explained_var_old, explained_var_new,
                )
            )

    def run(self):
        """run the training pipeline"""
        np.random.seed(0)
        try:
            for i in range(self.num_total_iter):
                n_data, n_game = self.collect_selfplay_data()

                print(
                    "iteration {}: total {} data collected from {} game(s)".format(
                        i, n_data, n_game
                    )
                )

                self.policy_value_net.set_train_mode()
                self.policy_update()
                self.data_buffer.clear()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.save_freq == 0:
                    print(
                        "saving current model at {}: file={}".format(
                            i + 1, self.config.get_current_model_name()
                        )
                    )
                    self.policy_value_net.save_model(
                        self.config.get_current_model_name()
                    )
                if (i + 1) % self.eval_freq == 0:
                    print("evalutating current model: {}".format(i + 1))
                    current_mcts_player = MCTSPlayer(
                        self.policy_value_net,
                        c_puct=self.config.c_puct,
                        n_playout=self.config.n_playout,
                    )
                    win_ratio = evaluate.evaluate_policy(self.game, current_mcts_player)
                    if win_ratio > self.best_win_ratio:
                        print(
                            "saving the new best policy at {}! win_ratio={}, file={}".format(
                                i, win_ratio, self.config.get_best_model_name(),
                            )
                        )
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
