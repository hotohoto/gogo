import argparse
from datetime import datetime

DEFAULT_SIZE = 7
DEFAULT_KOMI_MAP = {
    19: 7.5,
    13: 5.5,
    9: 5.5,
    7: 4.5,
    6: 3.5,
    5: 2.5,
    3: 0.5,
}
DEFAULT_ALPHA_ZERO_N_PLAYOUT = 10
DEFAULT_PURE_MCTS_N_PLAYOUT = 10


class Config:
    def __init__(
        self,
        size=DEFAULT_SIZE,
        komi=None,
        enforce_superko=False,
        c_puct=5,
        model_file=None,
        n_playout=DEFAULT_ALPHA_ZERO_N_PLAYOUT,  # set larger n_playout for better performance
    ):
        # game settings
        self.size = size
        self.komi = DEFAULT_KOMI_MAP[size] if komi is None else komi
        self.enforce_superko = enforce_superko

        # training time settings
        self.c_puct = c_puct
        self.model_file = model_file

        # inference time settings
        self.n_playout = n_playout

    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
        parser.add_argument("--model-file", type=str)
        parser.add_argument(
            "--n-playout", type=int, default=DEFAULT_ALPHA_ZERO_N_PLAYOUT
        )
        args = parser.parse_args()

        return Config(**vars(args))

    @staticmethod
    def get_datetime():
        return datetime.today().strftime("%Y%m%d%H%M%S")

    def get_current_model_name(self):
        return f"current_model_{self.size}.model"

    def get_best_model_name(self):
        stamp = self.get_datetime()
        return f"best_model_{self.size}_{stamp}.model"
