import argparse
from datetime import datetime


class Config:
    def __init__(
        self,
        width=6,
        height=6,
        game_n_row=4,
        c_puct=5,
        model_file=None,
        n_playout=400,  # set larger n_playout for better performance
    ):
        # game settings
        self.width = width
        self.height = height
        self.game_n_row = game_n_row

        # training time settings
        self.c_puct = c_puct
        self.model_file = model_file

        # inference time settings
        self.n_playout = n_playout

    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--size", type=int, default=6)
        parser.add_argument("--model", type=str)
        args = parser.parse_args()

        return Config(
            width=int(args.size), height=int(args.size), model_file=args.model
        )

    @staticmethod
    def get_datetime():
        return datetime.today().strftime("%y%m%d%H%M%S")

    def get_current_model_name(self):
        return f"current_model_{self.width}_{self.height}_{self.game_n_row}.model"

    def get_best_model_name(self):
        stamp = self.get_datetime()
        return f"best_model_{self.width}_{self.height}_{self.game_n_row}_{stamp}.model"
