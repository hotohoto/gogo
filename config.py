import argparse


class Config:
    def __init__(self, width=6, height=6, model_file=None, game_n_row=4):
        self.width = width
        self.height = height
        self.model_file = model_file
        self.game_n_row = game_n_row

    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--size', type=int, default=6)
        parser.add_argument('--model', type=str)
        args = parser.parse_args()

        return Config(width=int(args.size), height=int(args.size), model_file=args.model)

    def get_current_model_name(self):
        return f"current_model_{self.width}_{self.height}_{self.game_n_row}.model"

    def get_best_model_name(self):
        return f"best_model_{self.width}_{self.height}_{self.game_n_row}.model"
