import os
import argparse
from pprint import pprint
import json

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument('--test', action='store_true', help='codetest')
        self.parser.add_argument('--history_size', type=int, default=10, help='past frame number')
        self.parser.add_argument('--progress_weight', type=float, default=1.0, help='progress weight')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
        self.parser.add_argument('--checkpoint_name', type=str, required=True, help='Name of the checkpoint file')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        if self.opt.config:
            with open(self.opt.config, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self.opt, key, value)

        if not os.path.isdir(self.opt.ckpt):
            os.makedirs(self.opt.ckpt)

        self._print()
        return self.opt
