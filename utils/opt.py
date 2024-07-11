#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
#from utils import log
import sys


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--test', action='store_true', help='codetest')

        # ===============================================================
        #                     Model options


        # ===============================================================
        #                     Running options

        self.parser.add_argument('--history_size', type=int, default=10, help='past frame number')
        self.parser.add_argument('--progress_weight', type=float, default=1.0, help='progress weight')
        # ===============================================================



        # ===============================================================
        #                     Augumentation
        # ===============================================================




    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        # if not self.opt.is_eval:
        #     script_name = os.path.basename(sys.argv[0])[:-3]
        #     log_name = '{}_in{}_out{}_ks{}_dctn{}_ds{}'.format(script_name, self.opt.input_n,
        #                                                   self.opt.output_n,
        #                                                   self.opt.kernel_size,
        #                                                   self.opt.dct_n, self.opt.dataset)
        #     self.opt.exp = log_name
        #     # do some pre-check
        #     ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        #     if not os.path.isdir(ckpt):
        #         os.makedirs(ckpt)
        #         #log.save_options(self.opt)
        #     self.opt.ckpt = ckpt
        #     #log.save_options(self.opt)
        # self._print()
        # log.save_options(self.opt)
        return self.opt
