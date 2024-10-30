import sys

if not "Informer" in sys.path:
    sys.path += ["Informer"]

from Informer.exp.exp_informer import Exp_Informer as Exp
import torch
from config import args, setting
import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--train_epochs", type=int, default=6)
arg_parser.add_argument("--lradj", type=str, default="type1")
arg_parser.add_argument("--patience", type=int, default=20)

parsed_args = arg_parser.parse_args()

for key, value in vars(parsed_args).items():
    setattr(args, key, value)

print("Args in experiment:")
print(args)

# set experiments
exp = Exp(args)

# train
print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
exp.train(setting)

# test
print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
exp.test(setting)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.mps.is_available():
    torch.mps.empty_cache()
