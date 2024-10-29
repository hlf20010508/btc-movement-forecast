import sys

if not "Informer" in sys.path:
    sys.path += ["Informer"]

from Informer.exp.exp_informer import Exp_Informer as Exp
import torch
from config import args, setting

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

torch.cuda.empty_cache()
