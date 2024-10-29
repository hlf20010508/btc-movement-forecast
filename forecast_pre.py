import sys

if not "Informer" in sys.path:
    sys.path += ["Informer"]

from Informer.exp.exp_informer import Exp_Informer as Exp
from config import args, setting
import numpy as np

# If you already have a trained model, you can set the arguments and model path, then initialize a Experiment and use it to predict
# Prediction is a sequence which is adjacent to the last date of the data, and does not exist in the data
# If you want to get more information about prediction, you can refer to code `exp/exp_informer.py function predict()` and `data/data_loader.py class Dataset_Pred`

exp = Exp(args)

exp.predict(setting, True)

# the prediction will be saved in ./results/{setting}/real_prediction.npy
prediction = np.load("./results/" + setting + "/real_prediction.npy")

print(prediction.shape)
