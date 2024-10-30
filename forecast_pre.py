import sys

if not "Informer" in sys.path:
    sys.path += ["Informer"]

from Informer.exp.exp_informer import Exp_Informer as Exp
from config import args, setting
import numpy as np
import matplotlib.pyplot as plt

# If you already have a trained model, you can set the arguments and model path, then initialize a Experiment and use it to predict
# Prediction is a sequence which is adjacent to the last date of the data, and does not exist in the data
# If you want to get more information about prediction, you can refer to code `exp/exp_informer.py function predict()` and `data/data_loader.py class Dataset_Pred`

exp = Exp(args)

exp.predict(setting, True)

# the prediction will be saved in ./results/{setting}/real_prediction.npy
prediction = np.load("./results/" + setting + "/real_prediction.npy")

print(prediction)

plt.figure()
plt.plot(prediction[0, :, -1])
plt.show()

# When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
# The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

# preds = np.load("./results/" + setting + "/pred.npy")
# trues = np.load("./results/" + setting + "/true.npy")

# plt.figure()
# plt.plot(trues[0, :, -1], label="GroundTruth")
# plt.plot(preds[0, :, -1], label="Prediction")
# plt.legend()
# plt.show()

# [samples, pred_len, dimensions]
# preds.shape, trues.shape
