from Informer.exp.exp_informer import Exp_Informer as Exp
from config import args, setting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def predict():
    exp = Exp(args)

    exp.predict(setting, True)


def show_prediction():
    # the prediction will be saved in ./results/{setting}/real_prediction.npy

    prediction = np.load("./results/" + setting + "/real_prediction.npy").squeeze()
    true = pd.read_csv("./data/btcusdt_valid.csv")["Close"]

    length = min(len(true), len(prediction))

    plt.figure()
    plt.plot(true[:length], label="Truth")
    plt.plot(prediction[:length], label="Prediction")
    plt.legend()
    plt.show()


def show_test():
    # When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
    # The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

    preds = np.load("./results/" + setting + "/pred.npy")
    trues = np.load("./results/" + setting + "/true.npy")

    plt.figure()
    plt.plot(trues[0, :, -1], label="GroundTruth")
    plt.plot(preds[0, :, -1], label="Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    predict()
    show_prediction()
