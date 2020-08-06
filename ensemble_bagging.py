import os
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn import linear_model
import sklearn
import pickle


def load_model(in_model_fp):
    if not os.path.exists(in_model_fp):
        print("Error: Can not find ensemble model.")
        return False
    obj_file = open(in_model_fp, "rb")
    bagging_model = pickle.load(obj_file)
    obj_file.close()
    return bagging_model


def predict_single_data(in_model, in_risk_clinical, in_risk_bmus, in_risk_cdfi):
    in_data = np.array([[float(in_risk_clinical), float(in_risk_bmus), float(in_risk_cdfi)]])
    y_red = in_model.predict(in_data)
    y_predict = y_red[0]
    return y_predict


def main():
    root_dir = os.path.abspath(".")
    fp_model = os.path.join(root_dir, "models", "model_ensemble.pkl")
    ensemble_model = load_model(fp_model)
    while True:
        stop_info = input("Is quit(Y/N):")
        if stop_info == 'Y':
            print("Predict ensemble is over!")
            break
        risk_values = input("Input the risk of Clinical, BMUS, CDFI:")
        risk_value_list = risk_values.strip().split()
        if len(risk_value_list) == 3:
            res_predcit = predict_single_data(ensemble_model, risk_value_list[0], risk_value_list[1],
                                              risk_value_list[2])
            if res_predcit < 0.5:
                res_clf = 0
            else:
                res_clf = 1
            print("Prediction is %s (%.2f) of Risk Clinical=%s, Risk BMUS=%s, Risk CDFI=%s." % (
            res_clf, res_predcit, risk_value_list[0], risk_value_list[1], risk_value_list[2]))
        else:
            print("Error: Must input 3 parameters")


if __name__ == '__main__':
    main()

