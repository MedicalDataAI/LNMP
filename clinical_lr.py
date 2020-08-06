import os
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn import linear_model
import sklearn
import pickle

def load_model(in_model_fp=r"./models/model_clinical.pkl"):
    if not os.path.exists(in_model_fp):
        print("Error: Can not find Clinical model.")
        return False
    obj_file = open(in_model_fp, "rb")
    ml_model = pickle.load(obj_file)
    obj_file.close()
    return ml_model


def predict_single_data(in_model, in_sex, in_age, in_maxsize):
    in_data = np.array([[in_sex, in_age, in_maxsize]])
    y_pred = in_model.predict_proba(in_data)
    y_predict = y_pred[0, 1]
    return y_predict


def predict_from_csv(in_model, in_csv_fp, in_sex_name='Sex', in_age_name='Age', in_maxsize_name="MaxSize",
                     out_predict_name='Risk_clinical', in_save_dir='./res'):
    if not os.path.exists(in_csv_fp):
        print("Error: Can not find the file including clinical data.")
        return False
    df_data = pd.read_csv(in_csv_fp, encoding='utf-8')
    set_dict_config = dict()
    set_dict_config[in_sex_name] = int
    set_dict_config[in_age_name] = int
    set_dict_config[in_maxsize_name] = float
    df_data = df_data.astype(set_dict_config)
    y_predict = []
    y_predict_clf = []
    pre_cnt = 1
    cnt_all = len(df_data[in_sex_name].to_list())
    for sex, age, maxsize in zip(df_data[in_sex_name].to_list(), df_data[in_age_name].to_list(), df_data[in_maxsize_name].to_list()):
        tmp_predict = predict_single_data(in_model, sex, age, maxsize)
        y_predict.append(tmp_predict)
        if tmp_predict < 0.5:
            res_clf = 0
        else:
            res_clf = 1
        y_predict_clf.append(res_clf)
        print("[%d/%d]: Prediction is %s (%.2f) of Sex=%s, Age=%s, MaxSize=%s." % (
            pre_cnt, cnt_all, res_clf, tmp_predict, sex, age, maxsize))
        pre_cnt = pre_cnt + 1

    df_data[out_predict_name] = y_predict
    df_data['predict_Label'] = y_predict_clf
    save_fp = os.path.join(in_save_dir, "result_clinical.csv")
    df_data.to_csv(save_fp, index=False, encoding='utf-8')


if __name__ == '__main__':
    root_dir = os.path.abspath(".")
    fp_model = os.path.join(root_dir, "models", "model_clinical.pkl")
    fp_clinical = os.path.join(root_dir, "data", "clinical.csv")
    save_dir = os.path.join(root_dir, "res")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    clinical_model = load_model(fp_model)
    predict_from_csv(
        in_model=clinical_model,
        in_csv_fp=fp_clinical,
        in_save_dir=save_dir)
