from __future__ import print_function
# to filter some unnecessory warning messages
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os
import numpy as np
import pandas as pd
import glob
import keras
from keras import backend as K
import random as rn
import cv2
import shutil


def load_model(in_model_fp=r"./models/model_cdfi.h5"):
    if not os.path.exists(in_model_fp):
        print("Error: Can not find CDFI model.")
        return False
    cnn_model = keras.models.load_model(in_model_fp)
    cnn_model.summary()
    return cnn_model

def predict_single_img(in_model, in_img_fp, in_img_size, in_pre_fun):
    img = keras.preprocessing.image.load_img(in_img_fp, target_size=(in_img_size[0], in_img_size[1]))
    img_np = keras.preprocessing.image.img_to_array(img)
    img_np = np.squeeze(img_np)
    x = np.expand_dims(img_np, axis=0)
    x = in_pre_fun(x)
    preds = in_model.predict(x)
    return preds


def predict_imgs_from_dir(
        in_model, in_src_dir, in_img_size=[224, 224, 3], img_type='bmp',
        in_save_dir=r"./res"
):
    fuzzy_fp = os.path.join(in_src_dir, '*.%s' % img_type)
    print(fuzzy_fp)
    img_fps = glob.glob(fuzzy_fp)
    if 0 == len(img_fps):
        print("Error: Can not find images with %s type." % img_type)
        return False
    cnn_model = in_model
    cnn_model_preprocess = keras.applications.resnet50.preprocess_input
    cnn_model.summary()

    y_predict = predict_single_img(cnn_model, img_fps[0], in_img_size, cnn_model_preprocess)
    file_id = []
    y_predict_class = []
    img_cnt = 1
    for in_fp in img_fps:
        tmp_pre = predict_single_img(cnn_model, in_fp, in_img_size, cnn_model_preprocess)
        if tmp_pre[0] < 0.5:
            tmp_clf = 0
        else:
            tmp_clf = 1
        y_predict = np.vstack((y_predict, tmp_pre))
        print("Info [%d / %d]: Prediction is %s (%.2f) of %s." % (img_cnt, len(img_fps), str(tmp_clf), tmp_pre[0], os.path.basename(in_fp)))
        img_cnt = img_cnt + 1
        y_predict_class.append(tmp_clf)
        file_id.append(os.path.basename(in_fp))
    y_predict = np.delete(y_predict, 0, axis=0)

    save_dict = dict()
    save_dict['ID'] = file_id
    for predict_index in range(y_predict.shape[1]):
        save_dict['Rsik_cdfi'] = y_predict[:, predict_index].tolist()
    save_dict['predict_Label'] = y_predict_class
    save_df = pd.DataFrame(save_dict)
    save_fp = os.path.join(in_save_dir, "result_cdfi.csv")
    save_df.to_csv(save_fp, index=False, encoding='utf-8')


if __name__ == '__main__':
    img_type = "png"
    root_dir = os.path.abspath(".")
    fp_cnn_model = os.path.join(root_dir, "models", "model_cdfi.h5")
    cdfi_dir = os.path.join(root_dir, "data", "cdfi")
    save_dir = os.path.join(root_dir, "res")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    cnn_model = load_model(fp_cnn_model)
    predict_imgs_from_dir(
        in_model=cnn_model,
        in_src_dir=cdfi_dir,
        img_type=img_type,
        in_save_dir=save_dir)


