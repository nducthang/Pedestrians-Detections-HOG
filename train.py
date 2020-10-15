import utils
import hog
import nms
import cv2
import numpy as np
import joblib
from sklearn import svm

TRAIN_POS_LST = './data/INRIAPerson/train_64x128_H96/pos.lst'
TRAIN_POS_DIR = './data/INRIAPerson/96X160H96/Train'

TRAIN_NEG_LST = './data/INRIAPerson/train_64x128_H96/neg.lst'
TRAIN_NEG_DIR = './data/INRIAPerson/Train'

if __name__ == '__main__':
    positive_features, negative_features = utils.load_INRIAPerson_data(
        train_pos_lst=TRAIN_POS_LST, train_pos_dir=TRAIN_POS_DIR, train_neg_lst=TRAIN_NEG_LST, train_neg_dir=TRAIN_NEG_DIR)

    print('Our positive feature matrix: ',
          positive_features.shape)  # (2416, 3780)
    print('Our negative feature matrix: ',
          negative_features.shape)  # (12180, 3780)

    x = np.concatenate((negative_features, positive_features),
                       axis=0)  # (14596, 3730)
    y = np.array([0]*negative_features.shape[0] +
                 [1]*positive_features.shape[0])

    print('X: ', x.shape)
    print('Y: ', y.shape)
    print('Start training model with X & Y samples...')

    model = svm.SVC(C=0.01, kernel='rbf', probability=True)
    model = model.fit(x, y)

    print('Done training model!')

    out_model_name = 'model_hog_person.joblib'
    joblib.dump(svm_model, out_model_name)
    print("Trained model is saved %s" % out_model_name)