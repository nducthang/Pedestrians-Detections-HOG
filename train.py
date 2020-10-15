import utils
import hog
import nms
import cv2
import numpy as np
import joblib
from sklearn import svm

# TRAIN_POS_LST = './data/INRIAPerson/train_64x128_H96/pos.lst'
# TRAIN_POS_DIR = './data/INRIAPerson/96X160H96/Train'

TRAIN_NEG_LST = './data/INRIAPerson/train_64x128_H96/neg.lst'
TRAIN_NEG_DIR = './data/INRIAPerson/Train'

FILE_MODEL = 'model_hog_svm_3.joblib'

ANNOTATION_PENNFUDAN = './data/PennFudanPed/Annotation'
ANNOTATION_INRIA = './data/INRIAPerson/Train/annotations'
ANNOTATION_INRIA2 = './data/INRIAPerson/Test/annotations'

if __name__ == '__main__':
    # Load dữ liệu postive
    positive_features1 = utils.load_pos_INRIA_data(ANNOTATION_INRIA)
    positive_features2 = utils.load_pos_INRIA_data(ANNOTATION_INRIA2)
    positive_features3 = utils.load_pos_PennPed_data(ANNOTATION_PENNFUDAN)
    # Nối dữ liệu postive lại thành một vector (num_pos, num_feature)
    positive_features = np.concatenate(
        (positive_features1, positive_features2, positive_features3), axis=0)
    # Load dữ liệu negative, với mỗi ảnh negative, lấy ngẫu nhiên train_neg_num_patches_per_image box làm mẫu
    negative_features = utils.load_neg_INRIAPerson_data(
        train_neg_lst=TRAIN_NEG_LST, train_neg_dir=TRAIN_NEG_DIR, train_neg_num_patches_per_image=20)
    print('Our positive feature matrix: ',
          positive_features.shape)  # (2249, 3780)
    print('Our negative feature matrix: ',
          negative_features.shape)  # (12180, 3780)
    # Ghép lại để tạo dữ liệu chuẩn bị huấn luyện
    x = np.concatenate((negative_features, positive_features),
                       axis=0)  # (14429, 3730)
    y = np.array([0]*negative_features.shape[0] +
                 [1]*positive_features.shape[0])
    print('X: ', x.shape)
    print('Y: ', y.shape)
    print('Start training model with X & Y samples...')
    # Huấn luyện mô hình sử dụng SVM
    model = svm.SVC(C=0.01, kernel='rbf', probability=True)
    model = model.fit(x, y)
    print('Done training model!')
    # Lưu mô hình lại
    joblib.dump(model, FILE_MODEL)
    print("Trained model is saved %s" % FILE_MODEL)
