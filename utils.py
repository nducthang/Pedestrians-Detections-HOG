from PIL import Image
import cv2
import os
import numpy as np
from hog import hog
import random
import yaml
from os import listdir
from os.path import isfile, join
import tqdm


def read_image_with_pillow(img_path, is_gray=True):
    pil_im = Image.open(img_path).convert('RGB')
    img = np.array(pil_im)
    img = img[:, :, ::-1].copy()  # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_neg_INRIAPerson_data(train_neg_lst, train_neg_dir, train_neg_num_patches_per_image=10, train_neg_patch_size_range=(0.4, 1.0)):
    print("Extract negative data from INRIA Person data...")
    # Read and extract negative samples (background)
    with open(train_neg_lst) as f:
        neg_lines = f.readlines()

    negative_features = []
    neg_lines = [os.path.join(
        train_neg_dir, '/'.join(pl.split('/')[1:])).strip() for pl in neg_lines]
    for idx, nline in tqdm.tqdm(enumerate(neg_lines), total=len(neg_lines)):
        img_path = nline
        if not os.path.isfile(img_path):
            print('[neg] skipped %s' % img_path)
            continue
        img = read_image_with_pillow(img_path, is_gray=True)
        img_h, img_w = img.shape
        img_min_size = min(img_h, img_w)

        # random ngẫu nhiên kích thước patch ảnh, chọn vị trí (x, y) ngẫu nhiên trên ảnh sau đó crop ra
        # random crop
        negative_patches = []
        for num_neg_idx in range(train_neg_num_patches_per_image):
            random_patch_size = random.uniform(
                train_neg_patch_size_range[0], train_neg_patch_size_range[1])
            random_patch_height = int(random_patch_size*img_min_size)
            random_patch_width = int(
                random_patch_height * random.uniform(0.3, 0.7))
            random_position_x = random.randint(0, img_w-random_patch_width)
            random_position_y = random.randint(0, img_h-random_patch_height)
            # crop image -> image patch
            npatch = img[random_position_y:random_position_y+random_patch_height,
                         random_position_x:random_position_x+random_patch_width]
            negative_patches.append(npatch)

        for npatch in negative_patches:
            img = cv2.resize(src=npatch, dsize=(64, 128))
            f = hog(img)
            negative_features.append(f)

    negative_features = np.array(negative_features)

    return negative_features


def load_pos_INRIAPerson_data(train_pos_lst, train_pos_dir):
    print("Extract positive data from INRIA Person data...")
    # read and extract positive samples (person)
    with open(train_pos_lst) as f:
        pos_lines = f.readlines()

    positive_features = []
    pos_lines = [os.path.join(
        train_pos_dir, '/'.join(pl.split('/')[1:])).strip() for pl in pos_lines]
    for idx, pline in tqdm.tqdm(enumerate(pos_lines), total=len(pos_lines)):
        img_path = pline
        if not os.path.isfile(img_path):
            print('[pos] Skipped %s' % img_path)
            continue

        img = read_image_with_pillow(img_path, is_gray=True)
        f = hog(img)
        positive_features.append(f)

    positive_features = np.array(positive_features)

    return positive_features


def svm_predict(feature_vector, svm_model):
    pred_y1 = svm_model.predict(np.array([feature_vector]))  # class [0] or [1]
    pred_y = svm_model.predict_proba(
        np.array([feature_vector]))  # proba [[0.01 0.99]]
    max_class, max_prob = max(enumerate(pred_y[0]))
    return max_class, max_prob


def load_pos_PennPed_data(annotation_path):
    print("Loading data from Penm Fudan Ped...")
    positive_features = []
    onlyfiles = [f for f in listdir(
        annotation_path) if isfile(join(annotation_path, f))]
    for file_txt in tqdm.tqdm(onlyfiles, total=len(onlyfiles)):
        with open(join(annotation_path, file_txt)) as f:
            documents = yaml.full_load(f)
        file_name = documents['Image filename']
        img = cv2.imread(join('./data/', file_name))
        for item, doc in documents.items():
            if "box" in item:
                coor = documents[item].split('-')
                coor = [c.strip().translate(str.maketrans('', '', '() '))
                        for c in coor]
                x1, y1 = map(int, coor[0].split(','))
                x2, y2 = map(int, coor[1].split(','))
                sub_img = img[y1:y2, x1:x2, :]
                f = hog(sub_img)
                positive_features.append(f)
    return np.array(positive_features)


if __name__ == '__main__':
    load_pos_PennPed_data('./data/PennFudanPed/Annotation')
