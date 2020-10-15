import numpy as np
import cv2
from numpy import linalg as LA


def hog(img, cell_size=8, block_size=2, bins=9):
    # Chuyển sang ảnh xám và resize ảnh
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lấy chiêù cao và chiêù rộng cuả ảnh
    h, w = img.shape  # 128, 64
    if h != 128 or w != 64:
        img = cv2.resize(src=img, dsize=(64, 128))
        h, w = img.shape  # 128, 64

    """ TÍNH GRADIENT CỦA ẢNH """
    x_kernel = np.array([[-1, 0, 1]])
    y_kernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, x_kernel)
    dy = cv2.filter2D(img, cv2.CV_32F, y_kernel)

    # Độ lớn và hướng gradient cuả toàn bộ ảnh
    maginutude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # radient
    orientation = np.degrees(orientation)  # -90 -> 90
    orientation += 90  # 0 -> 180

    """ VOTE VÀO HISTOGRAM """
    num_cell_x = w // cell_size # 8
    num_cell_y = h // cell_size # 16

    # hist_tensor Lưu lại vote của từng cell
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy*cell_size:cy*cell_size +
                              cell_size, cx*cell_size:cx*cell_size+cell_size]
            mag = maginutude[cy*cell_size:cy*cell_size +
                             cell_size, cx*cell_size:cx*cell_size+cell_size]

            hist, _ = np.histogram(ori, bins=bins, range=(
                0, 180), weights=mag)  # 1D vector, 9 elements
            hist_tensor[cy, cx, :] = hist

    # Chuẩn hoá theo block
    redundant_cell = block_size - 1  # số cell thừa
    feature_tensor = np.zeros(
        [num_cell_y-redundant_cell, num_cell_x-redundant_cell, block_size*block_size*bins])
    for bx in range(num_cell_x-redundant_cell):
        for by in range(num_cell_y-redundant_cell):
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()
            feature_tensor[by, bx, :] = v/LA.norm(v, 2)
            if np.isnan(feature_tensor[by, bx, :]).any():
                feature_tensor[by, bx, :] = v

    # Vector đặc trưng đã chuẩn hoá
    return feature_tensor.flatten() # 3780 feature
