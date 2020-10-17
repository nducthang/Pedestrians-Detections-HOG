from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import cv2
import argparse
import imutils
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    IMG_PATH = 'b.png'
    image = cv2.imread(IMG_PATH)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(img = image, winStride = (4, 4), padding = (8, 8), scale = 1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs = None, overlapThresh=0.5)
    # 2. Bounding box với ảnh suppression
    # Khởi tạo plot
    ax2 = plt.subplot(1, 2, 2)
    # Vẽ bounding box cuối cùng trên ảnh
    for (xA, yA, xB, yB) in pick:
        # w = xB-xA
        # h = yB-yA
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # Hiển thị hình ảnh
        # plt.imshow(image)
        # plt.title('Ảnh sau non max suppression')
        # rectFig = patches.Rectangle((xA, yA),w,h,linewidth=1,edgecolor='r',facecolor='none')
        # ax2.add_patch(rectFig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)

