from hog import hog
import utils
from nms import non_max_suppression_fast
import numpy as np
import cv2
import time
import joblib
import os

MODEL_PATH = 'model_hog_person.joblib'
IMG_PATH = 'viet_han.jpg'
IMG_LST_HEIGHT = [140, 210, 315, 470]
WINDOW_WIDTH = 64
WINDOW_HEIGHT = 128
WINDOW_STEP = 16
PROB_THRESHOLD = 0.9

if __name__ == '__main__':
    svm_model = joblib.load(MODEL_PATH)

    # Load image & extract HOG feature
    d_img = cv2.imread(IMG_PATH)  # ảnh màu
    img = utils.read_image_with_pillow(
        img_path=IMG_PATH, is_gray=True)  # ảnh xám
    h, w = img.shape[:2]

    n_windown = 0
    time_start = time.time()
    boxes = []

    for idx, new_height in enumerate(IMG_LST_HEIGHT):
        new_width = int(new_height/h*w)
        if WINDOW_WIDTH > new_width or WINDOW_HEIGHT > new_height:
            continue

        new_img = cv2.resize(src=img, dsize=(new_width, new_height))
        max_x = new_width - WINDOW_WIDTH
        max_y = new_height - WINDOW_HEIGHT

        print('Scale (h=%d, w=%d)' % (new_height, new_width))

        for x in range(0, max_x + 1, WINDOW_STEP):
            for y in range(0, max_y + 1, WINDOW_STEP):
                n_windown += 1
                patch = new_img[y:y+WINDOW_HEIGHT, x:x+WINDOW_WIDTH]
                f = hog(patch)
                is_person, prob = utils.svm_predict(f, svm_model)
                if is_person and prob > PROB_THRESHOLD:
                    x1 = int(x/new_width*w)
                    y1 = int(y/new_height*h)
                    x2 = int((x+WINDOW_WIDTH)/new_width * w)
                    y2 = int((y+WINDOW_HEIGHT)/new_height*h)
                    boxes.append([x1, y1, x2, y2])

    pboxes = non_max_suppression_fast(np.array(boxes))

    for box in pboxes:
        cv2.rectangle(d_img, (box[0], box[1]),
                      (box[2], box[3]), (0, 255, 0), 2)

    time_end = time.time()

    print("Processed %d windowns in %.2f seconds" %
          (n_windown, time_end-time_start))

    cv2.imshow('test', d_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
