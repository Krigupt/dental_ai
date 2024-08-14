import pandas as pd
import cv2
from ultralytics import YOLO

import tensorflow as tf
import keras

import numpy as np

def run_model(filename):
    has_disease = False

    model = YOLO('weights/best-3.pt')

    img = cv2.imread(filename)

    res = model.predict(img, save=True, save_txt=True, project='runs', name=f'photo')
    fp = res[0].save_dir

    with open(f'{fp}/labels/image0.txt', 'r') as f:
        s = f.read()

    x = s.split('\n')
    x.pop()

    height, width, _ = img.shape

    model = keras.layers.TFSMLayer("modelk3", call_endpoint='serve')

    for line in x:
        words = line.split(' ')
        xc, yc, w, h = int(float(words[1]) * width), int(float(words[2]) * height), int(float(words[3]) * width), int(float(words[4]) * height)
        x_min = max(xc - (w // 2), 0)
        y_min = max(yc - (h // 2), 0)
        x_max = min(xc + (w // 2), width)
        y_max = min(yc + (h // 2), height)

        crop_img = img[y_min:y_max, x_min:x_max]
        
        cv2.imwrite('buffer.jpg', crop_img)
        tooth_tensor = tf.io.read_file('buffer.jpg')
        tooth_tensor = tf.image.decode_image(tooth_tensor, channels=3)
        tooth_tensor = tf.image.resize(tooth_tensor, size=[224, 224])
        tooth_tensor = tf.cast(tooth_tensor, tf.float32)
        tooth_tensor = tf.expand_dims(tooth_tensor, axis=0)
        tooth_tensor=tooth_tensor/255.

        class_names = ['CROWN', 'Cavity', 'FILLING', 'IMPACTED', 'IMPLANT', 'ROOT CANAL']
        res = np.array(model(tooth_tensor))[0]
        pred = res.argmax()

        abbr = {'CROWN': 'CR', 'Cavity': 'CA', 'FILLING': 'F', 'IMPACTED': 'IM', 'IMPLANT': 'IL', 'ROOT CANAL': 'R'}
        
        if(res[pred] > 0.9):
            has_disease = True
            pred_class = class_names[pred]
            
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
            cv2.rectangle(img, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), -1)
            cv2.putText(img, abbr[pred_class], (xc - 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 0, 0), 1)

    cv2.imwrite('static/output/output.jpg', img)

    return (has_disease, f'{fp}')