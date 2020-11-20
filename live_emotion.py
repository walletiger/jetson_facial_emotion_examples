#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys

sys.path.append('/workspace/hugo_py')

import cv2
import numpy as np
import time
from ctypes import *
import traceback 
from keras.models import model_from_json
from keras. preprocessing.image import img_to_array 
from darknet_fd import DarkNetFD

#https://github.com/walletiger/jetson_nano_py/blob/master/camera.py
from camera import JetCamera

cap_w = 640
cap_h = 360
cap_fps = 10


def emotion_detect(model, img, x1, y1, x2, y2):

    try:
        detected_face = img[y1:y2, x1:x2] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        img_pixels = img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
    except:
        return 


    img_pixels /= 255

    predictions = model.predict(img_pixels)

    #find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotion = emotions[max_index]

    cv2.putText(img, emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)



def main():
    model = model_from_json(open("keras_models/facial_expression_model_structure.json", "r").read())
    model.load_weights('keras_models/facial_expression_model_weights.h5')

    cam = JetCamera(cap_w, cap_h, cap_fps)
    fd = DarkNetFD() 

    cam.open()

    cnt = 0
    while True:
        try:
            ret, frame = cam.read()
            #print("camera read one frame ")
            if not ret:
                break

            t0 = time.time()
            res = fd.detect(frame)
            t1 = time.time()

            cnt += 1


            for ret in res:
                r = ret[2]
                #print("ret = %s, %s" % (ret, r))
                x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
                emotion_detect(model, frame, x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0))
            t2 = time.time()

            if cnt % 10 == 0:
                print("frame cnt [%d] yoloface detect delay = %.1fms" % (cnt, (t1 - t0) * 1000))
                print("emotion detect delay = %.1fms" %(t2 - t1))

            cv2.imshow('haha', frame)
            cv2.waitKey(1)
        except:
            traceback.print_exc()
            break 

    cam.close()


if __name__ == '__main__':
    main()
