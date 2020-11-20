#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys
sys.path.append('/workspace/hugo_py')

import cv2
import numpy as np
import time
from darknet_fd import DarkNetFD
import traceback 
from facial_emotion_recognition import EmotionRecognition

#https://github.com/walletiger/jetson_nano_py/blob/master/camera.py
from camera import JetCamera
cap_w = 640
cap_h = 360
cap_fps = 20


def emotion_detect(model, img, x1, y1, x2, y2):
    try:
        detected_face = img[y1 - 4: y2 + 4, x1 - 4: x2 + 4] #crop detected face
        model.recognise_emotion(detected_face, return_type='BGR')
        img[y1 - 4: y2 + 4, x1 - 4: x2 + 4] = detected_face #crop detected face
    except:
        traceback.print_exc()
        return 



def main():
    model = EmotionRecognition(device='gpu', gpu_id=0)
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

            #if cnt % 1 == 0:
            #    print("frame cnt [%d] yoloface detect delay = %.1fms" % (cnt, (t1 - t0) * 1000))

            for ret in res:
                r = ret[2]
                #print("ret = %s, %s" % (ret, r))
                x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                emotion_detect(model, frame, x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0))
            t2 = time.time()

            #print("emotion detect delay = %.1fms" %(t2 - t1) * 1000)

            cv2.imshow('haha', frame)
            cv2.waitKey(1)
        except:
            traceback.print_exc()
            break 

    cam.close()


if __name__ == '__main__':
    main()
