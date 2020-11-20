# -*- coding: utf-8 -*-
import sys 
sys.path.append('/workspace/hugo_py')

import os
import time
import cv2
import numpy as np
from darknet_fd import DarkNetFD
import traceback 

sys.path.append('/workspace/hugo_py')

#https://github.com/walletiger/jetson_nano_py/blob/master/camera.py
from camera import JetCamera
cap_w = 640
cap_h = 360
cap_fps = 10

emtions = ['面无表情', '高兴', '吃惊', '伤心', '生气', '厌恶', '害怕', '不屑一顾']

ft2 = cv2.freetype.createFreeType2()
ft2.loadFontData("DroidSansFallback.ttf", 0)

def detect_emotion(net, img, x1, y1, x2, y2):
    padding =  4

    if y1 < padding or x1 < padding:
        return  
    face = img[y1 - padding: y2 + padding, x1 - padding: x2 + padding ]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray, (64, 64))
    procesed_face = resized_face.reshape(1, 1, 64, 64)

    t0 = time.time()
    net.setInput(procesed_face)

    Output = net.forward()
    t1 = time.time()

    #print("emotion detect  cost = %.1fms \n" % (t1 - t0) * 1000)

    # Compute softmax values for each sets of scores
    expanded = np.exp(Output - np.max(Output))

    probablities = expanded / expanded.sum()

    prob = np.squeeze(probablities)
    predicted_emotion = emtions[prob.argmax()]
    # Write predicted emotion on image
    ft2.putText(img, predicted_emotion, (x1, y1 + (1*2)), fontHeight=25, color=(0, 0, 255), thickness=-1,
                     line_type=cv2.LINE_4, bottomLeftOrigin=False)
    #cv2.putText(img,'{}'.format(predicted_emotion),,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    # Draw a rectangular box on the detected face
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)


def main():
    cam = JetCamera(cap_w, cap_h, cap_fps)
    cam.open()

    model='onnx_models/emotion-ferplus-8.onnx'

    net = cv2.dnn.readNetFromONNX(model)

    
    fd = DarkNetFD()


    while True:
        ret, img = cam.read()

        if not ret:
            break 


        t0 = time.time()
        faces = fd.detect(img)
        t1 = time.time()

        #print("fd cost = %.1fms \n" % (t1 - t0) * 1000)

        #faces = fd.hog_detect(img)
        for ret in faces:
            r = ret[2]
            #print("ret = %s, %s" % (ret, r))
            x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
            try:
                detect_emotion(net, img, x1, y1, x2, y2)
            except:
                traceback.print_exc()
                break 

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0))


        cv2.imshow('haha', img)
        cv2.waitKey(1)

    cam.close()


if __name__ == '__main__':
    main()






