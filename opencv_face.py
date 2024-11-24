#coding=utf-8
"""
xml数据集
    https://github.com/opencv/opencv/tree/master/data/haarcascades
"""

import cv2
import numpy as np
 
photos = list()  # 样本图像列表
lables = list()  # 标签列表
photos.append(cv2.imread(r"./faces/face1.jpg", 0))  # 记录第1张人脸图像
lables.append(0)  # 第1张图像对应的标签
 
names = {"0": "LXE"}  # 标签对应的名称字典
 
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建LBPH识别器
recognizer.train(photos, np.array(lables))  # 识别器开始训练
 
#  i = cv2.imread(r"./faces/face1.jpg", 0)  # 待识别的人脸图像
#  label, confidence = recognizer.predict(i)  # 识别器开始分析人脸图像
#  print("confidence = " + str(confidence))  # 打印评分
#  print(names[str(label)])  # 数组字典里标签对应的名字
 
#  cv2.waitKey() 
#  cv2.destroyAllWindows() 


#  exit(1)


#  import face_recognition
#  import cv2
#  import numpy as np
import os,time

#video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture('demo.mp4')
print("==============================")
print("width:   ",video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height:   ",video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:   ",video_capture.get(cv2.CAP_PROP_FPS))
print("frame_count:   ",video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("==============================")


points={}
start_time = time.time()

def get_point():
    index=0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        #rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        rgb_frame = frame[:, :, ::-1]
        gary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(index)
        flag = False
        
        faces=haar_cascade.detectMultiScale(gary,1.3,5)
        for (x,y,w,h) in faces:
            label, confidence = recognizer.predict(gary)  # 识别器开始分析人脸图像
            #0 表示完全匹配。通常情况下， 认为小于 50 的值是可以接受的，如果该值大于 80 则认 为差别较大
            print("confidence = " + str(confidence))  # 打印评分
            #print(names[str(label)])
            if int(confidence) < 80 :
                flag = True
                cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 0, 255), 2)
                cv2.putText(frame, names[str(label)], (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        index+=1

        #if flag:
        #    cv2.waitKey(3000)
        if index >20000000:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

get_point()
elapsed_time = time.time() - start_time
print(f"The elapsed time is {elapsed_time} seconds.")

video_capture.release()
cv2.destroyAllWindows()
