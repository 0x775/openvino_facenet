#coding=utf-8

import face_recognition
import cv2
import numpy as np
import os,time

known_faces_dir = "faces"
known_face_encodings = []
known_face_names = []

for file in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, file))
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(file)[0])

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
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        print(index)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                #cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                points[str(index)] = {"left":left,"top":top,"right":right,"bottom":bottom,"name":name}
        #cv2.imshow('Video', frame)
        index+=1

        if index >200:
            break
        #print(points)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

def show_points():
    video_capture = cv2.VideoCapture('demo.mp4')
    index = 0
    while True:
        ret,frame = video_capture.read()
        if not ret:
            break
        if points.get(str(index)):
            point = points[str(index)]
            cv2.rectangle(frame, (point['left'],point['top']), (point['right'], point['bottom']), (0, 0, 255), 2)
            cv2.putText(frame, point['name'], (point['left'] + 6, point['bottom'] - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Video",frame)
        index+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

get_point()
elapsed_time = time.time() - start_time
print(f"The elapsed time is {elapsed_time} seconds.")


show_points()

video_capture.release()
cv2.destroyAllWindows()
