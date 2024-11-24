#coding=utf-8
"""
conda 
    python==3.8
    opencv-python=4.5.4.58
    opencv-contrib-python==4.5.4.58
    openvino==2021.4.1
    numpy==1.19.5   ***
"""
import cv2
from openvino.inference_engine import IECore
import numpy as np
from timeit import default_timer as timer
 
# ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
ie = IECore()
device = "CPU"
# ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
model_xml = "data/face-detection-0200.xml"
model_bin = "data/face-detection-0200.bin"
net = ie.read_network(model=model_xml)
# ---------------------------Step 3. Configure input & output----------------------------------------------------------
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[input_blob].shape
print("outputs's shape = ", net.outputs[output_blob].shape)
 
src = cv2.imread("imgs/22.jpg")
#src_ = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
image = cv2.resize(src, (w, h))
image = image.transpose(2, 0, 1)
# ---------------------------Step 4. Loading model to the device-------------------------------------------------------
exec_net = ie.load_network(network=net, device_name=device)
# ---------------------------Step 5. Create infer request--------------------------------------------------------------
# ---------------------------Step 6. Prepare input---------------------------------------------------------------------
# ---------------------------Step 7. Do inference----------------------------------------------------------------------
tic = timer()
res = exec_net.infer(inputs={input_blob: [image]})
toc = timer()
print("the cost time is(ms): ", 1000*(toc - tic))
print("the latance is:", exec_net.requests[0].latency)
# ---------------------------Step 8. Process output--------------------------------------------------------------------
res = res[output_blob]
dets = res.reshape(-1, 7)
sh, sw, _ = src.shape
for det in dets:
    conf = det[2]
    if conf > 0.5:
        # calss_id...
        xmin = int(det[3] * sw)
        ymin = int(det[4] * sh)
        xmax = int(det[5] * sw)
        ymax = int(det[6] * sh)
        cv2.putText(src, str(round(conf, 3)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, 7)
        cv2.rectangle(src, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
cv2.imshow("src", src)
cv2.waitKey(0)


#test.video
import os,time

video_capture = cv2.VideoCapture('demo.mp4')
print("==============================")
print("width:   ",video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height:   ",video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:   ",video_capture.get(cv2.CAP_PROP_FPS))
print("frame_count:   ",video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("==============================")

tic = timer()
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    image = cv2.resize(frame, (w, h))
    image = image.transpose(2, 0, 1)
    
    res = exec_net.infer(inputs={input_blob: [image]})
    #toc = timer()
    #print("the cost time is(ms): ", 1000*(toc - tic))
    #print("the latance is:", exec_net.requests[0].latency)
    # ---------------------------Step 8. Process output--------------------------------------------------------------------
    res = res[output_blob]
    dets = res.reshape(-1, 7)
    sh, sw, _ = frame.shape
    for det in dets:
        conf = det[2]
        if conf > 0.5:
            # calss_id...
            xmin = int(det[3] * sw)
            ymin = int(det[4] * sh)
            xmax = int(det[5] * sw)
            ymax = int(det[6] * sh)
            cv2.putText(frame, str(round(conf, 3)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, 7)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.imshow("src", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

toc = timer()
print("the cost time is(ms): ", 1000*(toc - tic))
print("time is(s): ", int(1000*(toc - tic))/1000)
print("time is(min): ", int(1000*(toc - tic))/1000/60)
cv2.destroyAllWindows()
