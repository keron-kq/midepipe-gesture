import pyvidu as vidu
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

#初始化mdeiapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# 初始化 TOF 相机设备
intrinsic = vidu.intrinsics()
extrinsic = vidu.extrinsics()

device = vidu.PDdevice()
if not device.init():
    print("device init failed")
    exit(-1)

print("device init succeed  ", device.getSerialsNumber())
stream_num = device.getStreamNum()
print("stream_num = {}\n".format(stream_num))

with vidu.PDstream(device, 1) as tof_stream:
    suc_2 = tof_stream.init()
    print(suc_2)
    streamName = tof_stream.getStreamName()
    print(f"Stream name: {streamName}, init success: {suc_2}")

    tof_stream.set("Distance",5.0)
    tof_stream.set("StreamFps",30)
    # tof_stream.set("Resolution",1)
    tof_stream.set("AutoExposure",True)
    # tof_stream.set("Exposure",420)
    tof_stream.set("Gain",1.0)
    tof_stream.set("Threshold",0)
    tof_stream.set("DepthFlyingPixelRemoval",0)
    tof_stream.set("DepthSmoothStrength",0)

    while True:
        frame = tof_stream.getPyMat()
        print(bool(frame))
        for j, mat in enumerate(frame):
            cv.imshow("%s: %i" % (streamName, j), mat)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break