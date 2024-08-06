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

# 使用第一个数据流作为 RGB 流
with vidu.PDstream(device, 0) as rgb_stream:
    suc_1 = rgb_stream.init()
    print(suc_1)
    streamName = rgb_stream.getStreamName()
    print(f"Stream name: {streamName}, init success: {suc_1}")

    rgb_stream.set("AutoExposure", False)
    rgb_stream.set("Exposure", 29984)
    rgb_stream.set("Gain", 24)
    rgb_stream.set("Resolution", 3)
    rgb_stream.set("StreamFps", 30)

    while True:
        frame = rgb_stream.getPyMat()
        # print(bool(frame))
        for j, mat in enumerate(frame):
            cv.imshow("%s: %i" % (streamName, j), mat)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break