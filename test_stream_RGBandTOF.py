import pyvidu as vidu
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

# 初始化 MediaPipe Hands
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
with vidu.PDstream(device, 0) as rgb_stream, vidu.PDstream(device, 1) as tof_stream:

    suc_1 = rgb_stream.init()
    print(suc_1)
    stream_name_rgb = rgb_stream.getStreamName()
    print(f"Stream name: {stream_name_rgb}, init success: {suc_1}")

    suc_2 = tof_stream.init()
    print(suc_2)
    stream_name_tof = tof_stream.getStreamName()
    print(f"Stream name: {stream_name_tof}, init success: {suc_2}")

    rgb_stream.set("AutoExposure", False)
    rgb_stream.set("Exposure", 29984)
    rgb_stream.set("Gain", 24)
    rgb_stream.set("StreamFps", 30)

    tof_stream.set("Distance", 5.0)
    tof_stream.set("StreamFps", 30)
    tof_stream.set("AutoExposure", True)
    tof_stream.set("Exposure", 420)
    tof_stream.set("Gain", 1.0)
    tof_stream.set("Threshold", 0)
    tof_stream.set("DepthFlyingPixelRemoval", 0)
    tof_stream.set("DepthSmoothStrength", 0)

    while True:
        # 读取 RGB 数据流
        frame1 = rgb_stream.getPyMat()
        if not frame1:
            print("Failed to read RGB frames")
            continue

        # 读取 ToF 数据流
        frame2 = tof_stream.getPyMat()
        if not frame2:
            print("Failed to read ToF frames")
            continue

        # 取第一帧作为显示内容
        mat1 = frame1[0]
        mat2 = frame2[0]

        # 调整 RGB 图像大小以匹配 ToF 图像的高度
        height, width = mat2.shape
        mat1_resized = cv.resize(mat1, (width, height))

        # 显示 RGB 图像
        cv.imshow("RGB Stream", mat1_resized)

        # 显示 ToF 图像
        cv.imshow("ToF Stream", mat2)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
