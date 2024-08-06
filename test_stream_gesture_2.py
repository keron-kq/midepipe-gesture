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

    previous_positions = None

    while True:
        # 读取 RGB 数据流
        frame1 = rgb_stream.getPyMat()
        if not frame1:
            # print("Failed to read RGB frames")
            continue

        # 读取 ToF 数据流
        frame2 = tof_stream.getPyMat()
        if not frame2:
            # print("Failed to read ToF frames")
            continue

        # 取第一帧作为显示内容
        mat1 = frame1[0]
        mat2 = frame2[0]

        # 调整 RGB 图像大小以匹配 ToF 图像的高度
        height, width = mat2.shape
        mat1_resized = cv.resize(mat1, (width, height))

        # 使用 MediaPipe 检测手掌骨骼
        rgb_image = cv.cvtColor(mat1_resized, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_positions = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    # 将骨骼点从归一化坐标转换为像素坐标
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    # 获取深度值 (z)
                    z = mat2[y, x]

                    # 三维坐标 (x, y, z)
                    point_3d = (x, y, z)
                    hand_positions.append(point_3d)

                    # 在RGB图像和深度图中绘制骨骼点
                    cv.circle(mat1_resized, (x, y), 5, (0, 255, 0), -1)
                    cv.circle(mat2, (x, y), 5, (0, 0, 255), -1)  # 使用红色

                # 绘制骨骼连线
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_x = int(hand_landmarks.landmark[start_idx].x * width)
                    start_y = int(hand_landmarks.landmark[start_idx].y * height)
                    end_x = int(hand_landmarks.landmark[end_idx].x * width)
                    end_y = int(hand_landmarks.landmark[end_idx].y * height)
                    cv.line(mat1_resized, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    cv.line(mat2, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)  # 使用红色

            if previous_positions:
                # 计算手掌中心的三维坐标变化
                previous_center = np.mean(previous_positions, axis=0)
                current_center = np.mean(hand_positions, axis=0)

                dx = current_center[0] - previous_center[0]
                dy = current_center[1] - previous_center[1]
                dz = current_center[2] - previous_center[2]

                # 放大缩小：根据z轴变化
                if dz > 500:
                    pyautogui.hotkey('ctrl','-')  # 放大
                elif dz < -500:
                    pyautogui.hotkey('ctrl','+')  # 缩小

                # 旋转：根据x轴和y轴变化
                if dx > 500:
                    pyautogui.hotkey('ctrl', 'left')  # 右旋
                elif dx < -500:
                    pyautogui.hotkey('ctrl', 'right')  # 左旋
                if dy > 500:
                    pyautogui.hotkey('ctrl', 'down')  # 下旋
                elif dy< -500:
                    pyautogui.hotkey('ctrl', 'up')  # 上旋

            previous_positions = hand_positions

        # 显示 RGB 图像+
        cv.imshow("RGB Stream", mat1_resized)

        # 显示 ToF 图像
        cv.imshow("ToF Stream", mat2)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
