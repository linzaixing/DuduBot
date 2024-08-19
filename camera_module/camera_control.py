import cv2
import os
import re
import time
from yolo_module import onnx_yolov8_detect as onnx_yolo


def get_next_filename(save_type, path):
    if save_type == 1:
        str_type = 'img'
        suffix = 'jpg'
    elif save_type == 2:
        str_type = 'video'
        suffix = 'avi'
    else:
        print('保存类型错误')
        return None  # 处理未知的save_type情况

    # 定义匹配img_(video)数字.jpg(.mp4)的正则表达式
    pattern = re.compile(rf'{str_type}_(\d+)\.{suffix}')

    max_num = -1

    for filename in os.listdir(path):
        # 查找匹配正则表达式的文件名
        match = pattern.match(filename)
        if match:
            # 提取数字部分并转换为整数
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    if max_num == -1:
        # 如果没有找到匹配的文件
        if save_type == 1:
            return path + '/img_0.jpg'
        elif save_type == 2:
            return path + '/video_0.avi'
    else:
        if save_type == 1:
            # 否则返回img_max+1.jpg
            return path + f'/img_{max_num + 1}.jpg'
        elif save_type == 2:
            return path + f'/video_{max_num + 1}.avi'


class CameraController:
    def __init__(self):
        self.cap = None
        self.out = None
        self.ret = None
        self.fourcc = None
        self.frame = None
        self.detect = None
        self.ai_detector = None
        self.i = 0

        self.session = None
        self.model_inputs = None
        self.input_width = None
        self.input_height = None

    def open_camera(self) -> None:
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap.set(cv2.CAP_PROP_FOURCC, self.fourcc)

        if not self.cap.isOpened():
            print("无法打开摄像头")
            return

        # 初始化帧数计数器和起始时间
        frame_count = 0
        start_time = time.time()
        while True:
            # 读取帧
            self.ret, self.frame = self.cap.read()

            # 如果没有正确读取帧，ret为False
            if not self.ret:
                print("无法接收帧，请退出")
                break

            if self.detect:
                # yolov8 model detect
                output_image = onnx_yolo.detect_run(self.frame,
                                                    self.ai_detector,
                                                    self.session,
                                                    self.model_inputs,
                                                    self.input_width,
                                                    self.input_height)
                # 计算帧速率
                frame_count += 1
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000
                fps = frame_count / (end_time - start_time)
                aver_cost = elapsed_time / frame_count

                # 将FPS绘制在图像上
                cv2.putText(output_image,
                            f"{frame_count}, FPS: {fps:.2f}, cost: {aver_cost:.2f} ms",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA)
                self.frame = output_image

            if self.out:
                # 写入帧到video文件
                self.out.write(self.frame)

            # 展示帧
            cv2.imshow('Camera', self.frame)

            # 按 'q' 退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close_camera()
                break

        # 关闭相机
        self.close_camera()

    def frame_edit(self, model_path):
        if self.cap is not None:
            self.ai_detector = onnx_yolo.YoloUtil()

            # 初始化检测模型，加载模型并获取模型输入节点信息和输入图像的宽度、高度
            self.session, self.model_inputs, self.input_width, self.input_height = self.ai_detector.init_detect_model(
                model_path)
            print('开始人体检测')
            self.detect = True
        else:
            print('camera未打开')

    def save_capture(self, save_type, file_path=os.path.dirname(os.path.abspath(__file__))):
        if self.cap is not None:
            print('保存路径: ', file_path)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                print('创建保存文件夹：', file_path)

            save_path = get_next_filename(save_type, file_path)
            if save_type == 1:
                print('保存图片至：', save_path)
                cv2.imwrite(save_path, self.frame)
            elif save_type == 2:
                print('视频录制中，结果将保存在：', save_path)
                self.out = cv2.VideoWriter(save_path, self.fourcc, 30.0, (640, 480))
        else:
            print('camera未打开')

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            if self.out is not None:
                self.out.release()
            # 关闭所有OpenCV窗口
            cv2.destroyAllWindows()
            self.cap = None
            self.out = None
            self.detect = None
            text = "Camera have closed!"
            print(text)
            return text
        else:
            text = "Camera have closed!"
            print(text)
            return text
