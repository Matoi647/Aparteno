import detect
import argparse
import numpy as np
import sys
import os
import json
from pathlib import Path
from utils.general import (print_args, xywh2xyxy, yaml_load, increment_path)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# demo01
# 在电脑端上运行，每当按下键盘上的“1”时，将该指令传输至树莓派并等待接收树莓派回传的图片。
# 使用OpenCV和Python访问树莓派相机

import cv2
import socket
import struct
import pickle

def demo01():
    # 创建一个套接字对象，用于和树莓派通信
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到树莓派的IP地址和端口号（你可以自己修改）
    s.connect(('192.168.4.3', 8000))

    while True:
        key0 = cv2.waitKey(1) & 0xFF
        key = input("输入 1 获得图片，输入 q 退出程序：")
        if key == '1':
            # if key == ord('1'):   # 如果按下键盘上的“1”
            print('Sending command')
            cmd_data = '1'  # 指令内容为字符串“1”
            # 将指令内容编码为字节流（bytes类型）
            cmd_data = cmd_data.encode()
            # 获取字节流的长度，并转换为4个字节的二进制数据（struct.pack返回bytes类型）
            cmd_len = struct.pack('I', len(cmd_data))
            # 发送指令长度和指令内容给树莓派
            s.sendall(cmd_len + cmd_data)

            # 接收4个字节的图像长度
            img_len = s.recv(4)
            if not img_len:
                break  # 如果没有数据，跳出循环
            # 解码图像长度
            img_len = struct.unpack('I', img_len)[0]

            # 接收图像内容，并解码为numpy数组
            img_data = b''  # 初始化一个空字节流变量
            while len(img_data) < img_len:  # 循环接收数据，直到达到图像长度
                data = s.recv(4096)  # 每次接收4096个字节
                if not data:
                    break  # 如果没有数据，跳出循环
                img_data += data  # 将接收到的数据拼接到img_data变量中

            if len(img_data) == img_len:  # 如果接收完整
                print('Received image')
                # 将字节流转换为numpy数组（pickle.loads返回numpy数组）
                img_data = pickle.loads(img_data)
                # 将numpy数组解码为图像（cv2.imdecode返回cv2.Mat类型）
                img = cv2.imdecode(img_data, 1)
                cv2.imshow('Image from Raspberry Pi', img)  # 显示图像
                cv2.imwrite(ROOT / 'TestPictures/img0.png', img)

            # demo01()
            labels = demo02("TestPictures")
            print("lllll", labels)
            json_data = demo04(labels)
            print("jjjjjj", json_data)

        elif key == 'q':  # 如果按下键盘上的“q”
            break  # 跳出循环

    s.close()  # 关闭套接字对象
    cv2.destroyAllWindows()  # 销毁所有窗口



# demo02
def parse_opt(source):
    source = Path(source)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best30.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'TestPictures', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC_trash.yaml', help='(optional) dataset.yaml path')
    # parser.add_argument('--data', type=str, help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-conf', action='store_true', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--name', default='detect_output', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def demo02(source:str):
    source = Path(source)
    data = ROOT / 'data/VOC_trash.yaml'
    opt = parse_opt(source)
    detect.main(opt)

    # output_path = Path(source / 'detect_output')
    output_path = increment_path(Path(source) / 'detect_output', exist_ok=True)  # increment run

    labels = np.loadtxt(output_path / 'labels' / 'img0.txt')

    detect_num = labels.shape
    if isinstance(labels[0], float):
        labels = labels.reshape(1, 6)
    # print(labels)

    xywh = labels[:, 1:5]# xywh为中心坐标
    # xyxy = xywh2xyxy(xywh)
    labels[:, 1:5] = xywh

    labels = labels[:, 0:3]# 只保留class, x, y

    cls44 = np.asarray(labels[:, 0], int)# 44分类编号
    names = yaml_load(data)['names']
    cls_names = [names[c] for c in cls44]# 类别名
    # cls4 = np.zeros(detect_num, dtype=int)# 4分类编号
    for i, cls_name in enumerate(cls_names):
        cls_name = cls_name.split('-')[0]
        if cls_name == '垃圾桶':
            labels[i][0] = 0
        elif cls_name == '干垃圾':
            labels[i][0] = 1
        elif cls_name == '可回收垃圾':
            labels[i][0] = 2
        elif cls_name == '湿垃圾':
            labels[i][0] = 3
        elif cls_name == '有害垃圾':
            labels[i][0] = 4
        else:
            labels[i][0] = -1

    print("labels: ", labels)
    for label in labels:
        x, y = label[1], label[2]
        label[1] = 35.22976016 * x + -1.38619677 * y - 15.59901126
        label[2] = -2.15526187 * x - 28.45227606 * y + 35.18944689


    # dt = np.dtype([('class', int), ('x', float), ('y', float), ('w', float), ('h', float), ('confidence', float)])
    dt = np.dtype([('class', int), ('x', float), ('y', float)])

    labels = np.array([tuple(e) for e in labels], dtype=dt)
    labels = np.sort(labels, order=['class'])
    labels = np.asarray(labels)
    # print("xyxy:\n", xyxy)
    # print(labels)
    return labels


def demo04(label):
    # 创建嵌套字典数据
    BOTTOM=0
    HEIGHT=60
    data={}
    index=0
    beginx=-15.0
    beginy=-5.0
    end=-10
    for i in label:
        x=i[1]
        y=i[2]
        target=i[0]
        targetx=0.0
        targety=0.0
        if(target==1):
            targetx=15.06
            targety=2.56
        elif(target==2):
            targetx=13.82
            targety=-11.98
        elif(target==3):
            targetx=-15.06
            targety=2.56
        elif(target==4):
            targetx=-13.82
            targety=-11.98
        data[str(index)]={
            "pa": 4,
            "po": True,
            "x1": beginx,
            "x2": x,
            "y1": beginy,
            "y2": y,
            "z1": HEIGHT,
            "z2": HEIGHT
        }#0
        index+=1
        data[str(index)]={
            "pa": 4,
            "po": True,
            "x1": x,
            "x2": x,
            "y1": y,
            "y2": y,
            "z1": HEIGHT,
            "z2": BOTTOM
        }#1
        index+=1
        data[str(index)]={
            "pa": 4,
            "po": False,
            "x1": x,
            "x2": x,
            "y1": y,
            "y2": y,
            "z1": BOTTOM,
            "z2": BOTTOM
        }#2
        index+=1
        data[str(index)]={
            "pa": 4,
            "po": False,
            "x1": x,
            "x2": x,
            "y1": y,
            "y2": y,
            "z1": BOTTOM,
            "z2": HEIGHT
        }#3
        index+=1
        data[str(index)]={
            "pa": 4,
            "po": False,
            "x1": x,
            "x2": targetx,
            "y1": y,
            "y2": targety,
            "z1": HEIGHT,
            "z2": HEIGHT
        }#4
        index+=1
        data[str(index)]={
            "pa": 4,
            "po": True,
            "x1": targetx,
            "x2": targetx,
            "y1": targety,
            "y2": targety,
            "z1": HEIGHT,
            "z2": HEIGHT
        }#5
        index+=1
        beginx=targetx
        beginy=targety
    data[str(index)]={
            "pa": 4,
            "po": True,
            "x1": beginx,
            "x2": -15.0,
            "y1": beginy,
            "y2": -5.0,
            "z1": HEIGHT,
            "z2": HEIGHT
    }
    # 将数据写入JSON文件
    with open(ROOT / 'TestPictures/data2.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)
        print("finished!\n")
    return data

# # demo04
# import json
# def demo04(labels):
#     # 创建嵌套字典数据
#     x = labels[0]['x']
#     y = labels[0]['y']
#     target = labels[0]['class']
#     BOTTOM = 0
#     HEIGHT = 60
#
#     targetx = 0.0
#     targety = 0.0
#     if (target == 1):
#         targetx = 18
#         targety = 0
#     elif (target == 2):
#         targetx = 13.82
#         targety = -11.98
#     elif (target == 3):
#         targetx = -15.06
#         targety = 2.56
#     elif (target == 4):
#         targetx = -13.82
#         targety = -11.98
#
#     data = {
#         "0": {
#             "pa": 4,
#             "po": True,
#             "x1": 0.0,
#             "x2": x,
#             "y1": 10.0,
#             "y2": y,
#             "z1": HEIGHT,
#             "z2": HEIGHT
#         },
#         "1": {
#             "pa": 4,
#             "po": True,
#             "x1": x,
#             "x2": x,
#             "y1": y,
#             "y2": y,
#             "z1": HEIGHT,
#             "z2": BOTTOM
#         },
#         "2": {
#             "pa": 4,
#             "po": False,
#             "x1": x,
#             "x2": x,
#             "y1": y,
#             "y2": y,
#             "z1": BOTTOM,
#             "z2": BOTTOM
#         },
#         "3": {
#             "pa": 4,
#             "po": False,
#             "x1": x,
#             "x2": x,
#             "y1": y,
#             "y2": y,
#             "z1": BOTTOM,
#             "z2": HEIGHT
#         },
#         "4": {
#             "pa": 4,
#             "po": False,
#             "x1": x,
#             "x2": targetx,
#             "y1": y,
#             "y2": targety,
#             "z1": HEIGHT,
#             "z2": HEIGHT
#         },
#         "5": {
#             "pa": 4,
#             "po": True,
#             "x1": targetx,
#             "x2": targetx,
#             "y1": targety,
#             "y2": targety,
#             "z1": HEIGHT,
#             "z2": HEIGHT
#         },
#         "6": {
#             "pa": 4,
#             "po": True,
#             "x1": targetx,
#             "x2": 0.0,
#             "y1": targety,
#             "y2": 10.0,
#             "z1": HEIGHT,
#             "z2": HEIGHT
#         }
#     }
#
#     # 将数据写入JSON文件
#     with open(ROOT / 'TestPictures/data2.json', 'w') as outfile:
#         json.dump(data, outfile, indent=4)
#         print("finished!\n")


if __name__ == "__main__":
    # demo01()
    # labels = demo02("TestPictures")
    # print("lllll", labels)
    # demo04(labels)
    demo01()