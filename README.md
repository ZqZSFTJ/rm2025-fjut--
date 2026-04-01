基于 YOLOv5 + TensorRT 的目标检测与追踪系统
📖 项目简介
本项目是一个基于 YOLOv5 和 TensorRT 的高性能目标检测与追踪系统，集成了海康威视相机（hik_camera）、串口通信（my_serial）和 GUI 界面（ui_design），适用于工业检测、智能监控等场景。

✨ 主要特性
🚀 TensorRT 加速：支持 ONNX 和 TensorRT 引擎，大幅提升推理速度

🎯 YOLOv5 目标检测：高精度实时目标检测

🔍 DeepSORT 目标追踪：多目标持续追踪

📷 海康相机支持：可直接接入海康工业相机

🖥️ 图形界面：基于 PyQt 的可视化操作界面

📹 视频录制：支持检测过程实时录制

📍 定位功能：目标位置信息输出

📁 项目结构

下载
E:.
├── calibration.py              # 相机标定模块
├── config.py                   # Python 配置文件
├── deepsortTracker.py          # DeepSORT 追踪器实现
├── hik_camera.py               # 海康相机驱动接口
├── icudaengine.py              # TensorRT ICudaEngine 封装
├── location.py                 # 目标定位算法
├── main.py                     # 主程序入口
├── my_serial.py                # 串口通信模块
├── onnx_engine.py              # ONNX 推理引擎
├── plan.txt                    # 项目计划文档
├── readme.md                   # 项目说明文档（本文件）
├── requirements.txt            # Python 依赖列表
├── ui_design.py                # PyQt 界面设计
├── utils.py                    # 通用工具函数
├── video_recorder.py           # 视频录制模块
└── yolov5Detector.py           # YOLOv5 检测器封装
🔧 环境配置
系统要求
操作系统：Windows 10/11

Python 版本：3.8（推荐）

CUDA 版本：11.x 或 12.x（根据 TensorRT 版本）

显卡：NVIDIA GPU（支持 CUDA）


下载
python my_serial.py
📦 核心模块说明
模块	功能描述
yolov5Detector.py	YOLOv5 检测器封装，支持 ONNX 和 TensorRT 推理
deepsortTracker.py	DeepSORT 多目标追踪算法实现
TRTEngine.py	TensorRT 推理引擎，提供高性能推理接口
hik_camera.py	海康相机 SDK 封装，支持实时视频流采集
ui_design.py	PyQt5 图形界面，可视化检测结果
video_recorder.py	视频录制模块，支持检测画面保存
location.py	目标定位算法，输出坐标信息
my_serial.py	串口通信，用于与下位机交互
🎮 使用说明
图形界面操作
运行 python ui_design.py 启动界面

选择相机源（本地相机/海康相机/视频文件）

加载模型（YOLOv5 权重或 TensorRT 引擎）

点击"开始检测"

检测结果实时显示，支持截图和录制


📧 联系方式
如有问题或建议，请联系项目维护者。

最后更新：2026年4月1日
