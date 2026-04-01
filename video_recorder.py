import cv2
import os
import time
from datetime import datetime
import numpy as np
import logging
import json

class VideoRecorder:
    def __init__(self, base_dir="E:\\RM_INFO"):
        self.base_dir = base_dir
        # 创建时间戳文件夹
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_dir, self.timestamp)
        
        # 创建目录结构
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "raw_video"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "processed_video"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "map_video"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "logs"), exist_ok=True)
        # os.makedirs(os.path.join(self.session_dir, "positions"), exist_ok=True)
        
        # 初始化日志
        self.setup_logger()
        
        # 视频写入器
        self.raw_writer = None
        self.processed_writer = None
        self.map_writer = None
        
        self.is_recording = False
        self.logger.info(f"视频录制会话初始化于 {self.session_dir}")
        
        # 位置数据记录
        # self.position_log_file = os.path.join(self.session_dir, "positions", "positions.json")
        # self.positions = []
        
        # 帧计数器
        self.frame_count = 0

        # 视频文件格式，可以是'mp4'或'avi'
        self.video_format = 'mp4'
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('radar_system')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器避免重复
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建文件处理器
        log_file = os.path.join(self.session_dir, "logs", "system_log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_recording(self, frame_size, map_size=None):
        """开始录制视频"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.frame_count = 0
        
        # 确保帧大小是整数
        frame_size = (int(frame_size[0]), int(frame_size[1]))
        
        # 设置视频编解码器和文件路径
        if self.video_format == 'mp4':
            # MP4使用H.264编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            raw_path = os.path.join(self.session_dir, "raw_video", "raw.mp4")
            processed_path = os.path.join(self.session_dir, "processed_video", "processed.mp4")
            map_path = os.path.join(self.session_dir, "map_video", "map.mp4")
        else:
            # AVI默认使用XVID编码
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            raw_path = os.path.join(self.session_dir, "raw_video", "raw.avi")
            processed_path = os.path.join(self.session_dir, "processed_video", "processed.avi")
            map_path = os.path.join(self.session_dir, "map_video", "map.avi")
        
        fps = 30.0
        
        # 初始化视频写入器
        self.raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, frame_size)
        self.processed_writer = cv2.VideoWriter(processed_path, fourcc, fps, frame_size)
        
        if map_size:
            map_size = (int(map_size[0]), int(map_size[1]))
            self.map_writer = cv2.VideoWriter(map_path, fourcc, fps, map_size)
        else:
            self.map_writer = cv2.VideoWriter(map_path, fourcc, fps, (800, 450))
        
        self.logger.info(f"开始录制视频到 {self.session_dir}, 格式: {self.video_format}")
        
        # 检查视频写入器是否正常初始化
        if not self.raw_writer.isOpened():
            self.logger.warning(f"原始视频写入器初始化失败，可能不支持此编解码器，尝试使用另一种格式")
            # 尝试使用另一种编解码器
            if self.video_format == 'mp4':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                raw_path = os.path.join(self.session_dir, "raw_video", "raw.avi")
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                raw_path = os.path.join(self.session_dir, "raw_video", "raw.mp4")
            self.raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, frame_size)
    
    def record_frame(self, raw_frame, processed_frame, map_frame):
        """记录一帧视频"""
        if not self.is_recording:
            return
            
        # 添加调试输出
        if raw_frame is None:
            self.logger.warning("原始帧为空")
            return
        if processed_frame is None:
            self.logger.warning("处理后的帧为空")
            return
        if map_frame is None:
            self.logger.warning("地图帧为空")
            return
        
        try:
            # 写入视频帧
            if self.raw_writer.isOpened():
                self.raw_writer.write(raw_frame)
            
            if self.processed_writer.isOpened():
                self.processed_writer.write(processed_frame)
                
            if self.map_writer.isOpened():
                self.map_writer.write(map_frame)
                
            self.frame_count += 1
            
            # 每100帧记录一次日志
            if self.frame_count % 100 == 0:
                self.logger.info(f"已录制 {self.frame_count} 帧")
                
        except Exception as e:
            self.logger.error(f"写入视频帧出错: {e}")
    
    def log_message(self, message, level="info"):
        """记录信息到日志文件"""
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
        elif level.lower() == "debug":
            self.logger.debug(message)
    
    def log_referee_data(self, double_vulnerability_chance, opponent_double_vulnerability, color):
        """记录裁判系统数据"""
        self.logger.info(f"双倍易伤机会: {double_vulnerability_chance}, "
                         f"对方双倍易伤状态: {opponent_double_vulnerability}, "
                         f"队伍颜色: {color}")
    
    def log_serial_packet(self, packet, positions=None, seq=None):
        """记录串口数据包"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 记录到日志
        packet_hex = str(packet)
        self.logger.info(f"串口数据包 seq={seq}: {packet_hex}")
        
        # 如果同时提供了位置数据，也进行记录
        if positions is not None:
            self.log_positions(positions, seq)
    
    def log_received_serial_packet(self, packet, mark_value=None, seq=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        packet_hex = str(packet)
        self.logger.info(f"串口接收数据包 seq={seq}: {packet_hex}")
        self.logger.info(f"串口接收标记进度 mark value={mark_value}")
    
    def log_positions(self, positions, seq=None):
        """记录机器人位置数据"""
        # 仅记录到日志，不再保存到JSON文件
        self.logger.info(f"位置数据 seq={seq}: {positions}")
        
        # def convert_numpy_types(obj):
        #     if isinstance(obj, np.generic):
        #         return obj.item()
        #     elif isinstance(obj, np.ndarray):
        #         return obj.tolist()
        #     elif isinstance(obj, list):
        #         return [convert_numpy_types(item) for item in obj]
        #     elif isinstance(obj, dict):
        #         return {k: convert_numpy_types(v) for k, v in obj.items()}
        #     else:
        #         return obj
        
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # entry = {
        #     "timestamp": timestamp,
        #     "seq": seq,
        #     "positions": convert_numpy_types(positions)
        # }
        # self.positions.append(entry)
        
        # # 每10条记录保存一次
        # if len(self.positions) % 10 == 0:
        #     self._save_positions()
        
        # # 记录到日志
        # self.logger.info(f"位置数据 seq={seq}: {positions}")
    
    def _save_positions(self):
        """保存位置数据到JSON文件"""
        # 已注释掉，不再保存位置数据
        pass
        # if not self.positions:
        #     return
            
        # try:
        #     with open(self.position_log_file, 'w') as f:
        #         json.dump(self.positions, f, indent=2)
        #     self.logger.info(f"保存位置数据: {len(self.positions)}条记录")
        # except Exception as e:
        #     self.logger.error(f"保存位置数据失败: {str(e)}")
        #     # 尝试更强力的类型转换
        #     try:
        #         # 确保所有数据都被转换为Python原生类型
        #         import numpy as np
        #         def deep_convert(obj):
        #             if isinstance(obj, np.generic):
        #                 return obj.item()
        #             elif isinstance(obj, np.ndarray):
        #                 return obj.tolist()
        #             elif isinstance(obj, list):
        #                 return [deep_convert(item) for item in obj]
        #             elif isinstance(obj, dict):
        #                 return {k: deep_convert(v) for k, v in obj.items()}
        #             else:
        #                 return obj
                
        #         converted_positions = deep_convert(self.positions)
        #         with open(self.position_log_file, 'w') as f:
        #             json.dump(converted_positions, f, indent=2)
        #         self.logger.info(f"使用强制类型转换后成功保存位置数据: {len(self.positions)}条记录")
        #     except Exception as e2:
        #         self.logger.error(f"二次尝试保存位置数据失败: {str(e2)}")
    
    def stop_recording(self):
        """停止录制视频"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # 释放视频写入器
        if self.raw_writer is not None:
            self.raw_writer.release()
            
        if self.processed_writer is not None:
            self.processed_writer.release()
            
        if self.map_writer is not None:
            self.map_writer.release()
        
        # 保存剩余的位置数据
        # self._save_positions()
            
        self.logger.info(f"停止录制会话于 {self.session_dir}，共录制 {self.frame_count} 帧") 