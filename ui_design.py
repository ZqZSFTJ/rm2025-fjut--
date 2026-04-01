import cv2
import numpy as np
import os
import time
from collections import deque

class MapVisualizer:
    def __init__(self, map_path="image//map.jpg", history_length=10):
        self.map_path = map_path
        self.map_img = self._load_map_image()
        self.original_map = self.map_img.copy() if self.map_img is not None else None
        self.map_height, self.map_width = self.map_img.shape[:2] if self.map_img is not None else (0, 0)
        self.history_length = history_length
        
        self.colors = {
            'B': (255, 140, 0),
            'R': (0, 0, 255)
        }
        self.position_history = {}
        self.map_size = (28.0, 15.0)
        self.last_update_time = time.time()
        self.last_positions = {}
        
        self.overlay_alpha = 0.7
        
        cv2.namedWindow("Radar Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Radar Map", 800, 450)
        
        # 添加用于临时存储新添加的敌方和友方位置
        self.current_enemy_positions = {}
        self.current_friendly_positions = {}
        
        # 当前地图显示帧
        self.current_map_frame = None
        
    def _load_map_image(self):
        if not os.path.exists(self.map_path):
            print(f"警告: 地图图像未找到 '{self.map_path}'")
            return np.zeros((800, 1500, 3), dtype=np.uint8)
        try:
            img = cv2.imread(self.map_path)
            if img is None:
                print(f"警告: 无法加载地图图像 '{self.map_path}'")
                return np.zeros((800, 1500, 3), dtype=np.uint8)
            return img
        except Exception as e:
            return np.zeros((800, 1500, 3), dtype=np.uint8)
    
    def _world_to_pixel(self, world_pos):
        x_scale = self.map_width / self.map_size[0]
        y_scale = self.map_height / self.map_size[1]
        pixel_x = int(world_pos[0] * x_scale)
        pixel_y = int(self.map_height - world_pos[1] * y_scale)
        
        return (pixel_x, pixel_y)
    
    def _update_position_history(self, robot_id, position):
        if robot_id not in self.position_history:
            self.position_history[robot_id] = deque(maxlen=self.history_length)
            
        pixel_pos = self._world_to_pixel(position)
        self.position_history[robot_id].append(pixel_pos)
    
    def _draw_position_history(self, map_display, robot_id):
        if robot_id not in self.position_history or len(self.position_history[robot_id]) < 2:
            return
        
        team = robot_id[0]  # 'B'或者'R'
        color = self.colors.get(team, (0, 255, 0))
        
        points = list(self.position_history[robot_id])
        for i in range(1, len(points)):
            alpha = 0.3 + 0.7 * (i / len(points))
            cv2.line(map_display, points[i-1], points[i], color, thickness=max(1, int(3*alpha)), lineType=cv2.LINE_AA)
        
    def update_map(self, enemy_positions, friendly_positions = None):
        map_display = self.original_map.copy()
        current_time = time.time()
        overlay = map_display.copy()
        grid_spacing_meters = 5.0
        grid_color = (150, 150, 150)
        
        for y in range(0, int(self.map_size[1]) + 1, int(grid_spacing_meters)):
            start_point = self._world_to_pixel((0, y))
            end_point = self._world_to_pixel((self.map_size[0], y))
            cv2.line(map_display, start_point, end_point, grid_color, 1, cv2.LINE_AA)
            
        for x in range(0, int(self.map_size[0]) + 1, int(grid_spacing_meters)):
            start_point = self._world_to_pixel((x, 0))
            end_point = self._world_to_pixel((x, self.map_size[1]))
            cv2.line(map_display, start_point, end_point, grid_color, 1, cv2.LINE_AA)
            
        for enemy_id, position in enemy_positions.items():
            team = enemy_id[0]
            color = self.colors.get(team, (0, 255, 0))
            self._update_position_history(enemy_id, position)
            pixel_pos = self._world_to_pixel(position)
            self._draw_position_history(map_display, enemy_id)
            
            cv2.circle(overlay, pixel_pos, 15, color, -1)
            cv2.circle(map_display, pixel_pos, 10, color, -1)
            cv2.circle(map_display, pixel_pos, 12, color, 2)
            cv2.putText(map_display, enemy_id, (pixel_pos[0] + 15, pixel_pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            cv2.putText(map_display, enemy_id, (pixel_pos[0] + 15, pixel_pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            
            self.last_positions[enemy_id] = position
            
        if friendly_positions:
            for friendly_id, position in friendly_positions.items():
                team = friendly_id[0]
                color = self.colors.get(team, (0, 255, 0))
                
                self._update_position_history(friendly_id, position)
                pixel_pos = self._world_to_pixel(position)
                self._draw_position_history(map_display, friendly_id)
                    
                triangle_points = np.array([
                    [pixel_pos[0], pixel_pos[1] - 12],
                    [pixel_pos[0] - 8, pixel_pos[1] + 6],
                    [pixel_pos[0] + 8, pixel_pos[1] + 6]], np.int32).reshape((-1, 1, 2))
                
                cv2.fillPoly(overlay, [triangle_points], color)
                cv2.fillPoly(map_display, [triangle_points], color)
                cv2.polylines(map_display, [triangle_points], True, (255, 255, 255), 2)
                cv2.putText(map_display, friendly_id, (pixel_pos[0] + 15, pixel_pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                cv2.putText(map_display, friendly_id, (pixel_pos[0] + 15, pixel_pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
                
                self.last_positions[friendly_id] = position
        
        cv2.addWeighted(overlay, 0.3, map_display, 0.7, 0, map_display)
        
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(map_display, f"Time: {timestamp}", (10, self.map_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        scale_start = (self.map_width - 120, self.map_height - 30)
        scale_end = (self.map_width - 20, self.map_height - 30)
        cv2.line(map_display, scale_start, scale_end, (255, 255, 255), 2)
        
        self.last_update_time = current_time
        self.current_map_frame = map_display.copy()
        
        return map_display
    
    def show_map(self, enemy_positions, friendly_positions=None, window_name="Radar Map"):
        map_display = self.update_map(enemy_positions, friendly_positions)
        cv2.imshow(window_name, map_display)

    def clear(self):
        self.current_enemy_positions = {}
        self.current_friendly_positions = {}

    def add_enemy(self, robot_id, x, y):
        self.current_enemy_positions[robot_id] = (x, 15-y)

    def add_friendly(self, robot_id, x, y):
        self.current_friendly_positions[robot_id] = (x, 15-y)

    def update(self):
        map_display = self.update_map(self.current_enemy_positions, self.current_friendly_positions)
        cv2.imshow("Radar Map", map_display)
        cv2.waitKey(1)

    def get_map_frame(self):
        """获取当前地图帧，如果尚未更新过地图，则创建空白地图帧"""
        if self.current_map_frame is None:
            # 如果还没有更新过地图，先创建一个基本地图帧
            map_display = self.original_map.copy()
            
            # 添加网格线
            grid_spacing_meters = 5.0
            grid_color = (150, 150, 150)
            
            for y in range(0, int(self.map_size[1]) + 1, int(grid_spacing_meters)):
                start_point = self._world_to_pixel((0, y))
                end_point = self._world_to_pixel((self.map_size[0], y))
                cv2.line(map_display, start_point, end_point, grid_color, 1, cv2.LINE_AA)
                
            for x in range(0, int(self.map_size[0]) + 1, int(grid_spacing_meters)):
                start_point = self._world_to_pixel((x, 0))
                end_point = self._world_to_pixel((x, self.map_size[1]))
                cv2.line(map_display, start_point, end_point, grid_color, 1, cv2.LINE_AA)
            
            # 添加时间戳
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(map_display, f"Time: {timestamp}", (10, self.map_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 添加比例尺
            scale_start = (self.map_width - 120, self.map_height - 30)
            scale_end = (self.map_width - 20, self.map_height - 30)
            cv2.line(map_display, scale_start, scale_end, (255, 255, 255), 2)
            
            self.current_map_frame = map_display.copy()
            
        return self.current_map_frame
        
    def get_map_size(self):
        return (self.map_width, self.map_height)