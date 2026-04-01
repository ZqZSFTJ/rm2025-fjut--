import cv2
import numpy as np
from location import Location
from config import Config

def calculate_homography(frame_shape):
    h, w = frame_shape[:2]
    image_corners = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
    map_corners = np.array([[0.0, 0.0], [0.0, 15.0], [28.0, 15.0], [28.0, 0.0]], dtype=np.float32)
    return cv2.findHomography(image_corners, map_corners)[0]

def draw_detection_info(frame, x1, y1, label, color):
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_y = max(y1 - th - 4, 20)
    
    cv2.rectangle(img=frame, pt1=(int(x1), int(text_y - th - 2)), pt2=(int(x1 + tw), int(text_y + 2)), color=color, thickness=-1)
    cv2.putText(frame, label, (int(x1), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
def draw_preditions(frame, detections, fps=None):
    armor_cls = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7',
                'R1', 'R2', 'R3', 'R4', 'R5', 'R7']
    
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_name, pos, armor_info = det
        
        # 绘制车辆ROI区域
        car_color = (0, 255, 0) if any(a in cls_name for a in armor_cls) else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), car_color, 2)
        label = f"{cls_name} {conf:.2f}"
        
        # 如果有装甲板信息，绘制装甲板ROI和标记点
        if armor_info is not None:
            a_x1, a_y1, a_x2, a_y2 = armor_info[:4]
            
            # 装甲板ROI（在车辆区域内）
            armor_x1 = int(x1 + a_x1)
            armor_y1 = int(y1 + a_y1)
            armor_x2 = int(x1 + a_x2)
            armor_y2 = int(y1 + a_y2)
            
            # 绘制装甲板ROI
            cv2.rectangle(frame, (armor_x1, armor_y1), (armor_x2, armor_y2), (125, 125, 125), 2)
            
            # 标记定位点（车辆ROI的x轴中点，装甲板下沿的y值）
            car_center_x = int((x1 + x2) / 2)
            bottom_y = armor_y2
            cv2.circle(frame, (car_center_x, bottom_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Position Ref", (car_center_x + 10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if pos is not None:
            label += f" ({pos[0]:.2f}, {pos[1]:.2f})"
            
            # 添加距离信息
            # distance = np.sqrt(pos[0]**2 + pos[1]**2)
            # dist_label = f"Dist: {distance:.2f}m"
            # cv2.putText(frame, dist_label, (int(x1), int(y2) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        draw_detection_info(frame, x1, y1, label, car_color)
    return frame

def process_armor(roi, armor_detector, car_box, camera_matrix, locator=None):
    armor_dets = armor_detector.detect(roi)
    if not armor_dets:
        return None
    
    best_armor = max(armor_dets, key=lambda x:x[4])
    armor_cls = armor_detector.class_names[best_armor[5]]
    
    # 装甲板在ROI中的坐标
    a_x1, a_y1, a_x2, a_y2 = best_armor[:4]
    
    # 计算车辆ROI在原始图像中的坐标
    car_x1, car_y1, car_x2, car_y2 = car_box
    
    # 新的定位点：车辆ROI的x轴中点，装甲板下沿的y值
    position_ref_x = (car_x1 + car_x2) / 2
    position_ref_y = car_y1 + a_y2  # 装甲板下沿的y坐标
    
    armor_type = 'large' if armor_cls in ['B1', 'R1'] else 'small'
    
    if locator:
        # 将新的定位点传给定位功能
        position_3d = pixel_to_3d((position_ref_x, position_ref_y), locator)
    else:
        # 默认位置
        position_3d = (0, 0, 0)
    
    # 返回装甲板类别、3D位置和装甲板ROI信息
    return armor_cls, position_3d, (a_x1, a_y1, a_x2, a_y2)

def process_frame(frame, car_detector, armor_detector, camera_matrix, map_size, locator=None):
    H = calculate_homography(frame.shape)
    final_detections = []
    
    for car in car_detector.detect(frame):
        if len(car) != 6:
            continue
        
        x1, y1, x2, y2, conf, cls_id = car
        cls_name = car_detector.class_names[cls_id]
    
        if cls_name == 'car':
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                # final_detections.append((x1, y1, x2, y2, conf, cls_name, None))
                continue
            
            armor_result = process_armor(roi, armor_detector, (x1, y1, x2, y2), camera_matrix, locator)
            if armor_result:
                armor_cls, position_3d, armor_roi = armor_result
                final_detections.append((x1, y1, x2, y2, conf, armor_cls, position_3d, armor_roi))
            else:
                final_detections.append((x1, y1, x2, y2, conf, 'car', None, None))
        else:
            final_detections.append((x1, y1, x2, y2, conf, cls_name.capitalize(), None, None))
    
    return final_detections

def pixel_to_3d(uv_point, locator):
    img_x, img_y = uv_point
    world_point = locator.parse((img_x, img_y))
    
    # 蓝方视角坐标转换
    if Config.COLOR == "B":
        return (28-world_point[0], -world_point[1], world_point[2])
    # 红方视角坐标转换
    else:
        return (world_point[0], -world_point[1], world_point[2])
    

    