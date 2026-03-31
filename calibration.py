import cv2
import numpy as np
import json
from config import Config
from hik_camera import HikCamera

# 新地图
# 因为opencv和dji的坐标系是不一样的，opencv是右手系，dji是左手系，所以五点标定点的y轴要反一下
calibrate_map_point = {     # x, y, z
    "left_buff" : (9.535-0.160, -(7.500-0.750), 0),
    "right_buff" : (9.535-0.160, -(7.500+0.750), 0),
    "self_tower" : (10.9945, -11.517, 1.331+0.400),
    "enemy_base" : (25.504, -7.500, 1.043+0.200),
    "enemy_tower" : (16.832, -3.6435, 1.331+0.400)
}#参数,std::unordered_map、boost.hana

# 旧地图点
# calibrate_map_point = {
#     "left0" : (8.67 , -5.715, 0.120 + 0.3),
#     "right0": (8.67 , -5.715 - 0.4, 0.120 + 0.3),
#     "self_tower": (11.1865, -12.419, 1.003+0.118),
#     "enemy_base": (26.153, -7.5, 1.043+0.2),
#     "enemy_tower": (16.64, -2.4215, 1.331+0.118)
# }

current_point = 0
real_map_point = []
point_names = list(calibrate_map_point.keys())# for_each
display_frame = None
print("start calibrate...")

def click_callback(event, x, y, flags, param):
    global current_point, real_map_point, display_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_point < 5:
        cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display_frame, point_names[current_point], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        real_map_point.append((x, y))
        current_point += 1
        # if display_frame is not None:
        #     temp_frame = display_frame.copy()
        #     draw_existing_points(temp_frame)
        #     cv2.imshow("calibrate", temp_frame)
        cv2.imshow("calibrate", display_frame)#点击_回调函数
        
def draw_existing_points(display_frame):
    if display_frame is not None:
        for i, (x, y) in enumerate(real_map_point):# 元组对迭代/std::views::enumerate(c++20)/
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, point_names[i], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)#绘制函数
            
def finalize_calibrate():
    cv2.destroyAllWindows()
    if len(real_map_point) != 5:
        print('需要5个标定点, 当前只有', len(real_map_point))
        return#退出函数
    
    # 标定
    
    object_points = np.array([calibrate_map_point[name] for name in point_names], dtype=np.float32)
    image_points = np.array(real_map_point, dtype=np.float32)
    
    camera_matrix = Config.CAMERA_MATRIX
    dist_coeffs = Config.DIST_COEFFS
    
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        print("求解pnp不成功")
        return
    
    calibrate_data = {
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "real_points": {name: (int(p[0]), int(p[1])) for name, p in zip(point_names, real_map_point)}
    }
    
    with open("json/calibrate_result.json", "w") as f:
        json.dump(calibrate_data, f, indent=4)
    print("标定成功！结果已保存到 json/calibrate_result.json")

def calibrate_with_video(video_path):
    global display_frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", click_callback)
    
    try:
        while cap.isOpened() and current_point < 5:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            draw_existing_points()
            cv2.imshow('calibrate', display_frame)
            
            if current_point >= 5:
                    break
                
            if cv2.waitKey(30) == 27:  # ESC
                    break
    finally:
        cap.release()
        finalize_calibrate()
        
def calibrate_with_hik_camera(camera_config):
    global display_frame
    camera = HikCamera(camera_config)
    cv2.namedWindow('calibrate', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('calibrate', click_callback)
    
    try:
        while current_point < 5:
            frame = camera.get_latest_frame()
            if frame is not None:
                display_frame = frame.copy()
                draw_existing_points()
                cv2.imshow('calibrate', display_frame)
            
            if current_point >= 5:
                break
            
            key = cv2.waitKey(1)
            if key == 27:   # ESC
                break
    except Exception as e:
        print(f"相机标定出错: {str(e)}")
    finally:
        camera.stop()
        finalize_calibrate()
        
def calibrate_with_image(image_path):
    global display_frame
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法加载图片: {image_path}")
        return
    # frame = cv2.resize(frame, (1500, 1000))
    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", click_callback)
    try:
        display_frame = frame.copy()
        draw_existing_points(display_frame)
        cv2.imshow('calibrate', display_frame)
        while current_point < 5:
            key = cv2.waitKey(30)
            if key == 27:   # ESC
                break
    finally:
        finalize_calibrate()

if __name__ == "__main__":
    mode = Config.SELECT_MODE
    
    if mode == 'test':
        #calibrate_with_video(r"D:\fjut_radar\image\test20250512.mp4")
        calibrate_with_image(Config.IMG_PATH)
    elif mode == 'hik':
        calibrate_with_hik_camera(Config.HIK_CONFIG)