import cv2
import json
import yaml
import numpy as np
from config import Config
from pathlib import Path

camera_matrix = Config.CAMERA_MATRIX
dist_coeffs = Config.DIST_COEFFS  # 畸变系数

class Location:
    def __init__(self, calibrate_path="json\\calibrate_result.json", map_config_path="yaml\\RM2025_Points.yaml"):
        with open(calibrate_path) as f:
            data = json.load(f)
            self.camera_matrix = Config.CAMERA_MATRIX.copy()
            self.dist_coeffs = Config.DIST_COEFFS
            self.rvec = np.array(data["rvec"])
            self.tvec = np.array(data["tvec"])
        
        with open(map_config_path) as f:
            self.raw_map_data = yaml.safe_load(f)
            
        # self.region_height = {      # 高度映射(旧地图)
        #     "Left_Road" : 0.2,
        #     "Right_Road" : 0.2,
        #     "Self_Ring_High" : 0.6,
        #     "Enemy_Ring_High" : 0.6,
        #     "Enemy_Left_High" : 0.4,
        #     "Enemy_Right_High" : 0.4,
        #     "Enemy_Buff" : 0.8
        # }
        self.region_height = {      # 高度映射(新地图)
            "Self_Tower": 0.25,
            "Enemy_Tower": 0.25,
            "Middle_High": 0.4,
            "Self_Left_High": 0.473,
            "Enemy_Left_High": 0.473,
            "Self_Ring_High": 0.2,
            "Enemy_Ring_High": 0.2,
            "Enemy_Buff": 0.8
        }
        
        self.regions = {}
        for region_name, points in self.raw_map_data.items():
            points_3d = np.array([[p['x'], p['y'], p['z']] for p in points], dtype=np.float32)
            self.regions[region_name] = {
                "points_3d" : points_3d,                                    # 世界坐标系的3d点位
                "height" : self.region_height.get(region_name, 0.0),        # 获取在这个点位涵盖的应有的高度
                "points_2d" : None                                          # 图像坐标系的2d点位
            }
        
        self.update_2d_projection()
    
    # 把区域内的region_name里的所有世界坐标系的坐标都暂时转换成图像坐标系中的坐标，为了便于后面调用get_height函数的时候得到高度（用于高度补偿）
    def update_2d_projection(self):
        # img_width, img_height = 1920, 1080      # MV-CS020-10UC相机分辨率
        
        for region_name, region in self.regions.items():
            # 点映射把世界的3d点映射到图像中的2d点里
            points_2d, _ = cv2.projectPoints(region["points_3d"], self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
            # 然后把这个2d点存储在定义的结构里的points_2d部分，这样这个结构就完整了
            region["points_2d"] = points_2d.reshape(-1, 2).astype(np.int32)
            # print(f"[UPDATE 2D PROJECTION] Region: \n{region_name}, 3D points: {region['points_3d']}")
            # print(f"[UPDATE 2D PROJECTION] Projected 2D points: \n{region['points_2d']}")
        # print(self.regions.items())
            
    def update_calibration(self, new_calibrate_path):
            with open(new_calibrate_path) as f:
                data = json.load(f)
                self.rvec = np.array(data["rvec"])
                self.tvec = np.array(data["tvec"])
            self.update_2d_projection()
            
    def update_camera_matrix(self, new_camera_matrix):
        self.camera_matrix = new_camera_matrix
        self.update_2d_projection()
    
    # 高度获取，通过判断img_point的落点是在哪个区域来获取对应区域的region_height
    def get_height(self, img_point):
        img_point = np.array(img_point)
        for region_name, region in self.regions.items():
            points_2d = region["points_2d"]
            if points_2d is None:
                continue
            # print(f"[HEIGHT] Region: {region_name}, cv2.pointPolygonTest result: {cv2.pointPolygonTest(region['points_2d'], (float(img_point[0]), float(img_point[1])), False)}")
            # 判断点是否在多边形内部(如果>0，就说明点在多边形内部; =0说明点在多边形边界上; <0说明点在多边形外部)
            if cv2.pointPolygonTest(points_2d, (float(img_point[0]), float(img_point[1])), False) >= 0:
                # print(f"[HEIGHT] Region: {region_name}, height: {region['height']}")
                return region["height"]
        return 0.0
    
    # 位置解算, 利用相机外参进行平面映射
    # 默认己方为红方，最后输出的时候，如果是蓝方再转换
    # 红方补给区为地图坐标系的原点，场地长边向蓝方为X轴正方向，场地短边向红方停机坪为Y轴正方向
    def parse(self, img_point):     
        # 把图像坐标系里的点位，通过内参矩阵K(即self.camera_matrix)和外参[R|t](即rvec + tvec，旋转+平移)投影到世界坐标系
        # 用相机投影模型的反向投影方法（因为相机投影模型是从世界到图像坐标系的）
        # 需要从图像坐标(u, v)得到世界坐标(X_w, Y_w)，再高度通过get_height获得补偿的高度height
        cx = self.camera_matrix[0, 2]   # 主点在x轴方向
        cy = self.camera_matrix[1, 2]   # 主点在y轴方向
        fx = self.camera_matrix[0, 0]   # 焦距在x轴方向
        fy = self.camera_matrix[1, 1]   # 焦距在y轴方向
        
        # 去畸变（保留原始内参）
        pt_dist = np.array([img_point], dtype=np.float32)
        pt_undist = cv2.undistortPoints(pt_dist, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        u, v = pt_undist[0][0]
        
        # 计算像素坐标系的归一化坐标（等价于 x_n = (u - cx)/fx, y_n = (v - cy)/fy）
        x_n = (u - cx) / fx
        y_n = (v - cy) / fy
        # print("--------------------------")
        # print(f"原图像坐标系中的坐标: {img_point}")
        # print(f"去畸变后的像素坐标: {pt_undist[0][0]}")
        # print("--------------------------")
        
        # 使用去畸变之后的图像坐标系中的像素值，即2d点pt_undist: (u_bar, v_bar)和之前预定在RM2024_Points.yaml中圈的范围做对比，找到属于的范围圈的高度height
        # 这个height就是世界坐标系里的Z_w
        # 因为RM2024_Points.yaml中的坐标值是世界坐标系的，所以需要把里面那个世界坐标系的点暂时转换成像素坐标系的坐标点
        height = self.get_height(img_point)
        # print(f"height: {height}")
        if height > 0.79:
            return np.array([19.322, -1.915, height], dtype=np.float32)
        
        # 相机内参K就是self.camera_matrix
        # 求相机外参，即旋转矩阵和平移向量
        R, _ = cv2.Rodrigues(self.rvec)     # 旋转向量 变成 旋转矩阵
        t = self.tvec.flatten()             # 平移向量 简化成 一维数组的平移向量
        
        # PnP解算对应的光线方向向量
        # 使用归一化坐标作为方向向量
        ray_dir = np.array([x_n, y_n, 1.0])
        
        # 转换到世界坐标系
        ray_dir_world = R.T @ ray_dir
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)  # 标准化
        
        # 相机中心点在世界坐标系中的位置
        C = -R.T @ t
        
        # 计算射线与平面(z=height)相交
        # 平面方程: z = height
        # 射线方程: P = C + t * ray_dir_world
        # 求解t: C[2] + t * ray_dir_world[2] = height
        
        # 检查射线是否平行于平面(z=height)
        if abs(ray_dir_world[2]) < 1e-6:
            print("警告: 射线几乎平行于平面，无法计算准确的交点")
            return np.array([128, 0, height], dtype=np.float32)  # 返回一个明显错误的值
        
        # 计算射线参数t
        t_param = (height - C[2]) / ray_dir_world[2]
        
        # 计算交点
        P_w = C + t_param * ray_dir_world
        
        # 反向投影验证解算的世界坐标系是不是对的
        # 如果是对的就会输出原来输入的img_point
        obj_pt = np.array([[P_w[0], P_w[1], height]], dtype=np.float32)
        img_pt_proj, _ = cv2.projectPoints(obj_pt, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        # print("反向投影坐标:", img_pt_proj)
        
        # 计算投影误差，并进行误差过滤
        error = np.linalg.norm(img_pt_proj[0][0] - pt_undist[0][0])
        # print(f"投影误差: {error}")
        
        # 过滤异常数据
        if error > 100:  # 误差大于阈值视为异常
            return np.array([0, 0, 0], dtype=np.float32)
        
        # 确保z值为height  
        P_w[2] = height
        
        return np.array([P_w[0], P_w[1], height], dtype=np.float32)
    
    # 把这个点的像素点位区域给画出来，便于直观的理解输入的img_point到底跑到哪个区域去了
    def draw_regions(self, frame, thickness=2):
        colors = {
            "Middle_Line": (0, 255, 255),
            "Self_Ring_High": (255, 0, 0),
            "Enemy_Ring_High": (255, 0, 0),
            "Enemy_Buff": (0, 0, 255),
            "Self_Left_High": (255, 255, 255),
            "Enemy_Left_High": (255, 255, 255)
        }
        
        for region_name, region in self.regions.items():
            points_2d = region["points_2d"]
            if points_2d is None:
                continue
            
            # 换成整型点值比较好画
            points = np.array(points_2d, dtype=np.int32)
            # 获取该区域的color，没有的话就用绿色
            color = colors.get(region_name, (0, 255, 0))
            cv2.polylines(frame, [points], True, color, thickness)
        return frame

if __name__ == "__main__":
    locator = Location()
    # img_width, img_height = 4096, 3000      # Hikvision CH-120-10UC 的分辨率是 4096×3000， 东大的，RM2024_Points.yaml是他们的(但应该没关系？)
    # img_width, img_height = 1920, 1080      # Hikvision MV-CS020-10UC 的分辨率是1920×1080，我们的老相机
    img_width, img_height = 3072, 2048      # Hikvision MV-CS060-10UC-PRO 的分辨率是3072×2048, 我们的新相机
    frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    frame = locator.draw_regions(frame, thickness=2)
    
    # test_point = [1463, 609]          # 在1080*1920中的R3
    test_point = [2340, 1156]       # R3 这些是在3072*2048中的
    # test_point = [1194, 974]        # R4
    # test_point = [1383, 854]        # R2
    # test_point = [2528, 847]        # R1
    # test_point = [1580, 446]          # 在1080*1920中的R1
    
    # scale_x = 1920 / 3072
    # scale_y = 1080 / 2048
    # test_point1 = [int(test_point[0] * scale_x), int(test_point[1] * scale_y)]
    # print(test_point1)
    
    # 在测试点位置添加明显的标记
    cv2.circle(frame, tuple(test_point), 5, (0, 0, 255), -1)
    # 在测试点旁边显示坐标
    cv2.putText(frame, f"Test Point: {test_point}", (test_point[0]+10, test_point[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    world_coord = locator.parse(test_point)
    # 因为opencv的是右手系，dji的是左手系，所以解算出来的要转换一下（即只要-y即可），目前先默认为右手系
    # 红方补给区为地图坐标系的原点，场地长边向蓝方为X轴正方向，场地短边向红方停机坪为Y轴正方向
    # 测试图片及其坐标放的是蓝方的视角（在实际地图上要转换），所以要用28-x值，-y+15值
    # 如果视角是红方的，那么就只要-y值即可
    
    # R3解算出来应该是X=19.32, Y=9.61， Z=0.0
    # R4解算出来应该是X=18.05, Y=4.74， Z=0.0
    # R2解算出来应该是X=13.32, Y=5.87， Z=0.0
    # R1解算出来应该是X=14.15, Y=12.92, z=0.0

    print(f"解算结果(原): X={world_coord[0]:.2f}, Y={world_coord[1]:.2f}, Z={world_coord[2]:.2f}")
    print(f"解算结果: X={28-world_coord[0]:.2f}, Y={-world_coord[1]:.2f}")     
    
    # 在图像上显示解算结果
    result_text = f"解算结果: X={world_coord[0]:.2f}, Y={world_coord[1]:.2f}"
    cv2.putText(frame, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 0), 2, cv2.LINE_AA)
    
    # print(frame.shape)
    # target_weight = int(frame.shape[1] / 1.333334)
    # target_height = int(frame.shape[0] / 1.46484375)
    # frame1 = cv2.resize(frame, (target_weight+1, target_height))
    # print(frame1.shape)
    
    frame1 = cv2.resize(frame, (1500, 1000))
    cv2.imshow("Regions and Test Point", frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()