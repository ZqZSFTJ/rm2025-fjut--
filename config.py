import numpy as np

class Config:
    COLOR = "R"     # B: Blue; R:Red
    SELECT_MODE = 'test'    # test: 测试；hik: 海康相机
    PORT = "COM3"
    
    # 这是我们自己的新相机MV-CS060-10UC-PRO的相机内参
    CAMERA_MATRIX = np.array([
        [3359.900898,    0.      ,  1492.082497],
        [   0.      , 3344.312184,  1009.967128],
        [   0.      ,    0.      ,    1.       ]], dtype=np.float32)
    DIST_COEFFS = np.array([-0.113551, 0.168385, 0.000048, -0.001705, 0.000000], dtype=np.float32)
    
    ARMOR_MODEL_PATH = "model//armor.engine"
    CAR_MODEL_PATH = "model//car.engine"
    ARMOR_YAML_PATH = "yaml//armor.yaml"
    CAR_YAML_PATH = "yaml//car.yaml"
    
    MAP_SIZE = (28.0, 15.0)
    
    HIK_CONFIG = {
        'sn': 'DA6214861',
        'exposure': 10000.0,
        'gain': 15.0,
        'frame_rate': 210.0,
        'rotate_180': False,
        'log_level': "INFO"
    }
    
    # VIDEO_PATH = 'image\\test20250512.mp4'
    #VIDEO_PATH = "E:\\RM_INFO\\20250513_181011\\raw_video\\raw.mp4"
    #VIDEO_PATH = "D:\\fjut_lidar\\image\\test.mp4"
    VIDEO_PATH = r"E:\fjut_radar\image\test20250512.mp4"
    IMG_PATH = r"E:\fjut_radar\image\test2025.jpg"