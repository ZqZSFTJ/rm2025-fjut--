import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import cv2
from TRTEngine3 import TRTEngine


class Track:
    def __init__(self, detection, track_id, kalman_filter, n_init=3, max_age=30):
        self.track_id = track_id
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = kalman_filter.initiate(detection)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 'tentative'
        self.features = []
        self.n_init = n_init
        self.max_age = max_age
        self.class_name = detection[-1]
        
    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection, feature):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, detection)
        self.features.append(feature)
        self.time_since_update = 0
        self.hits += 1
        if self.hits >= self.n_init:
            self.state = 'confirmed'
        self.class_name = detection[-1]
        
    def mark_missed(self):
        if self.state == 'confirmed' and self.time_since_update > self.max_age:
            self.state = 'deleted'
        elif self.state == 'tentative':
            self.state = 'deleted'
        
    def get_bbox(self):
        cx, cy, gamma, h = self.mean[:4]
        w = gamma * h
        return [
            int(cx - w/2),      # x1
            int(cy - h/2),      # y1
            int(cx + w/2),      # x2
            int(cy + h/2)       # y2
        ]

class KalmanFilter:
    def __init__(self):
        self._motion_mat = np.eye(8, 8)
        self._update_mat = np.eye(4, 8)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, detection):
        x1, y1, x2, y2 = detection[:4]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        gamma = w / h
        return np.array([cx, cy, gamma, h, 0, 0, 0, 0]), np.eye(8) * 1e-4
                    
    # 卡尔曼滤波器 预测阶段
    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def update(self, mean, covariance, detection):
        x1, y1, x2, y2 = detection[:4]
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        gamma = w / h
        measurement = np.array([x1 + w/2, y1 + h/2, gamma, h])
        
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.lstsq(projected_cov.T, np.dot(covariance, self._update_mat.T).T, rcond=None)[0].T
        
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance
        
    # 匈牙利匹配 协方差矩阵映射
    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        
        return mean, covariance
        

class DeepSORTTracker:
    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.2, trt_engine_path="reid.trt"):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.kalman_filter = KalmanFilter()
        self.tracks = []
        self.next_id = 1
        self.track_history = defaultdict(deque)
        self.reid_engine = TRTEngine(trt_engine_path)
        self.input_shape = self._get_engine_input_shape()
    
    def _get_engine_input_shape(self):
        for binding in self.reid_engine.engine:
            if self.reid_engine.engine.binding_is_input(binding):
                return tuple(self.reid_engine.engine.get_binding_shape(binding))
        return (3, 256, 128)
    
    def _preprocess(self, crop):
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(crop)
        
        resized = cv2.cuda.resize(gpu_img, (128, 256))
        converted = cv2.cuda.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = cv2.cuda.subtract(
            cv2.cuda.multiply(converted, 1.0/255.0), 
            (0.485, 0.456, 0.406),
            dtype=cv2.CV_32F
        )
        std_dev = cv2.cuda.multiply(normalized, (1/0.229, 1/0.224, 1/0.225))
        return std_dev.download().transpose(2, 0, 1)
    
    def update(self, detections, frame):
        self.perf_mon.start('frame')
        self.perf_mon.start('feature_extract')
        features = [self._extract_feature(frame, d) for d in detections]
        self.perf_mon.end('feature_extract')
        for track in self.tracks:
            track.predict()
        
        confirmed_tracks = [t for t in self.tracks if t.state == 'confirmed']
        unconfirmed_tracks = [t for t in self.tracks if t.state != 'confirmed']
        
        matches, unmatched_tracks, unmatched_detections = self._match(confirmed_tracks, detections, features)
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], features[det_idx])
        for det_idx in unmatched_detections:
            self._create_new_track(detections[det_idx], features[det_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        self.tracks = [t for t in self.tracks if t.state != 'deleted']
        active_tracks = []
        for track in self.tracks:
            if track.state == 'confirmed':
                bbox = track.get_bbox()
                self.track_history[track.track_id].append((bbox, track.class_name))
                if len(self.track_history[track.track_id]) > 30:
                    self.track_history[track.track_id].popleft()
                active_tracks.append(track)
        
        self.perf_mon.end('frame')
        self.perf_mon.counters['frame'] += 1
        if self.perf_mon.counters['frame'] % 100 == 0:
            self.perf_mon.print_stats()
              
        return active_tracks

    def _match(self, tracks, detections, features):
        cost_matrix = self._cosine_distance(tracks, features)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_cosine_distance:
                matches.append((i, j))
                
        unmatched_tracks = set(range(len(tracks))) - set(row_ind)
        unmatched_detections = set(range(len(detections))) - set(col_ind)
        
        return matches, list(unmatched_tracks), list(unmatched_detections)
    
    def _cosine_distance(self, tracks, features):
        cost_matrix = np.zeros((len(tracks), len(features)))
        for i, track in enumerate(tracks):
            if len(track.features) > 0:
                track_feature = np.mean(track.features, axis=0)
                cost_matrix[i, :] = np.dot(features, track_feature) / (np.linalg.norm(features, axis=1) * np.linalg.norm(track_feature))
            else:
                cost_matrix[i, :] = 0.5
        return (1 - cost_matrix)
        
    def _create_new_track(self, detection, feature):
        new_track = Track(detection, self.next_id, self.kalman_filter, self.n_init, self.max_age) 
        new_track.features.append(feature)
        self.tracks.append(new_track)
        self.next_id += 1
    
    # 在ReID模型的基础上特征提取,tensorrt加速
    def _extract_feature(self, frame, detection):
        x1, y1, x2, y2 = detection[:4]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            return np.random.randn(self.input_shape[0]).astype(np.float32)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.random.randn(self.input_shape[0]).astype(np.float32)
        processed = self._preprocess(crop)
        feature = self.reid_engine(processed[np.newaxis, (1, 3, 256, 128)])
        return feature.flatten().astype(np.float32)