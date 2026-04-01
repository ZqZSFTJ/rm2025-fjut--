import cv2
import numpy as np
import yaml
from TRTEngine import TRTEngine

class YOLOv5Detector:
    def __init__(self, engine_path, yaml_path, conf_threshold=0.25, iou_threshold=0.45, max_det=1000, agnostic_nms=False):
        self.engine = TRTEngine(engine_path)
        self.yaml_path = yaml_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms

        self.input_name = self.engine.inputs[0].name
        self.input_shape = self.engine.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        # self.output_names = [output.name for output in self.engine.outputs]
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.class_names = data["names"]
            self.nc = data["nc"]
        
    def preprocess(self, image):
        image_height, image_width = image.shape[:2]
        scale = min(self.input_height / image_height, self.input_width / image_width)
        new_height = int(image_height * scale)
        new_width = int(image_width * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        padded_image = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        dh = (self.input_height - new_height) // 2
        dw = (self.input_width - new_width) // 2
        padded_image[dh:dh+new_height, dw:dw+new_width] = resized_image
        
        padded_image = padded_image.astype(np.float32) / 255.0
        padded_image = np.transpose(padded_image, (2, 0, 1))
        padded_image = np.expand_dims(padded_image, axis=0)
        
        assert padded_image.shape == (1, 3, self.input_height, self.input_width)
        return padded_image, (scale, dw, dh, image_width, image_height)
    
    def postprocess(self, outputs, params):
        scale, dw, dh, orig_width, orig_height = params
        preditions = np.squeeze(outputs[0]).astype(np.float32)
        if preditions.ndim == 1:
            preditions = np.expand_dims(preditions, 0)
            
        object_conf = preditions[:, 4]
        preditions = preditions[object_conf > self.conf_threshold]
        if len(preditions) == 0:
            return []
        
        class_confs = preditions[:, 5:]
        class_ids = np.argmax(class_confs, axis=1)
        class_scores = class_confs[np.arange(len(preditions)), class_ids]
        total_scores = object_conf[object_conf > self.conf_threshold] * class_scores
        
        valid_mask = total_scores > self.conf_threshold
        preditions = preditions[valid_mask]
        class_ids = class_ids[valid_mask]
        total_scores = total_scores[valid_mask]
        
        boxes = preditions[:, :4]
        boxes[:, 0] = (boxes[:, 0] - dw) / scale    # cx
        boxes[:, 1] = (boxes[:, 1] - dh) / scale    # cy
        boxes[:, 2] /= scale                        # width
        boxes[:, 3] /= scale                        # height
        
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        x1 = np.clip(x1, 0, orig_width)
        y1 = np.clip(y1, 0, orig_height)
        x2 = np.clip(x2, 0, orig_width)
        y2 = np.clip(y2, 0, orig_height)
        
        detections = []
        for i in range(len(x1)):
            detections.append([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), float(total_scores[i]), int(class_ids[i])])
        
        if len(detections) == 0 or not detections:
            return []
        

        final_detections = []
        if self.agnostic_nms:
            boxes = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in detections]
            scores = [d[4] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            
            if indices is not None:
                for i in indices.flatten():
                    final_detections.append(detections[i])
        else:
            detections_by_class = {}
            for det in detections:
                class_id = det[5]
                if class_id not in detections_by_class:
                    detections_by_class[class_id] = []
                detections_by_class[class_id].append(det)
            
            for class_id in detections_by_class:
                class_dets = detections_by_class[class_id]
                boxes = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in class_dets]
                scores = [d[4] for d in class_dets]
                indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
                if indices is not None:
                    for i in indices.flatten():
                        final_detections.append(class_dets[i])
        
        final_detections.sort(key=lambda x: x[4], reverse=True)
                    
        return final_detections[:self.max_det]
        
    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed_image, params = self.preprocess(image_rgb)
        outputs = self.engine.infer(preprocessed_image)
        return self.postprocess(outputs, params)

    @staticmethod
    def xywh2xyxy(x):
        y = np.copy(x)
        y[:,0] = x[:,0] - x[:,2]/2  # x1
        y[:,1] = x[:,1] - x[:,3]/2  # y1 
        y[:,2] = x[:,0] + x[:,2]/2  # x2
        y[:,3] = x[:,1] + x[:,3]/2  # y2
        return y
    
    def nms(self, boxes, scores):
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.conf_threshold,
            self.iou_threshold
        )
        return indices.flatten() if len(indices) > 0 else []