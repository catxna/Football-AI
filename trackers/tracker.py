import sys
import argparse
import os
import cv2
import torch
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from scipy.spatial.distance import cosine

from utils.bbox_utils import get_bbox_width, calculate_iou, get_center_of_bbox
sys.path.append('../')
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


class FastReIDWrapper:
    def __init__(self, config_path, model_path):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def get_features(self, image):
        if image is None or image.size == 0:
            print("Empty crop detected. Skipping feature extraction.")
            return None
            
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            if h < 16 or w < 16:
                print(f"Crop too small for ReID: {w}x{h}. Skipping feature extraction.")
                return None
                
            if h < 32 or w < 32:
                scale = max(32 / h, 32 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
            
        if image.shape[2] < 16 or image.shape[3] < 16:
            print(f"Tensor too small for ReID: {image.shape}. Skipping feature extraction.")
            return None
            
        image = image.to(self.predictor.model.device)
        
        try:
            features = self.predictor(image)
            return features
        except ValueError as e:
            if "Expected more than 1 spatial element" in str(e):
                print(f"ReID model spatial error with size {image.shape}. Skipping.")
                return None
            raise

    def compute_similarity(self, features1, features2):
        features1 = features1.flatten()
        features2 = features2.flatten()
        return 1 - cosine(features1, features2)


class Tracker:
    def __init__(self, model_path, reid_config_path=None, reid_model_path=None):
        self.model = YOLO(model_path)
        
        # Configure ByteTracker for players, refs, and balls
        args = argparse.Namespace()
        args.track_thresh = 0.2  # Lowered to include more ball detections
        args.match_thresh = 0.5   # Lowered for small objects like balls
        args.track_buffer = 120   # Increased to keep tracks longer
        args.mot20 = False
        args.min_box_area = 0     # Set to include small detections
        self.tracker = BYTETracker(args)
        
        self.reid_model = None
        if reid_config_path and reid_model_path:
            self.reid_model = FastReIDWrapper(reid_config_path, reid_model_path)

        self.previous_tracks = {"Player": {}, "Ball": {}, "Ref": {}}
        self.appearance_features = defaultdict(list)
        self.track_classes = {}   # Store class for each track
        self.referee_ids = set()
        self.main_ball_id = None
        
        self.debug_info = {
            "ball_detections": 0,
            "ball_tracks": 0
        }

    def detect_frames(self, frames):
        detections = []
        with torch.no_grad():
            for frame in frames:
                detection = self.model.predict(
                    frame, 
                    conf=0.05,
                    verbose=False,
                    classes=[0, 1, 2]
                )
                detections.extend(detection)
        return detections

    def add_or_update_track_id(self, proposed_track_id, new_features, threshold, object_type):
        best_match_id = None
        best_similarity = -1.0
        for existing_id, features_list in self.appearance_features.items():
            if not features_list:
                continue
            all_feats = torch.stack(features_list, dim=0)
            avg_feat = all_feats.mean(dim=0)
            sim = self.reid_model.compute_similarity(new_features, avg_feat)
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = existing_id
        effective_threshold = threshold if object_type != "Ball" else threshold * 0.8
        if best_similarity >= effective_threshold:
            self.appearance_features[best_match_id].append(new_features)
            return best_match_id
        self.appearance_features[proposed_track_id].append(new_features)
        return proposed_track_id

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        print(f"Processing {len(frames)} frames for object tracking...")
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print(f"Loading cached tracking data from {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"Player": [], "Ball": [], "Ref": []}

        self.debug_info["ball_detections"] = 0
        self.debug_info["ball_tracks"] = 0
        
        if not read_from_stub:
            self.referee_ids = set()
            self.track_classes = {}  # Reset track classes for new video

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            frame_h, frame_w = frames[0].shape[:2]
            frame = frames[frame_num]
            detection_list = []

            tracks["Player"].append({})
            tracks["Ball"].append({})
            tracks["Ref"].append({})
            
            for det in detection.boxes:
                bbox = det.xyxy[0].tolist()
                conf = det.conf[0].item() 
                class_id = int(det.cls[0].item())
                class_name = class_names[class_id]
                detection_list.append([*bbox, conf, class_id])
                
                if class_name == "Ball":
                    self.debug_info["ball_detections"] += 1
            
            if detection_list:
                detection_tensor = torch.tensor(detection_list, dtype=torch.float32)
                detection_with_tracks = self.tracker.update(
                    detection_tensor, (frame_h, frame_w), (frame_h, frame_w)
                )

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection.tlbr.tolist()
                    track_id = frame_detection.track_id
                    
                    if track_id not in self.track_classes:
                        max_iou = 0
                        best_class_id = None
                        for det in detection_list:
                            det_bbox = det[:4]
                            iou = calculate_iou(bbox, det_bbox)
                            if iou > max_iou:
                                max_iou = iou
                                best_class_id = int(det[5])
                        if max_iou > 0.1:  # Threshold for class assignment
                            self.track_classes[track_id] = class_names[best_class_id]
                        else:
                            continue  # Skip tracks without sufficient overlap
                    
                    class_name = self.track_classes[track_id]
                    
                    if self.reid_model:
                        x1, y1, x2, y2 = int(max(0, bbox[0])), int(max(0, bbox[1])), \
                                        int(min(frame.shape[1], bbox[2])), int(min(frame.shape[0], bbox[3]))
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            width, height = x2-x1, y2-y1
                            if width >= 16 and height >= 16 and crop.size > 0:
                                features = self.reid_model.get_features(crop)
                                if features is not None:
                                    track_id = self.add_or_update_track_id(
                                        track_id, features, threshold=0.985, object_type=class_name
                                    )
                    
                    if track_id in self.referee_ids or class_name == "Ref":
                        self.referee_ids.add(track_id)
                        tracks["Ref"][frame_num][track_id] = {"bbox": bbox}
                    elif class_name == "Player":
                        tracks["Player"][frame_num][track_id] = {"bbox": bbox}
                    elif class_name == "Ball":
                        tracks["Ball"][frame_num][track_id] = {"bbox": bbox}
                        self.debug_info["ball_tracks"] += 1

        ball_tracks = self.post_process_ball_tracks(tracks["Ball"])
        tracks["Ball"] = ball_tracks

        print(f"Debug: Detected {self.debug_info['ball_detections']} balls, Tracked {self.debug_info['ball_tracks']} balls")
        print(f"Debug: Identified {len(self.referee_ids)} referees with IDs: {self.referee_ids}")
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def post_process_ball_tracks(self, ball_positions):
        track_counts = {}
        for frame_balls in ball_positions:
            for track_id in frame_balls:
                track_counts[track_id] = track_counts.get(track_id, 0) + 1
        
        sorted_tracks = sorted(track_counts.items(), key=lambda x: x[1], reverse=True)
        processed_positions = [{} for _ in range(len(ball_positions))]
        
        if not sorted_tracks:
            print("Warning: No ball tracks found to process!")
            return processed_positions
            
        main_track_id = sorted_tracks[0][0]
        self.main_ball_id = main_track_id
        
        print(f"Selected main ball track ID: {main_track_id} with {track_counts[main_track_id]} detections")
        if len(sorted_tracks) > 1:
            print(f"Other ball track candidates: {[(tid, count) for tid, count in sorted_tracks[1:5]]}")
        
        detections = []
        for i, frame_balls in enumerate(ball_positions):
            if main_track_id in frame_balls:
                detections.append((i, frame_balls[main_track_id]["bbox"]))
        
        if len(detections) < 2:
            return ball_positions
            
        for frame_idx, bbox in detections:
            processed_positions[frame_idx][main_track_id] = {"bbox": bbox, "is_detection": True}
            
        for i in range(len(detections) - 1):
            start_frame, start_bbox = detections[i]
            end_frame, end_bbox = detections[i + 1]
            
            if end_frame - start_frame <= 1:
                continue
                
            start_center = [(start_bbox[0] + start_bbox[2])/2, (start_bbox[1] + start_bbox[3])/2]
            end_center = [(end_bbox[0] + end_bbox[2])/2, (end_bbox[1] + end_bbox[3])/2]
            
            start_width = start_bbox[2] - start_bbox[0]
            start_height = start_bbox[3] - start_bbox[1]
            end_width = end_bbox[2] - end_bbox[0]
            end_height = end_bbox[3] - end_bbox[1]
            
            gap_length = end_frame - start_frame
            distance = np.sqrt((end_center[0] - start_center[0])**2 + (end_center[1] - start_center[1])**2)
            
            max_speed = 50
            if gap_length > 30 or (distance / gap_length) > max_speed:
                continue
                
            for frame in range(start_frame + 1, end_frame):
                t = (frame - start_frame) / (end_frame - start_frame)
                x = start_center[0] + t * (end_center[0] - start_center[0])
                y = start_center[1] + t * (end_center[1] - start_center[1])
                
                if gap_length > 5:
                    h = min(30, gap_length * 1.5)
                    y = y - 4 * h * t * (1 - t)
                
                width = start_width + t * (end_width - start_width)
                height = start_height + t * (end_height - start_height)
                
                bbox = [x - width/2, y - height/2, x + width/2, y + height/2]
                edge_distance = min(frame - start_frame, end_frame - frame)
                confidence = 1.0 - edge_distance / (gap_length / 2)
                
                processed_positions[frame][main_track_id] = {
                    "bbox": bbox,
                    "is_detection": False,
                    "confidence": confidence
                }
        
        ball_frame_count = sum(1 for frame in processed_positions if main_track_id in frame)
        print(f"Final ball track: {ball_frame_count}/{len(processed_positions)} frames ({ball_frame_count/len(processed_positions)*100:.1f}%)")
        
        return processed_positions

    def draw_bbox(self, frame, bbox, color, track_id=None, object_type=None):
        if object_type == "Ball":
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int(((bbox[1] + bbox[3]) / 2) - 17)
            triangle_height = 12
            triangle_width = 6

            tip = (x_center, y_center)
            left = (x_center - triangle_width, y_center - triangle_height)
            right = (x_center + triangle_width, y_center - triangle_height)

            triangle_cnt = np.array([left, right, tip])
            cv2.drawContours(frame, [triangle_cnt], 0, (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
            cv2.drawContours(frame, [triangle_cnt], 0, color, thickness=-1, lineType=cv2.LINE_AA)
            return frame

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if object_type != "Ref":
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(
                frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            for track_id, player in tracks["Player"][frame_num].items():
                frame = self.draw_bbox(frame, player["bbox"], (0, 255, 0), track_id, "Player")
            
            for track_id, ref in tracks["Ref"][frame_num].items():
                frame = self.draw_bbox(frame, ref["bbox"], (255, 255, 255), track_id, "Ref")
            
            for track_id, ball in tracks["Ball"][frame_num].items():
                frame = self.draw_bbox(frame, ball["bbox"], (0, 0, 255), track_id, "Ball")
            
            output_video_frames.append(frame)
        return output_video_frames