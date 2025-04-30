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
from utils import get_center_of_bbox


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
            image = torch.from_numpy(image).float().permute(2, 0, 1)
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.predictor.model.device)
        features = self.predictor(image)
        return features

    def compute_similarity(self, features1, features2):
        features1 = features1.flatten()
        features2 = features2.flatten()
        return 1 - cosine(features1, features2)


class Tracker:
    def __init__(self, model_path, reid_config_path=None, reid_model_path=None):
        self.model = YOLO(model_path)
        args = argparse.Namespace()
        args.track_thresh = 0.4
        args.match_thresh = 0.95
        args.track_buffer = 120
        args.mot20 = False
        self.tracker = BYTETracker(args)

        self.reid_model = None
        if reid_config_path and reid_model_path:
            self.reid_model = FastReIDWrapper(reid_config_path, reid_model_path)

        self.previous_tracks = {"Player": {}, "Ball": {}, "Ref": {}}
        self.appearance_features = defaultdict(list)
        
        # Store IDs of referees to ensure they stay as refs throughout the video
        self.referee_ids = set()
        
        # For debugging
        self.debug_info = {
            "ball_detections": 0,
            "ball_tracks": 0
        }

    def detect_frames(self, frames):
        detections = []
        with torch.no_grad():
            for frame in frames:
                detection = self.model.predict(frame, conf=0.15)
                detections.extend(detection)
        return detections

    def add_or_update_track_id(self, proposed_track_id, new_features, threshold):
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
        if best_similarity >= threshold:
            self.appearance_features[best_match_id].append(new_features)
            return best_match_id
        self.appearance_features[proposed_track_id].append(new_features)
        return proposed_track_id

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"Player": [], "Ball": [], "Ref": []}

        # Reset debug counters
        self.debug_info["ball_detections"] = 0
        self.debug_info["ball_tracks"] = 0
        
        # Clear referee IDs if we're starting a new detection
        if not read_from_stub:
            self.referee_ids = set()

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            frame_h, frame_w = frames[0].shape[:2]
            frame = frames[frame_num]
            detection_list = []

            # Initialize empty dictionaries for this frame
            tracks["Player"].append({})
            tracks["Ball"].append({})
            tracks["Ref"].append({})
            
            # Process detections before tracking
            for det in detection.boxes:
                bbox = det.xyxy[0].tolist()
                conf = det.conf[0].item()
                class_id = int(det.cls[0].item())
                detection_list.append([*bbox, conf, class_id])
                
                # Special handling for balls - they often don't track well
                if class_names[class_id] == "Ball":
                    self.debug_info["ball_detections"] += 1
                    # Use a fixed ID (999) for all balls to ensure they're always drawn
                    tracks["Ball"][frame_num][999] = {"bbox": bbox}

            # Only use tracker for players and refs
            if detection_list:
                detection_tensor = torch.tensor(detection_list, dtype=torch.float32)
                detection_with_tracks = self.tracker.update(
                    detection_tensor, (frame_h, frame_w), (frame_h, frame_w)
                )

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection.tlbr.tolist()
                    track_id = frame_detection.track_id
                    cx1 = (bbox[0] + bbox[2]) / 2
                    cy1 = (bbox[1] + bbox[3]) / 2

                    class_id = None
                    min_dist = float('inf')
                    for det in detection_list:
                        det_bbox = det[:4]
                        cx2 = (det_bbox[0] + det_bbox[2]) / 2
                        cy2 = (det_bbox[1] + det_bbox[3]) / 2
                        dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            class_id = int(det[5])

                    if class_id is None:
                        continue

                    class_name = class_names[class_id]
                    if class_name == "Ball":
                        self.debug_info["ball_tracks"] += 1
                        # Use fixed ID for balls from tracker too
                        tracks["Ball"][frame_num][999] = {"bbox": bbox}
                    else:
                        if self.reid_model and class_name != "Ball":
                            crop = frames[frame_num][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                            features = self.reid_model.get_features(crop)
                            if features is not None:
                                track_id = self.add_or_update_track_id(track_id, features, threshold=0.985)
                        
                        # Check if this track_id is already a known referee
                        if track_id in self.referee_ids:
                            # If it's a known referee, always classify as referee regardless of current class
                            tracks["Ref"][frame_num][track_id] = {"bbox": bbox}
                        elif class_name == "Ref":
                            # Add this ID to the referee set for future frames
                            self.referee_ids.add(track_id)
                            tracks["Ref"][frame_num][track_id] = {"bbox": bbox}
                        elif class_name == "Player":
                            tracks["Player"][frame_num][track_id] = {"bbox": bbox}

        # Always interpolate ball positions to ensure continuity
        ball_tracks = self.interpolate_ball_positions(tracks["Ball"])
        tracks["Ball"] = ball_tracks

        print(f"Debug: Detected {self.debug_info['ball_detections']} balls, Tracked {self.debug_info['ball_tracks']} balls")
        print(f"Debug: Identified {len(self.referee_ids)} referees with IDs: {self.referee_ids}")

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_bbox(self, frame, bbox, color, track_id=None, object_type=None):
        # Special drawing for ball
        if track_id == 999:  # Ball has fixed ID 999
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int(((bbox[1] + bbox[3]) / 2) - 17)
            triangle_height = max(10, int((bbox[3] - bbox[1]) * 0.6))
            triangle_width = triangle_height // 2

            # Define triangle points
            tip = (x_center, y_center)  # Tip pointing down to the ball
            left = (x_center - triangle_width, y_center - triangle_height)
            right = (x_center + triangle_width, y_center - triangle_height)

            triangle_cnt = np.array([left, right, tip])

            # Draw black outline
            cv2.drawContours(frame, [triangle_cnt], 0, (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
            # Draw filled colored triangle on top
            cv2.drawContours(frame, [triangle_cnt], 0, color, thickness=-1, lineType=cv2.LINE_AA)

            return frame

        # Regular bbox drawing for players and refs
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

        # Only draw ID for players, not for referees
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
            
            # Draw players
            for track_id, player in tracks["Player"][frame_num].items():
                frame = self.draw_bbox(frame, player["bbox"], (0, 255, 0), track_id, "Player")
            
            # Draw referees
            for track_id, ref in tracks["Ref"][frame_num].items():
                frame = self.draw_bbox(frame, ref["bbox"], (255, 255, 255), track_id, "Ref")
            
            # Draw balls
            for track_id, ball in tracks["Ball"][frame_num].items():
                frame = self.draw_bbox(frame, ball["bbox"], (0, 0, 255), track_id, "Ball")
            
            output_video_frames.append(frame)
        return output_video_frames

    def interpolate_ball_positions(self, ball_positions):
        # Special handling for our fixed ball ID (999)
        track_id = 999
        
        # Extract positions for ball track
        positions = []
        frame_indices = []
        
        for i, frame_balls in enumerate(ball_positions):
            if track_id in frame_balls and "bbox" in frame_balls[track_id]:
                bbox = frame_balls[track_id]["bbox"]
                if len(bbox) == 4:  # Ensure valid bbox
                    positions.append(bbox)
                    frame_indices.append(i)
        
        # If we have no valid ball positions, return the original 
        if not positions:
            print("Warning: No valid ball positions found to interpolate")
            return ball_positions
        
        # Create the output structure with same shape as input
        interpolated_ball_positions = [{} for _ in range(len(ball_positions))]
        
        # First pass: Add all confirmed detections
        for i, bbox in zip(frame_indices, positions):
            interpolated_ball_positions[i][track_id] = {"bbox": bbox, "is_detection": True}
        
        # Second pass: Fill gaps with ballistic trajectories
        for gap_start_idx in range(len(frame_indices) - 1):
            start_frame = frame_indices[gap_start_idx]
            end_frame = frame_indices[gap_start_idx + 1]
            
            # Skip if there's no gap
            if end_frame - start_frame <= 1:
                continue
                
            start_bbox = positions[gap_start_idx]
            end_bbox = positions[gap_start_idx + 1]
            
            # Get centers for start and end positions
            start_center = [(start_bbox[0] + start_bbox[2])/2, (start_bbox[1] + start_bbox[3])/2]
            end_center = [(end_bbox[0] + end_bbox[2])/2, (end_bbox[1] + end_bbox[3])/2]
            
            # Calculate total distance and check if it's reasonable
            total_distance = np.sqrt((end_center[0] - start_center[0])**2 + 
                                (end_center[1] - start_center[1])**2)
            frames_in_gap = end_frame - start_frame
            
            # If the gap is too long or the distance too great, skip interpolation
            max_speed = 50  # pixels per frame - adjust based on your video
            if total_distance / frames_in_gap > max_speed and frames_in_gap > 15:
                print(f"Skipping unrealistic gap between frames {start_frame}-{end_frame} (speed: {total_distance/frames_in_gap:.1f} px/frame)")
                continue
            
            # Size of the ball (average of start and end)
            start_width = start_bbox[2] - start_bbox[0]
            start_height = start_bbox[3] - start_bbox[1]
            end_width = end_bbox[2] - end_bbox[0]
            end_height = end_bbox[3] - end_bbox[1]
            avg_width = (start_width + end_width) / 2
            avg_height = (start_height + end_height) / 2
            
            # Interpolate for each frame in the gap
            for frame in range(start_frame + 1, end_frame):
                # Normalized time (0 to 1)
                t = (frame - start_frame) / (end_frame - start_frame)
                
                # Linear interpolation for X position
                x = start_center[0] + t * (end_center[0] - start_center[0])
                
                # Ballistic interpolation for Y (parabolic trajectory)
                # The 4*h*t*(1-t) creates a parabola with max height h at t=0.5
                if frames_in_gap >= 10:  # Only apply ballistic model for longer gaps
                    h = min(30, frames_in_gap * 1.5)  # Adjust height based on gap length
                    y = start_center[1] + t * (end_center[1] - start_center[1]) - 4 * h * t * (1 - t)
                else:
                    # Short gaps use linear interpolation
                    y = start_center[1] + t * (end_center[1] - start_center[1])
                
                # Create bbox with consistent size
                bbox = [
                    x - avg_width/2, 
                    y - avg_height/2, 
                    x + avg_width/2, 
                    y + avg_height/2
                ]
                
                # Calculate distance from linear path (for confidence)
                linear_y = start_center[1] + t * (end_center[1] - start_center[1])
                deviation = abs(y - linear_y)
                
                # Calculate confidence based on position in gap and deviation
                # Frames closer to real detections have higher confidence
                edge_distance = min(frame - start_frame, end_frame - frame)
                edge_factor = edge_distance / ((end_frame - start_frame) / 2)
                confidence = max(0.1, 1 - edge_factor - (deviation / 100))
                
                interpolated_ball_positions[frame][track_id] = {
                    "bbox": bbox,
                    "is_detection": False,
                    "confidence": confidence
                }
        
        # Third pass: Filter out inconsistent velocities
        prev_center = None
        prev_frame = None
        
        for frame in range(len(interpolated_ball_positions)):
            if track_id in interpolated_ball_positions[frame]:
                bbox = interpolated_ball_positions[frame][track_id]["bbox"]
                center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                
                if prev_center is not None and prev_frame is not None:
                    # Calculate velocity
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    frames_elapsed = frame - prev_frame
                    speed = distance / frames_elapsed
                    
                    # If this is an interpolation (not a detection) and speed is too high, remove it
                    is_detection = interpolated_ball_positions[frame][track_id].get("is_detection", False)
                    if not is_detection and speed > max_speed:
                        if "confidence" in interpolated_ball_positions[frame][track_id]:
                            # Reduce confidence for suspicious interpolations
                            interpolated_ball_positions[frame][track_id]["confidence"] *= 0.5
                            
                            # If confidence is too low now, remove it
                            if interpolated_ball_positions[frame][track_id]["confidence"] < 0.3:
                                del interpolated_ball_positions[frame][track_id]
                
                # Only update previous position if this position wasn't removed
                if track_id in interpolated_ball_positions[frame]:
                    prev_center = center
                    prev_frame = frame
        
        # Final pass: Convert back to the original format (removing extra metadata)
        final_ball_positions = []
        for frame in range(len(interpolated_ball_positions)):
            frame_dict = {}
            if track_id in interpolated_ball_positions[frame]:
                frame_dict[track_id] = {"bbox": interpolated_ball_positions[frame][track_id]["bbox"]}
            final_ball_positions.append(frame_dict)
        
        print(f"Interpolated ball positions: {len([f for f in final_ball_positions if track_id in f])}/{len(final_ball_positions)} frames")
        
        return final_ball_positions