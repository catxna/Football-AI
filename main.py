import os
import cv2
import gc
import torch
from utils import read, save_vid
from trackers import Tracker
from ultralytics import YOLO

def get_next_filename(base_name, directory):
    """
    Automatically increments filename if it already exists.
    Example: RE_ID_1.avi → RE_ID_2.avi → RE_ID_3.avi
    """
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create output directory if it doesn't exist
    
    filename = f"{directory}/{base_name}1.avi"
    num = 1
    while os.path.exists(filename):
        num += 1
        filename = f"{directory}/{base_name}{num}.avi"
    return filename

def main():
    # Input video
    input_path = 'input_vids/pro_short.mp4'
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output setup
    output_dir = "output_vids"
    base_filename = "football_vid"
    output_path = get_next_filename(base_filename, output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize Tracker
    tracker = Tracker(
        model_path='models/best_pro.pt',
        reid_config_path='fast-reid/configs/Market1501/sbs_R101-ibn.yml',
        reid_model_path='models/market_sbs_R101-ibn.pth'
    )
    
    # Batch processing settings
    batch_size = 64  # Process 64 frames at a time (adjust based on GPU memory)
    frames = []
    processed_frames = 0
    
    print(f"Starting processing {total_frames} frames from {input_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames if any
            if frames:
                tracks = tracker.get_object_tracks(frames)
                # Ball interpolation is now handled inside get_object_tracks
                annotated_frames = tracker.draw_annotations(frames, tracks)
                for ann_frame in annotated_frames:
                    out.write(ann_frame)
                processed_frames += len(frames)
                print(f"Processed {processed_frames}/{total_frames} frames")
            break
        
        frames.append(frame)
        
        # Process batch when full or on last frame
        if len(frames) == batch_size:
            tracks = tracker.get_object_tracks(frames)
            # Ball interpolation is now handled inside get_object_tracks
            annotated_frames = tracker.draw_annotations(frames, tracks)
            for ann_frame in annotated_frames:
                out.write(ann_frame)
            
            processed_frames += len(frames)
            print(f"Processed {processed_frames}/{total_frames} frames")
            
            # Clear frames list and manage memory
            frames = []
            torch.cuda.empty_cache()  # Clear GPU memory after each batch
    
    # Cleanup
    cap.release()
    out.release()
    print(f"Processing complete. Video saved as: {output_path}")
    
    # Final memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()